import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.model_management as model_management
import comfy.utils
import folder_paths
import nodes


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _scalar(value, default=None):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


class FolderImageLoaderJStone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "tooltip": "图片文件夹完整路径；不递归子文件夹。"}),
                "image_limit": ("INT", {"default": 5, "min": 0, "max": 100000, "tooltip": "0 表示读取全部图片。"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "source_metadata")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load"
    CATEGORY = "Image Batch Timer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, directory, image_limit, start_index):
        folder = Path(directory.strip().strip('"')).expanduser()
        if not folder.is_dir():
            raise ValueError(f"图片文件夹不存在：{folder}")

        paths = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
            key=lambda path: path.name,
        )
        paths = paths[start_index:]
        if image_limit:
            paths = paths[:image_limit]
        if not paths:
            raise ValueError("指定范围内没有找到图片。")

        images = []
        metadata = []
        progress = comfy.utils.ProgressBar(len(paths))
        for batch_index, image_path in enumerate(paths):
            model_management.throw_exception_if_processing_interrupted()
            started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
            started = time.perf_counter()
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).unsqueeze(0)
            decode_seconds = round(time.perf_counter() - started, 6)
            images.append(tensor)
            metadata.append(json.dumps({
                "batch_index": batch_index,
                "source_index": start_index + batch_index,
                "image_name": image_path.name,
                "image_path": str(image_path.resolve()),
                "original_width": width,
                "original_height": height,
                "decode_seconds": decode_seconds,
                "decode_started_at": started_at,
            }, ensure_ascii=False))
            progress.update_absolute(batch_index + 1, len(paths))
        return (images, metadata)


class BatchTimerBase:
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("captions", "image_paths", "inference_seconds", "log_path")
    OUTPUT_IS_LIST = (True, True, True, False)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    CATEGORY = "Image Batch Timer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _prompt_dict(prompt):
        value = _scalar(prompt, {})
        return value if isinstance(value, dict) else {}

    @classmethod
    def _model_provenance(cls, prompt, unique_id, input_name, model):
        metadata = {
            "model_label": "未识别",
            "model_label_source": "unavailable",
            "model_source_node": "未识别",
        }
        try:
            prompt_value = cls._prompt_dict(prompt)
            node_id = str(_scalar(unique_id, ""))
            current = prompt_value.get(node_id, {})
            source_ref = current.get("inputs", {}).get(input_name)
            if isinstance(source_ref, (list, tuple)) and source_ref:
                source = prompt_value.get(str(source_ref[0]), {})
                source_type = source.get("class_type")
                if isinstance(source_type, str) and source_type:
                    metadata["model_source_node"] = source_type
                source_inputs = source.get("inputs", {})
                for key in ("model", "clip_name", "model_name", "ckpt_name", "checkpoint"):
                    label = source_inputs.get(key)
                    if isinstance(label, str) and label.strip():
                        metadata.update({
                            "model_label": label.strip(),
                            "model_label_source": "upstream_workflow",
                        })
                        return metadata
        except (AttributeError, TypeError):
            pass

        try:
            cached = getattr(getattr(model, "patcher", None), "cached_patcher_init", None)
            paths = cached[1][0] if cached and len(cached) > 1 and cached[1] else None
            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            if isinstance(paths, (list, tuple)) and paths:
                names = [Path(path).name for path in paths if isinstance(path, (str, os.PathLike))]
                if names:
                    metadata.update({
                        "model_label": ", ".join(names),
                        "model_label_source": "runtime_patcher",
                        "model_source_node": type(model).__name__,
                    })
        except (AttributeError, IndexError, TypeError):
            pass
        return metadata

    @staticmethod
    def _image_samples(images):
        samples = []
        for image in images:
            if not isinstance(image, torch.Tensor) or image.ndim != 4:
                raise ValueError("images 必须是 ComfyUI IMAGE 张量列表。")
            samples.extend(image[index:index + 1] for index in range(image.shape[0]))
        return samples

    @staticmethod
    def _metadata_rows(source_metadata):
        rows = []
        for value in source_metadata:
            if isinstance(value, dict):
                rows.append(value)
            elif isinstance(value, str):
                rows.append(json.loads(value))
            else:
                raise ValueError("source_metadata 格式无效。")
        return rows

    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _run_batch(self, images, source_metadata, model_type, provenance, parameters, infer):
        samples = self._image_samples(images)
        source_rows = self._metadata_rows(source_metadata)
        if len(samples) != len(source_rows):
            raise ValueError(f"缩放后图片数量（{len(samples)}）与路径元数据数量（{len(source_rows)}）不一致。")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        log_dir = Path(folder_paths.get_output_directory()) / "ComfyUI-ImageBatchTimer-JStone" / "image_batch_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.jsonl"
        captions, image_paths, durations = [], [], []
        progress = comfy.utils.ProgressBar(len(samples))

        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            for index, (image, source) in enumerate(zip(samples, source_rows)):
                model_management.throw_exception_if_processing_interrupted()
                self._sync_cuda()
                started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
                started = time.perf_counter()
                record = {
                    "batch_id": run_id,
                    "index": index,
                    **source,
                    "processed_width": int(image.shape[2]),
                    "processed_height": int(image.shape[1]),
                    "model_type": model_type,
                    **provenance,
                    **parameters,
                    "inference_started_at": started_at,
                    "timing_scope": "model_inference_only_after_external_resize",
                }
                try:
                    result = infer(image)
                    self._sync_cuda()
                    duration = round(time.perf_counter() - started, 6)
                    record.update({"status": "success", "inference_seconds": duration, "caption": result})
                    captions.append(result)
                    image_paths.append(source["image_path"])
                    durations.append(duration)
                except Exception as exc:
                    record.update({
                        "status": "error",
                        "inference_seconds": round(time.perf_counter() - started, 6),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                    raise
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                progress.update_absolute(index + 1, len(samples))
        return (captions, image_paths, durations, str(log_path))


class FlorenceBatchTimerJStone(BatchTimerBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_metadata": ("STRING", {"forceInput": True}),
                "florence2_model": ("FL2MODEL",),
                "task": (["more_detailed_caption", "detailed_caption", "caption"],),
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    FUNCTION = "run"

    def run(self, images, source_metadata, florence2_model, task, text_input, max_tokens,
            num_beams, prompt=None, unique_id=None):
        model = _scalar(florence2_model)
        if not isinstance(model, dict) or not {"patcher", "processor", "dtype"}.issubset(model):
            raise ValueError("florence2_model 必须连接 Florence2ModelLoader 的 FL2MODEL 输出。")
        task_value = _scalar(task, "more_detailed_caption")
        text_value = _scalar(text_input, "")
        max_tokens_value = int(_scalar(max_tokens, 512))
        num_beams_value = int(_scalar(num_beams, 3))
        provenance = self._model_provenance(prompt, unique_id, "florence2_model", model)
        parameters = {
            "task": task_value,
            "text_input": text_value,
            "precision": str(model["dtype"]).replace("torch.", ""),
            "max_tokens": max_tokens_value,
            "num_beams": num_beams_value,
            "do_sample": False,
            "seed": 1,
        }
        runner = nodes.NODE_CLASS_MAPPINGS["Florence2Run"]()

        def infer(image):
            return runner.encode(
                image, text_value, model, task_value, False,
                keep_model_loaded=True, num_beams=num_beams_value,
                max_new_tokens=max_tokens_value, do_sample=False,
                output_mask_select="", seed=1,
            )[2]

        return self._run_batch(images, source_metadata, "Florence-2", provenance, parameters, infer)


class QwenBatchTimerJStone(BatchTimerBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_metadata": ("STRING", {"forceInput": True}),
                "clip": ("CLIP",),
                "prompt_text": ("STRING", {"forceInput": True}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    FUNCTION = "run"

    def run(self, images, source_metadata, clip, prompt_text, max_tokens, temperature,
            top_k, top_p, min_p, repetition_penalty, seed, prompt=None, unique_id=None):
        model = _scalar(clip)
        prompt_value = str(_scalar(prompt_text, "")).strip()
        if not prompt_value:
            raise ValueError("prompt_text 不能为空；请从外部文本节点连接反推指令。")
        provenance = self._model_provenance(prompt, unique_id, "clip", model)
        inner_model = getattr(model, "cond_stage_model", None)
        signature = " ".join((
            provenance.get("model_label", ""),
            type(inner_model).__module__,
            type(inner_model).__qualname__,
        )).lower().replace("_", "").replace(".", "")
        if "qwen35" not in signature:
            raise ValueError("clip 输入必须连接加载 Qwen3.5 的 CLIPLoader。")

        max_tokens_value = int(_scalar(max_tokens, 512))
        sampling = {
            "sampling_mode": "on",
            "temperature": float(_scalar(temperature, 0.7)),
            "top_k": int(_scalar(top_k, 64)),
            "top_p": float(_scalar(top_p, 0.95)),
            "min_p": float(_scalar(min_p, 0.05)),
            "repetition_penalty": float(_scalar(repetition_penalty, 1.05)),
            "seed": int(_scalar(seed, 0)),
            "presence_penalty": 0.0,
        }
        parameters = {
            "task": "image_prompt_reverse",
            "prompt": prompt_value,
            "max_tokens": max_tokens_value,
            "sampling": sampling,
        }
        runner = nodes.NODE_CLASS_MAPPINGS["TextGenerate"]

        def infer(image):
            return runner.execute(
                model, prompt_value, max_tokens_value, sampling, image=image,
                thinking=False, use_default_template=True, mtp="auto",
            )[0]

        return self._run_batch(images, source_metadata, "Qwen3.5", provenance, parameters, infer)
