import importlib.util
import json
import sys
import tempfile
import time
import types
from pathlib import Path

from PIL import Image
import torch


root = Path(__file__).parent
tmp = tempfile.TemporaryDirectory()
base = Path(tmp.name)
images_dir = base / "中文图片"
images_dir.mkdir()
Image.new("RGB", (8, 9), "white").save(images_dir / "02_乙.png")
Image.new("RGB", (10, 7), "black").save(images_dir / "01_甲.png")

comfy = types.ModuleType("comfy")
management = types.ModuleType("comfy.model_management")
management.throw_exception_if_processing_interrupted = lambda: None
utils = types.ModuleType("comfy.utils")
utils.ProgressBar = lambda total: types.SimpleNamespace(update_absolute=lambda *args: None)
comfy.model_management = management
comfy.utils = utils
paths = types.ModuleType("folder_paths")
paths.get_output_directory = lambda: str(base / "output")
nodes = types.ModuleType("nodes")


class Florence:
    def encode(self, image, *args, **kwargs):
        assert image.shape == (1, 4, 4, 3)
        time.sleep(0.01)
        return None, None, "florence", None


class Qwen:
    @classmethod
    def execute(cls, clip, prompt, max_tokens, sampling, **kwargs):
        assert kwargs["image"].shape == (1, 4, 4, 3)
        assert prompt == "描述图片"
        assert sampling["seed"] == 0
        time.sleep(0.01)
        return ("qwen", "")


class QwenModel:
    def tokenize(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        pass

    def decode(self, *args, **kwargs):
        pass


nodes.NODE_CLASS_MAPPINGS = {"Florence2Run": Florence, "TextGenerate": Qwen}
sys.modules.update({
    "comfy": comfy,
    "comfy.model_management": management,
    "comfy.utils": utils,
    "folder_paths": paths,
    "nodes": nodes,
})
torch.cuda.is_available = lambda: False
spec = importlib.util.spec_from_file_location("jstone", root / "image_batch_timer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


loader = module.FolderImageLoaderJStone()
loaded_images, metadata_strings = loader.load(str(images_dir), 2, 0)
metadata = [json.loads(value) for value in metadata_strings]
assert [row["image_name"] for row in metadata] == ["01_甲.png", "02_乙.png"]
assert [(row["original_width"], row["original_height"]) for row in metadata] == [(10, 7), (8, 9)]
assert [tuple(image.shape) for image in loaded_images] == [(1, 7, 10, 3), (1, 9, 8, 3)]
assert all(row["decode_seconds"] >= 0 for row in metadata)
print("folder loader preserves filename order and path metadata")

# Simulate the external KJNodes Resize Image v2 output.
resized_images = [torch.zeros((1, 4, 4, 3)), torch.ones((1, 4, 4, 3))]

florence_model = {"patcher": object(), "processor": object(), "dtype": torch.float16}
florence_prompt = {
    "40": {"class_type": "ComfyUI-ImageBatchTimer-JStone-Florence2", "inputs": {"florence2_model": ["1", 0]}},
    "1": {"class_type": "Florence2ModelLoader", "inputs": {"model": "Florence-2-large-ft"}},
}
florence = module.FlorenceBatchTimerJStone()
captions, files, durations, log = florence.run(
    resized_images, metadata_strings, [florence_model], ["more_detailed_caption"], [""], [512], [3],
    prompt=[florence_prompt], unique_id=["40"],
)
rows = [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines()]
assert captions == ["florence", "florence"]
assert [Path(path).name for path in files] == ["01_甲.png", "02_乙.png"]
assert all(row["processed_width"] == 4 and row["processed_height"] == 4 for row in rows)
assert all(row["model_label"] == "Florence-2-large-ft" for row in rows)
assert all(row["timing_scope"] == "model_inference_only_after_external_resize" for row in rows)
assert all(duration >= 0.01 for duration in durations)
print("Florence typed runner preserves resize-to-path correspondence and timing")

qwen_prompt = {
    "41": {"class_type": "ComfyUI-ImageBatchTimer-JStone-Qwen35", "inputs": {"clip": ["1", 0]}},
    "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_4b_bf16.safetensors"}},
}
qwen = module.QwenBatchTimerJStone()
captions, files, durations, log = qwen.run(
    resized_images, metadata_strings, [QwenModel()], ["描述图片"], [512], [0.7], [64], [0.95], [0.05], [1.05], [0],
    prompt=[qwen_prompt], unique_id=["41"],
)
rows = [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines()]
assert captions == ["qwen", "qwen"]
assert [Path(path).name for path in files] == ["01_甲.png", "02_乙.png"]
assert all(row["model_label"] == "qwen3.5_4b_bf16.safetensors" for row in rows)
assert all(row["prompt"] == "描述图片" for row in rows)
print("Qwen typed CLIP runner accepts an external prompt and preserves correspondence")

schemas = {
    "loader": module.FolderImageLoaderJStone.INPUT_TYPES(),
    "florence": module.FlorenceBatchTimerJStone.INPUT_TYPES(),
    "qwen": module.QwenBatchTimerJStone.INPUT_TYPES(),
}
assert schemas["florence"]["required"]["florence2_model"][0] == "FL2MODEL"
assert schemas["qwen"]["required"]["clip"][0] == "CLIP"
assert schemas["qwen"]["required"]["prompt_text"][1]["forceInput"] is True
assert all("*" not in str(schema) for schema in schemas.values())
print("schemas are type-safe and contain no wildcard model socket")

ordinary_clip_prompt = {
    "41": {"class_type": "ComfyUI-ImageBatchTimer-JStone-Qwen35", "inputs": {"clip": ["1", 0]}},
    "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "clip_l.safetensors"}},
}
try:
    qwen.run(
        resized_images[:1], metadata_strings[:1], [QwenModel()], ["描述图片"], [512], [0.7], [64], [0.95], [0.05], [1.05], [0],
        prompt=[ordinary_clip_prompt], unique_id=["41"],
    )
    raise AssertionError("ordinary CLIP should be rejected")
except ValueError as error:
    assert "Qwen3.5" in str(error)
print("ordinary CLIP is not misidentified as Qwen3.5")
