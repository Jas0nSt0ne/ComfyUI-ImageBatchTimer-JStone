from .image_batch_timer import (
    FlorenceBatchTimerJStone,
    FolderImageLoaderJStone,
    QwenBatchTimerJStone,
)


NODE_CLASS_MAPPINGS = {
    "ComfyUI-ImageBatchTimer-JStone-FolderLoader": FolderImageLoaderJStone,
    "ComfyUI-ImageBatchTimer-JStone-Florence2": FlorenceBatchTimerJStone,
    "ComfyUI-ImageBatchTimer-JStone-Qwen35": QwenBatchTimerJStone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-ImageBatchTimer-JStone-FolderLoader": "Load Images From Folder",
    "ComfyUI-ImageBatchTimer-JStone-Florence2": "Florence-2 Batch Timer",
    "ComfyUI-ImageBatchTimer-JStone-Qwen35": "Qwen3.5 Batch Timer",
}
