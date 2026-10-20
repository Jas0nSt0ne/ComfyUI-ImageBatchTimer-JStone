# ComfyUI-ImageBatchTimer-JStone

一组用于批量图片描述与逐图计时的 ComfyUI 自定义节点。插件从文件夹读取图片，调用 Florence-2 或 Qwen3.5 逐张生成描述，并将结果、图片信息和耗时写入本地 JSONL 日志。

## 节点

### Load Images From Folder

读取文件夹中的 PNG、JPG、JPEG 和 WebP 图片，按文件名升序输出图片及其来源元数据。

- `directory`：图片文件夹路径，仅扫描该文件夹，不递归子文件夹。
- `image_limit`：读取数量，设为 `0` 时读取指定起始位置之后的全部图片。
- `start_index`：从排序结果中的指定位置开始读取，索引从 `0` 开始。
- 输出 `images` 和 `source_metadata`，后者包含文件路径、原始尺寸和读取耗时等信息。

### Florence-2 Batch Timer

接收图片、来源元数据和 `FL2MODEL`，逐张运行 Florence-2。支持选择描述任务、输入文本以及设置最大 token 数和 beam 数。

### Qwen3.5 Batch Timer

接收图片、来源元数据和加载了 Qwen3.5 的 `CLIP`。通过外部 `STRING` 节点连接提示词，并可设置最大 token 数、采样参数和种子。

两个计时节点均输出描述、图片路径、逐图推理耗时及日志文件路径。处理出错时，节点会记录错误并停止当前批次。

## 安装与依赖

将本插件文件夹放入 ComfyUI 的 `custom_nodes` 目录，然后重启 ComfyUI。插件使用 ComfyUI 环境中的 PyTorch、NumPy 和 Pillow。

- 使用 Florence-2 节点时，需要另外安装提供 `Florence2ModelLoader` 和 `Florence2Run` 的 Florence-2 自定义节点，并准备兼容的模型。
- 使用 Qwen3.5 节点时，需要支持 Qwen3.5 和 `TextGenerate` 的 ComfyUI 版本，并准备兼容的模型。
- 如需在推理前缩放图片，可使用兼容的图片缩放节点；例如安装 KJNodes 后使用 `Resize Image v2`。

具体可用版本取决于 ComfyUI 和相关节点的接口兼容性。

## 使用方法

1. 添加 `Load Images From Folder`，填写图片文件夹路径并设置读取范围。
2. 将 `images` 连接到所选计时节点的 `images`；需要缩放时，在这条连接中间加入图片缩放节点。
3. 将读取节点的 `source_metadata` 直接连接到计时节点的同名输入。图片与元数据按列表顺序对应，因此中间的图片处理节点需要保持图片数量和顺序。
4. 连接对应的模型。使用 Qwen3.5 时，还需将包含反推指令的文本节点连接到 `prompt_text`。
5. 运行工作流，从计时节点获取描述、路径、耗时和日志路径。

读取节点会先将选定图片载入内存；处理大量或高分辨率图片时，可通过 `image_limit` 和 `start_index` 分批执行。

## 日志与计时

每次运行生成一个 UTF-8 JSONL 文件，默认位置为 ComfyUI 输出目录下的：

```text
ComfyUI-ImageBatchTimer-JStone/image_batch_logs/
```

每行对应一张图片，包含批次、文件名和完整路径、原始及处理后尺寸、模型信息、参数、状态和结果等。Qwen 记录还包括输入提示词；请在分享日志前检查其中的内容。

- `decode_seconds`：读取和解码图片的耗时。
- `inference_seconds`：从图片进入计时节点的模型调用开始，到模型返回并完成 CUDA 同步的耗时；不包括文件读取、外部缩放和日志写入。

计时结果受硬件、模型加载状态及运行环境影响，适合在相同条件下比较，不代表整个工作流的总耗时。
