# openai_whisper_compatible_api.py 代码解析

## 一、整体结构

```
启动时
  └─ 读取环境变量（BACKEND / LOCAL_MODEL）
  └─ 加载模型（SenseVoice 或 MLX），只加载一次

运行时（每次收到请求）
  └─ POST /v1/audio/transcriptions
       ├─ 保存音频到临时文件
       ├─ 计算音频时长（用于进度条）
       ├─ 在线程池中运行推理（不阻塞主线程）
       ├─ 同时在主线程更新进度条
       └─ 返回 { "text": "..." }
```

---

## 二、启动阶段：配置与模型加载

### 环境变量读取

```python
BACKEND = os.getenv("BACKEND", "sensevoice").lower()  # 默认 sensevoice
LOCAL_MODEL = os.getenv("LOCAL_MODEL", ...)            # 默认随 BACKEND 自动选择
```

这两行在**服务启动时**执行一次，之后不再改变。

### 模型加载（只执行一次）

```python
if BACKEND == "mlx":
    mlx_model = _load_mlx()
else:
    sensevoice_model = _load_sensevoice()
```

模型是全局变量，整个服务生命周期只加载一次。加载完成后，后续每次请求直接复用，不重复下载或初始化。

---

## 三、两个后端的加载逻辑

### `_load_sensevoice()`

```
检查本地缓存路径是否存在
  ├─ 存在 → 从本地加载（离线，快）
  └─ 不存在 → 从 ModelScope 下载（首次使用）
```

使用 FunASR 的 `AutoModel`，同时加载两个模型：
- **主模型**（`iic/SenseVoiceSmall`）：负责语音识别
- **VAD 模型**（`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`）：Voice Activity Detection，先检测哪些片段有人声，再送入主模型，提升长音频处理效率

### `_load_mlx()`

```
调用 mlx_audio.stt.utils.load_model(model_id)
  └─ 首次：从 Hugging Face 下载到 ~/.cache/huggingface/hub/
  └─ 之后：直接读本地缓存
```

比 SenseVoice 简单，单一模型，无需 VAD。

---

## 四、请求处理：`POST /v1/audio/transcriptions`

这是核心端点，完整流程如下：

### 第 1 步：保存临时文件

```python
tmp_file = Path("./tmp") / filename
shutil.copyfileobj(fileobj, upload_file)
```

把客户端上传的音频写到 `./tmp/` 目录。请求结束后（无论成功还是报错）会在 `finally` 块中删除：

```python
finally:
    if tmp_file.exists():
        tmp_file.unlink()
```

### 第 2 步：计算音频时长

```python
duration_seconds = _get_audio_duration_no_torch(tmp_file)
```

专门写了一个**不依赖 torch** 的时长计算函数：
- `.wav` 文件：用标准库 `wave` 读取帧数和采样率，精确计算
- 其他格式：返回默认值 5.0 秒

这样做的原因：MLX 环境可能没有安装 torch，如果用 torchaudio 获取时长，MLX 用户会报错。

### 第 3 步：推理（在线程池中运行）

```python
inference_fut = loop.run_in_executor(None, run_inference)
```

`run_inference` 是一个普通（同步）函数，被放到**线程池**里运行。

为什么要放到线程池？
FastAPI 是异步框架（基于 asyncio），如果在主线程直接运行推理，会阻塞整个事件循环，导致服务在推理期间无法响应任何其他请求。放到线程池后，主线程可以继续处理事件。

`run_inference` 内部的分支：

```python
def run_inference():
    if backend == "mlx":
        # 直接传文件路径，MLX 自己读文件
        return model_inference(audio_path=str(tmp_file), language=language)
    else:
        # SenseVoice 需要先用 torchaudio 解码为波形数据
        waveform, sample_rate = torchaudio.load(tmp_file)
        ...
        return model_inference(input_wav=input_wav, language=language)
```

两个后端的输入格式不同：
- **MLX**：接受文件路径字符串，内部自己处理音频解码
- **SenseVoice**：接受已解码的波形数组 `(采样率, numpy数组)`

### 第 4 步：进度条（与推理并行运行）

```python
async def update_progress(fut):
    start = time.perf_counter()
    while not fut.done():
        await asyncio.sleep(0.1)          # 每 100ms 更新一次
        elapsed = time.perf_counter() - start
        n = min(99, int(elapsed / estimated_seconds * 100))
        pbar.n = n
        pbar.refresh()
```

进度条是一个**异步任务**，与推理并发运行：
- 根据音频时长估算总耗时
- 每 100ms 计算当前应到达的百分比
- 最多推进到 99%，等推理真正完成后才跳到 100%

这是纯粹的终端显示功能，不影响推理结果。

---

## 五、推理函数

### `model_inference()` — 统一入口

```python
def model_inference(audio_path=None, input_wav=None, language="auto", ...):
    if model == "mlx":
        return model_inference_mlx(audio_path=audio_path, language=language)
    else:
        return model_inference_sensevoice(input_wav=input_wav, ...)
```

### `model_inference_mlx()`

```python
result = mlx_model.generate(audio_path, language=lang)
return result.text
```

极简。注意对 `language` 做了保护：

```python
lang = "auto" if lang_str in ("auto", "") else lang_str
```

防止传入 `None` 导致 `.lower()` 崩溃。

### `model_inference_sensevoice()`

比 MLX 复杂，需要处理音频格式：

```python
# 如果传入的是 (采样率, numpy数组) 元组
if isinstance(input_wav, tuple):
    fs, input_wav = input_wav
    input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max  # 归一化到 [-1, 1]
    if fs != 16000:
        # 重采样到 16kHz（SenseVoice 要求的采样率）
        resampler = torchaudio.transforms.Resample(fs, 16000)
        input_wav = resampler(torch.from_numpy(input_wav)[None, :])[0, :].numpy()
```

推理完成后，输出的文本包含特殊标签（如 `<|zh|><|HAPPY|>你好`），需要用 `format_str_v3()` 清洗。

---

## 六、标签后处理（仅 SenseVoice）

SenseVoice 模型输出格式示例：
```
<|zh|><|HAPPY|><|Speech|>今天天气真好<|zh|><|NEUTRAL|><|Speech|>出去走走吧
```

### `format_str_v2()`

第一层处理：
1. 统计每个标签出现的次数
2. 选出出现最多的情绪标签，转为 emoji（如 `<|HAPPY|>` → `😊`）
3. 有音频事件（掌声、背景音乐等）则加对应 emoji 前缀
4. 去除所有特殊标签，返回干净文本

### `format_str_v3()`

第二层处理，处理多语言混合片段：
1. 把文本按语言标签（`<|zh|>` / `<|en|>` 等）切分成多个片段
2. 对每个片段调用 `format_str_v2()`
3. 合并时去除重复的情绪/事件符号（比如连续两段都是😊，只保留一个）

---

## 七、其他端点

| 端点 | 作用 |
|------|------|
| `GET /` | 返回服务信息：版本、当前后端、模型名、可用端点列表 |
| `GET /health` | 健康检查，返回 `{"status": "healthy", "backend": ..., "model": ...}` |
| `GET /v1/models` | 模拟 OpenAI `/v1/models` 格式，返回当前模型信息 |
| `GET /docs` | FastAPI 自动生成的 Swagger UI（框架内置，无需手写） |

---

## 八、关键设计决策总结

| 决策 | 原因 |
|------|------|
| 模型只加载一次 | 模型加载耗时长（几秒到几十秒），放在启动阶段，请求时直接推理 |
| 推理放线程池 | FastAPI 是异步框架，同步阻塞操作必须放到线程池，否则服务失去响应 |
| 时长计算不用 torch | MLX 环境可能无 torch，用标准库 `wave` 保证兼容性 |
| 临时文件用 finally 删除 | 确保即使推理报错，临时文件也不会堆积 |
| language 默认 "auto" | 客户端可能不传此字段，后端代码需防止 `None.lower()` 崩溃 |
