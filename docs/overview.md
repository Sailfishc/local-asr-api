# RAPL 项目概览

## 1. 项目简介

**RAPL**（Remote Audio Processing Layer）是一个在本地运行的语音转文字（ASR）API 服务。它的核心价值体现在三个方面：

- **本地运行，保护隐私**：音频文件在本机处理，不上传到任何云端服务器。
- **兼容 OpenAI Whisper API**：接口格式与 OpenAI 兼容，可直接替换云端 ASR 服务，无需修改客户端代码。
- **支持多个推理后端**：通过环境变量即可在 SenseVoice（FunASR）和 MLX 之间切换，无需改动代码。

典型使用场景：配合 Spokenly 等支持"OpenAI 兼容 API"的语音输入应用，将 RAPL 作为本地 ASR 后端。

---

## 2. 工作原理（数据流）

从客户端发出请求到收到转录文字的完整流程如下：

```
客户端（如 Spokenly）
  │
  │  POST /v1/audio/transcriptions（multipart 上传音频文件）
  ▼
FastAPI 路由层
  │
  ├─ 将音频写入临时目录 ./tmp/
  ├─ 计算音频时长（不依赖 torch，兼容 MLX 纯环境）
  ├─ 在后台线程中启动推理
  └─ 在主线程更新 tqdm 进度条
  │
  ├─ [SenseVoice 后端]
  │    用 torchaudio 加载音频 → 转为 float32 → 调用 FunASR AutoModel.generate()
  │    → 用 format_str_v3() 解析情绪/事件标签 → 返回纯文本
  │
  └─ [MLX 后端]
       直接读取临时文件路径 → 调用 mlx_audio.stt.generate() → 返回 result.text
  │
  ▼
返回 JSON：{ "text": "转录结果" }
  │
  ▼
客户端收到文字
```

---

## 3. 目录结构

```
local-asr-api/
├── openai_whisper_compatible_api.py   # 主入口：FastAPI 服务、路由、进度条、后端调度
├── api.py                             # 备用入口：简单版 FunASR 接口（不含 MLX 支持）
├── model.py                           # SenseVoice 模型架构定义（供 api.py 使用）
├── requirements.txt                   # Python 依赖列表
├── utils/
│   ├── frontend.py                    # 音频前端处理：Mel 频谱提取、CMVN 归一化
│   ├── infer_utils.py                 # 推理辅助：token 转换、双语文本后处理
│   └── ctc_alignment.py               # CTC 强制对齐（用于时间戳生成）
├── docs/
│   └── overview.md                    # 本文档
├── README.md                          # 英文说明
├── README_zh.md                       # 中文说明（含架构图和使用指南）
└── CHANGELOG.md                       # 变更记录与设计说明
```

---

## 4. 支持的后端

### SenseVoice（FunASR / ModelScope）

- **模型**：`iic/SenseVoiceSmall`（默认）
- **能力**：支持中、英、粤、日、韩多语言，可检测情绪（开心、悲伤、愤怒等）和音频事件（背景音乐、掌声、笑声等）。
- **推理框架**：FunASR `AutoModel`，模型缓存于 `~/.cache/modelscope/hub/`。
- **本地路径覆盖**：可通过 `SENSEVOICE_LOCAL_PATH` 和 `SENSEVOICE_VAD_PATH` 指定离线模型目录，避免每次从网络加载。
- **情绪/事件标签处理**：`format_str_v3()` 函数解析模型输出中的特殊标签（如 `<|HAPPY|>`、`<|BGM|>`），转换为 emoji 或去除。

### MLX（Apple Silicon 优化）

- **模型**：`mlx-community/Qwen3-ASR-1.7B-8bit`（默认）；也可选用轻量版 `mlx-community/Qwen3-ASR-0.6B-8bit`，推理速度更快，适合低资源或实时场景。
- **能力**：在 macOS Apple Silicon 上利用 MLX 框架加速推理，速度通常快于 SenseVoice。
- **推理框架**：`mlx-audio`，模型缓存于 `~/.cache/huggingface/hub/`。
- **特点**：不依赖 PyTorch，适合纯 MLX 环境；直接读取音频文件路径进行推理。

### 潜在后端：Fun-ASR-Nano（暂不支持 MLX）

Fun-ASR-Nano-2512 是阿里通义实验室于 2025 年 12 月发布的端到端 ASR 大模型（0.8B 参数），支持 31 种语言，针对远场和高噪声环境深度优化。

**架构（四组件流水线）：**

```
音频输入
  └─ 音频编码器（0.2B，Transformer Encoder）
       └─ 音频适配器（连接层，2 层 Transformer）
            ├─ CTC 解码器（第一遍粗识别，用于热词定制）
            └─ LLM 解码器（0.6B，基于 Qwen3）← 最终输出
```

**能否集成 MLX？**

理论上可行（LLM 解码器基于 Qwen3，而 Qwen3 已有 MLX 支持），但目前存在以下障碍：

| 障碍 | 说明 |
|------|------|
| 无 MLX 格式权重 | Hugging Face 上仅有 PyTorch 格式，`mlx-community` 尚未转换 |
| `mlx-audio` 未实现该架构 | 仅支持 Whisper 系列和 Qwen3-ASR 等标准模型 |
| 多组件联动 | 需逐一移植编码器、适配器、CTC 层，工作量远大于单模型转换 |

**与 Qwen3-ASR 的对比：**

```
Qwen3-ASR（已支持 MLX）
  音频 → 编码器 → LLM解码器 → 文字
  结构简单，mlx-community 已转换

Fun-ASR-Nano（尚不支持 MLX）
  音频 → 编码器 → 适配器 → CTC粗识别 ─┐
                                      ├→ LLM解码器 → 文字
                               热词库 ─┘
  多组件联动，转换复杂度高
```

**当前建议**：若不需要热词定制和多任务检测，直接使用 `mlx-community/Qwen3-ASR-0.6B-8bit`，参数量相近（0.6B），MLX 支持开箱即用。

---

## 5. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 服务信息（版本、当前后端、可用端点） |
| `GET` | `/health` | 健康检查，返回 `{"status": "healthy", ...}` |
| `GET` | `/v1/models` | 返回当前加载的模型列表（兼容 OpenAI 格式） |
| `POST` | `/v1/audio/transcriptions` | 核心转录接口，接收音频文件，返回文字 |
| `GET` | `/docs` | FastAPI 自动生成的 Swagger UI |

### `/v1/audio/transcriptions` 请求格式

- **Content-Type**：`multipart/form-data`
- **字段**：
  - `file`（必填）：音频文件（支持 WAV、MP3 等格式）
  - `language`（可选，默认 `"auto"`）：指定语言代码，如 `zh`、`en`、`ja`，或使用 `auto` 自动检测

### 响应格式

```json
{ "text": "转录结果文字" }
```

---

## 6. 配置方式（环境变量）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `BACKEND` | `sensevoice` | 推理后端，可选 `sensevoice` 或 `mlx` |
| `LOCAL_MODEL` | 取决于 BACKEND | 模型标识（HF 或 ModelScope ID） |
| `SENSEVOICE_LOCAL_PATH` | `~/.cache/modelscope/hub/models/iic/SenseVoiceSmall` | SenseVoice 主模型本地路径 |
| `SENSEVOICE_VAD_PATH` | `~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | SenseVoice VAD 模型本地路径 |

启动示例：

```bash
# 使用 MLX 后端
export BACKEND=mlx
export LOCAL_MODEL=mlx-community/Qwen3-ASR-1.7B-8bit
python openai_whisper_compatible_api.py

# 使用 SenseVoice 并指定离线模型路径
export BACKEND=sensevoice
export SENSEVOICE_LOCAL_PATH=/path/to/local/SenseVoiceSmall
python openai_whisper_compatible_api.py
```

---

## 7. 核心模块说明

### `openai_whisper_compatible_api.py`（主入口）

项目的核心文件，职责包括：

- **FastAPI 应用初始化**：定义所有 HTTP 端点。
- **后端加载**：启动时根据 `BACKEND` 环境变量调用 `_load_sensevoice()` 或 `_load_mlx()`，模型仅加载一次。
- **请求处理**：接收音频文件 → 写入临时目录 → 在线程池中执行推理（避免阻塞异步事件循环）。
- **进度条**：使用 `tqdm` + `asyncio` 并行更新进度，根据音频时长估算完成百分比。
- **标签后处理**（SenseVoice）：`format_str_v2` / `format_str_v3` 解析模型输出的语言、情绪、事件特殊标签。

### `api.py`（备用入口）

原始版 FunASR 接口，仅支持 SenseVoice，不含 MLX 和进度条。端点为 `POST /api/v1/asr`，接受多文件批量转录，返回包含 `raw_text`、`clean_text`、`text` 的详细结果。适合需要批量处理或原始标签输出的场景。

### `model.py`

定义 `SenseVoiceSmall` 模型类，供 `api.py` 直接实例化使用。包含模型架构、权重加载（`from_pretrained`）和推理方法（`inference`）。

### `utils/frontend.py`

音频前端处理模块，负责将原始波形转换为模型输入所需的特征：

- **Mel 频谱提取**：将时域音频转为频域 Mel 特征图。
- **CMVN 归一化**（Cepstral Mean and Variance Normalization）：对特征进行均值方差归一化，提升模型鲁棒性。

### `utils/infer_utils.py`

推理辅助模块：

- **Token 转换**：将模型输出的 token ID 序列解码为文字。
- **双语文本后处理**：处理中英文混合输出，合并或格式化双语段落。

### `utils/ctc_alignment.py`

CTC（Connectionist Temporal Classification）强制对齐模块，用于将识别出的文字与音频时间轴对齐，生成带时间戳的转录结果。主要供需要精确时间戳的场景使用。
