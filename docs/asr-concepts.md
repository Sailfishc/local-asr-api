# ASR 概念层次指南

本文档从底层到上层逐层梳理 ASR（自动语音识别）相关概念，帮助读者建立完整的知识体系，理解 RAPL 项目中每个技术选型的背后逻辑。

---

## 第 0 层：音频基础

在模型看到任何数据之前，音频需要经历一系列预处理步骤。

### 采样率（Sample Rate）

声音是连续的模拟信号，计算机以固定间隔"拍照"来记录它，每秒拍照的次数就是**采样率**，单位是 Hz（赫兹）。

**为什么 ASR 标准是 16kHz？**

人类语音的有效频率范围约在 80 Hz ~ 8000 Hz 之间。根据奈奎斯特采样定理，要完整重建 8000 Hz 以内的信号，至少需要 16000 Hz（即 16kHz）的采样率。16kHz 在"捕获足够语音信息"和"文件体积合理"之间取得了平衡：

- 比 8kHz 电话音质更清晰，能区分 s/sh、n/l 等易混音素
- 比 44.1kHz CD 音质文件小很多，推理速度更快

### 波形 → Mel 频谱

原始音频是**波形**（时间 × 振幅的一维序列），神经网络直接处理它效率很低。实际工程中会将波形转换为 **Mel 频谱图**（Mel Spectrogram）：

```
原始波形（时间 × 振幅）
    │
    ├─ 短时傅里叶变换（STFT）→ 频谱图（时间 × 频率 × 能量）
    │
    └─ Mel 滤波器组 → Mel 频谱图（时间 × Mel 频段 × 能量）
```

**Mel 频段**是一种模拟人耳感知的非线性频率刻度，低频区分辨率高，高频区分辨率低——和人耳一样。这样转换后，模型输入的维度固定（例如 80 个 Mel 频段），同时保留了最关键的语音区分信息。

### CMVN 归一化

不同麦克风、不同录音环境会让音频的整体能量水平差异很大。**CMVN**（Cepstral Mean and Variance Normalization，倒谱均值方差归一化）对特征进行减均值、除方差操作，使模型输入在不同环境下保持一致的统计特性，提升鲁棒性。

> RAPL 中：`utils/frontend.py` 负责 Mel 频谱提取和 CMVN 归一化。

---

## 第 1 层：推理框架

模型训练好之后，需要一个**推理框架**来加载权重并运行计算。选择哪个框架直接决定了性能和兼容性。

### PyTorch

Meta AI 开源的通用深度学习框架，是目前 AI 研究领域的事实标准。

- **运行平台**：Linux / Windows / macOS，支持 CPU 和 NVIDIA GPU（CUDA）
- **生态**：绝大多数公开模型（SenseVoice、Whisper 原版、FunASR 系列）都以 PyTorch 格式发布
- **在 Apple Silicon 上的局限**：PyTorch 通过 MPS（Metal Performance Shaders）后端支持 Apple GPU，但利用率不如 MLX 充分，统一内存优势也无法完全发挥

### MLX

Apple 于 2023 年底开源的机器学习框架，**专为 Apple Silicon 设计**。

- **统一内存架构**：CPU 和 GPU 共享同一块物理内存，数据无需在 CPU ↔ GPU 之间复制，延迟极低
- **懒惰求值**：计算图只在需要结果时才真正执行，方便自动优化
- **在 Mac 上更快**：对于中等规模模型（0.6B ~ 8B 参数），MLX 推理速度通常比 PyTorch MPS 快 2~4 倍

**两者的关键区别**：PyTorch 权重和 MLX 权重**不兼容**，必须分别转换。一个模型要在 MLX 上运行，必须有人专门将 PyTorch 权重转换为 MLX 格式（`.safetensors` + 配置文件）并发布。

---

## 第 2 层：核心 ASR 技术

### VAD（Voice Activity Detection，语音活动检测）

VAD 是 ASR 流水线的**第一道过滤器**，负责检测音频中哪些片段包含人声，哪些是静音或噪音。

**解决的问题**：长音频中大量片段可能是静默、背景噪音或非语音声音。让主 ASR 模型处理这些片段既浪费计算资源，又可能产生幻听（把噪音识别成文字）。

**工作原理**：VAD 模型通常是轻量级的二分类器（有声 / 无声），处理速度远快于主 ASR 模型。它先扫描全文，切分出若干"有声片段"，再将这些片段依次送入主模型。

> RAPL 中：SenseVoice 后端集成了 FunASR 的 VAD 模型（`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`），通过 `SENSEVOICE_VAD_PATH` 配置路径。

### CTC（Connectionist Temporal Classification）

CTC 解决的是 ASR 中的核心难题：**输入帧数（几百帧）远多于输出字符数（几十个字），如何对齐？**

传统方案需要人工标注每个音频帧对应哪个字符，成本极高。CTC 引入了一个特殊的空白符（blank），允许模型在每一帧输出"当前对应的字符"或"空白"，然后通过合并连续相同字符、删除空白的规则得到最终文本。训练时只需要音频和最终文字，不需要帧级对齐标注。

**适用场景**：CTC 是流式识别（边录边出字）的首选，因为它允许逐帧输出，不需要等待整段音频结束。

> RAPL 中：`utils/ctc_alignment.py` 实现了 CTC 强制对齐，用于生成带时间戳的转录结果。Fun-ASR-Nano 也在流水线中使用 CTC 进行第一遍粗识别。

### Encoder-Decoder 架构

经典的序列到序列（seq2seq）架构，分为两个阶段：

```
音频帧序列
    │
    ▼
Encoder（编码器）
    │  理解音频，提取高维语义表示
    │  输出：上下文向量
    ▼
Decoder（解码器）
    │  逐字生成文本，每步参考 Encoder 输出和已生成的字符
    ▼
文字序列
```

**优势**：Decoder 可以利用语言模型能力纠错，生成更自然的文本。
**劣势**：必须等 Encoder 处理完整段音频才能开始解码，不适合实时流式场景。

Whisper 是这一架构的代表：其 Encoder 是 Transformer，Decoder 也是 Transformer，训练数据达 68 万小时。

### LLM-based Decoder（大语言模型解码器）

新一代 ASR 架构用**大语言模型**替代传统 Transformer Decoder：

```
音频帧序列
    │
    ▼
音频编码器（提取声学特征）
    │
    ▼
（可选）适配器（连接层，对齐音频和文本的特征空间）
    │
    ▼
LLM 解码器（如 Qwen3、LLaMA）
    │  具备强大的语言理解和生成能力
    ▼
文字序列
```

**优势**：LLM 具备更强的语言理解能力，能更好地处理口语化表达、专有名词、上下文歧义；可以通过提示词（prompt）灵活控制输出格式，甚至支持热词注入。
**代表模型**：Qwen3-ASR（编码器 + Qwen3 LLM）、Fun-ASR-Nano（编码器 + 适配器 + CTC + Qwen3 LLM）。

---

### 常见问题：为什么有些模型需要 VAD，有些不需要？为什么有些支持热词，有些不支持？

这两个问题背后都是同一个逻辑：**模型的架构决定了它能处理什么范围的输入，以及输出时能引入什么外部知识。**

#### 关于 VAD：输入侧的问题

根本原因是**模型的上下文窗口有限**。Whisper 和 Qwen3-ASR 这类模型内部接受的音频长度有上限（Whisper 是 30 秒一段），无法直接处理几分钟的长音频，必须先将音频切成小块再送进去。VAD 就是做这个切割工作的——它找到静音边界，沿着自然停顿切片，避免从句子中间断开。

`mlx-audio` 调用 Qwen3-ASR 时，内部其实也在做类似的事，只是封装在库里看不见。SenseVoice 之所以显式地配置一个 VAD 模型，是因为 FunASR 的 `AutoModel` 把"切割"这一步暴露出来，让用户可以单独配置、替换甚至关掉它——这是工程上的设计选择，不是说 SenseVoice 比 Qwen3-ASR 更"需要"VAD。

**结论**：VAD 不是某些模型特有的需求，而是所有模型在处理长音频时都需要解决的问题。区别只在于这个步骤是藏在库内部还是暴露给用户配置。

#### 关于热词：输出侧的问题

这取决于**解码阶段能不能引入外部知识**，不同解码机制的能力差异很大：

**传统 Encoder-Decoder（如 Whisper）**：解码器每一步根据概率分布选下一个 token，这个分布完全由模型权重决定。要插入热词，只能用"浅层融合"——在每步解码时手动给热词 token 加分。这种方式不稳定，效果有限，Whisper 官方不支持。

**CTC 解码器**：CTC 解码时会产生一个"帧-字符"的概率矩阵，热词可以以有限状态自动机（FSA）的形式叠加在解码过程中，强制提升特定词的路径得分。这是 Fun-ASR-Nano 的做法：CTC 先做一遍粗识别负责把热词"钉"进去，再把结果传给 LLM 解码器做最终润色。

**LLM Decoder（如 Qwen3-ASR）**：LLM 解码器本来是最灵活的，可以在 prompt 里直接写"请优先识别以下词汇：XXX"。但 Qwen3-ASR 目前的 `mlx-audio` 接口没有暴露这个能力，所以用起来感觉它"不支持热词"——这是**接口的限制**，不是模型本身做不到。

**结论**：热词支持是输出侧解码机制的设计问题。CTC 天然适合热词约束，传统 Encoder-Decoder 很难做好，LLM Decoder 理论上最灵活但需要接口配合。

#### 四个模型的对比小结

```
能力            Whisper    SenseVoice    Qwen3-ASR      Fun-ASR-Nano
───────────────────────────────────────────────────────────────────
长音频切割       库内部做    显式 VAD 模型  库内部做        显式 VAD 模型
热词支持         ✗（难）    ✗             接口未暴露       ✓（CTC 阶段）
```

---

## 第 3 层：具体模型对比

| 模型 | 架构 | 推理框架 | 参数量 | 特色 |
|------|------|----------|--------|------|
| **Whisper** | Encoder-Decoder（Transformer） | PyTorch（官方）/ MLX（社区转换） | 39M ~ 1.5B（多规格） | OpenAI 出品；定义了 `/v1/audio/transcriptions` API 标准；多语言；开箱即用 |
| **SenseVoice** | 多任务 Encoder（ASR + 情绪 + 事件 + VAD） | PyTorch / FunASR | ~25M（Small） | 阿里达摩院出品；一次推理同时输出转录文字、情绪标签、音频事件；5 语言 |
| **Qwen3-ASR** | 音频编码器 + Qwen3 LLM Decoder | MLX（mlx-audio） | 0.6B / 1.7B | 阿里通义出品；轻量，Mac 上开箱即用；mlx-community 已发布转换版本 |
| **Fun-ASR-Nano** | 编码器 + 适配器 + CTC + LLM Decoder（Qwen3） | PyTorch / FunASR | ~0.8B | 阿里通义 2025 年底发布；31 语言；热词定制；远场高噪声优化；目前无 MLX 支持 |

### Whisper

OpenAI 于 2022 年发布，是第一个真正意义上的"通用多语言开源 ASR 模型"。它定义了业界标准的 API 格式（`POST /v1/audio/transcriptions`），RAPL 兼容的正是这套接口。Whisper 有从 tiny（39M）到 large（1.5B）的多个规格，mlx-community 已将各规格转换为 MLX 格式。

### SenseVoice

阿里达摩院发布，核心特点是**多任务**：单次推理同时完成 ASR、情绪识别（开心、悲伤、愤怒等）和音频事件检测（背景音乐、掌声、笑声等）。模型输出包含特殊标签（如 `<|HAPPY|>`、`<|BGM|>`），需要 `format_str_v3()` 等后处理函数解析。

### Qwen3-ASR

阿里通义实验室发布，采用"音频编码器 + Qwen3 LLM"的端到端架构。mlx-community 已发布 8bit 量化版本（0.6B 和 1.7B），可直接通过 `mlx-audio` 加载。在 Mac 上无需 PyTorch，是 RAPL MLX 后端的当前默认模型。

### Fun-ASR-Nano

阿里通义实验室于 2025 年 12 月发布，架构更复杂：

```
音频输入
  └─ 音频编码器（~0.2B，Transformer Encoder）
       └─ 音频适配器（连接层，2 层 Transformer）
            ├─ CTC 解码器（第一遍粗识别，用于热词定制）
            └─ LLM 解码器（~0.6B，基于 Qwen3）← 最终输出
```

CTC 解码器的粗识别结果会与用户提供的热词库结合，引导 LLM 解码器更准确地输出特定词汇（人名、术语等）。目前仅有 PyTorch 格式，MLX 适配工作量大，RAPL 暂不支持。

---

## 第 4 层：框架与生态

### FunASR（`AutoModel`）

阿里开源的 ASR 工具箱，提供统一的模型加载和推理接口。

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
)
result = model.generate(input="audio.wav", language="auto")
```

`AutoModel` 屏蔽了不同模型的架构差异，支持从 ModelScope 或 Hugging Face 自动下载权重。SenseVoice 和 Fun-ASR-Nano 都通过 FunASR 接口加载。

> RAPL 中：SenseVoice 后端在 `openai_whisper_compatible_api.py` 的 `_load_sensevoice()` 函数中调用 FunASR `AutoModel`。

### mlx-audio

MLX 生态的音频推理库，封装了 Whisper 系列和 Qwen3-ASR 等模型的 MLX 推理逻辑。

```python
import mlx_audio.stt as stt

result = stt.generate(
    audio_path="audio.wav",
    model="mlx-community/Qwen3-ASR-0.6B-8bit",
)
print(result.text)
```

它直接接收音频文件路径，内部处理采样率转换、特征提取和模型推理，对上层调用者完全透明。

> RAPL 中：MLX 后端在 `_load_mlx()` 和推理函数中调用 `mlx_audio.stt.generate()`。

### mlx-community（Hugging Face 社区）

Hugging Face 上的一个开源社区（`huggingface.co/mlx-community`），专门将 PyTorch 格式的模型权重转换为 MLX 格式，并发布在 Hugging Face Hub 上。

**转换流程（简化）**：

```
PyTorch 权重（.bin / .safetensors）
    │
    └─ mlx_lm.convert / 自定义转换脚本
         │  重新排列权重布局，调整数值精度（如 8bit 量化）
         ▼
MLX 权重（.safetensors）+ 配置文件（config.json）
    │
    └─ 发布到 mlx-community/xxx 仓库
```

Qwen3-ASR 的 MLX 版本（`mlx-community/Qwen3-ASR-0.6B-8bit` 等）正是通过这一流程产生的。

---

## 第 5 层：RAPL 如何串联以上内容

### 整体关系图

```
┌─────────────────────────────────────────────────────┐
│                   客户端应用                         │
│   （Spokenly 等支持 OpenAI 兼容 API 的语音输入工具） │
└───────────────────┬─────────────────────────────────┘
                    │ POST /v1/audio/transcriptions
                    │（Whisper API 标准 — 第 3 层）
                    ▼
┌─────────────────────────────────────────────────────┐
│                RAPL FastAPI 服务                     │
│         openai_whisper_compatible_api.py             │
│                                                     │
│  音频接收 → 临时存储 → 时长计算 → 进度条 → 后端调度  │
└──────────────┬────────────────────┬─────────────────┘
               │                    │
    ┌──────────▼──────────┐  ┌──────▼──────────────────┐
    │   SenseVoice 后端   │  │      MLX 后端            │
    │                     │  │                          │
    │  第 4 层：FunASR    │  │  第 4 层：mlx-audio      │
    │  AutoModel          │  │  stt.generate()          │
    │    │                │  │    │                     │
    │    ▼                │  │    ▼                     │
    │  第 3 层：          │  │  第 3 层：               │
    │  SenseVoice         │  │  Qwen3-ASR               │
    │  (多任务 ASR)       │  │  (编码器 + LLM)          │
    │    │                │  │    │                     │
    │    ▼                │  │    ▼                     │
    │  第 2 层：VAD       │  │  第 2 层：               │
    │  第 1 层：PyTorch   │  │  LLM-based Decoder       │
    │  第 0 层：16kHz     │  │  第 1 层：MLX            │
    │          Mel 频谱   │  │  第 0 层：16kHz          │
    │          CMVN 归一化│  │          Mel 频谱        │
    └─────────────────────┘  └──────────────────────────┘
               │                    │
               └──────────┬─────────┘
                          ▼
               { "text": "转录结果" }
```

### 各层在 RAPL 中的对应

| 层次 | 概念 | RAPL 中的位置 |
|------|------|--------------|
| 第 0 层 | 采样率、Mel 频谱、CMVN | `utils/frontend.py`（SenseVoice）；mlx-audio 内部处理（MLX） |
| 第 1 层 | PyTorch / MLX | SenseVoice 后端用 PyTorch；MLX 后端用 MLX |
| 第 2 层 | VAD | SenseVoice 的 FunASR VAD 模型 |
| 第 2 层 | CTC | `utils/ctc_alignment.py`（时间戳）；Fun-ASR-Nano 内部（如将来集成） |
| 第 2 层 | LLM-based Decoder | Qwen3-ASR 的 LLM 解码层（MLX 后端） |
| 第 3 层 | Whisper | 定义了 RAPL 兼容的 API 标准 |
| 第 3 层 | SenseVoice | RAPL SenseVoice 后端 |
| 第 3 层 | Qwen3-ASR | RAPL MLX 后端（当前默认） |
| 第 3 层 | Fun-ASR-Nano | RAPL 潜在后端（暂不支持，见 overview.md） |
| 第 4 层 | FunASR AutoModel | `openai_whisper_compatible_api.py` → `_load_sensevoice()` |
| 第 4 层 | mlx-audio | `openai_whisper_compatible_api.py` → `_load_mlx()` |
| 第 4 层 | mlx-community | 提供 Qwen3-ASR MLX 权重的来源 |

### 一句话总结

RAPL 是一个**胶水层**：它向上暴露标准的 Whisper 兼容 API，向下按需调度 FunASR 或 mlx-audio，将第 0~4 层的复杂性全部封装，让客户端应用只需一次 HTTP 请求即可完成本地 ASR 推理。
