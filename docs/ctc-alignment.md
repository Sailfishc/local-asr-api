# CTC 强制对齐：原理与实现

本文档以 `utils/ctc_alignment.py` 为线索，从问题背景出发，逐步拆解 CTC 强制对齐的完整实现逻辑。

---

## 问题背景：为什么需要"强制对齐"

普通 CTC 解码是**自由解码**：给模型一段音频，它输出最可能的文字序列，不管每个字对应哪些帧。

**强制对齐**反过来：已知音频和文字（两个都有了），要找出每个字**精确对应音频中的哪一帧**。结果是带时间戳的转录。

---

## 函数签名

```python
def ctc_forced_align(
    log_probs: torch.Tensor,      # [B, T, C]  每一帧上每个字符的对数概率
    targets: torch.Tensor,        # [B, L]     已知的目标文字序列（token ID）
    input_lengths: torch.Tensor,  # [B]        每个样本的实际音频帧数
    target_lengths: torch.Tensor, # [B]        每个样本的实际文字长度
    blank: int = 0,               # blank token 的 ID
    ignore_id: int = -1,          # 需要忽略的 token ID（会被替换为 blank）
) -> torch.Tensor                 # [B, T]     每一帧对应的 token ID
```

---

## 第一步：把目标序列扩展成"带 blank 的标准 CTC 序列"

```python
_t_a_r_g_e_t_s_ = torch.cat(
    (
        torch.stack((torch.full_like(targets, blank), targets), dim=-1).flatten(start_dim=1),
        torch.full_like(targets[:, :1], blank),
    ),
    dim=-1,
)
```

CTC 的规则要求在每个真实 token 前后都可以插入 blank。假设目标序列是 `[A, B, C]`，扩展后变成：

```
[blank, A, blank, B, blank, C, blank]
```

长度从 L 变成 2L+1。后续所有计算都在这个扩展序列上进行。

---

## 第二步：标记哪些位置允许"跳两步"

```python
diff_labels = torch.cat(
    (
        torch.as_tensor([[False, False]], device=targets.device).expand(batch_size, -1),
        _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2],
    ),
    dim=1,
)
```

CTC 对齐时，在扩展序列中，当前位置 `s` 的上一帧可以来自三个地方：

```
情况 1：还在同一个位置 s（重复）
情况 2：从前一个位置 s-1 过来（正常推进）
情况 3：从前两个位置 s-2 跳过来（跳过一个 blank）
```

但情况 3 只在 `s-2` 和 `s` 是**不同字符**时才合法（不能跳过两个相同字符之间的 blank，否则会把 `AA` 合并成 `A`）。`diff_labels` 就是这个合法性掩码。

---

## 第三步：动态规划——前向计算最优路径分数

```python
best_score[:, padding_num + 0] = log_probs[:, 0, blank]
best_score[:, padding_num + 1] = log_probs[bsz_indices, 0, _t_a_r_g_e_t_s_[:, 1]]

for t in range(1, input_time_size):
    prev = torch.stack(
        (best_score[:, 2:], best_score[:, 1:-1], torch.where(diff_labels, best_score[:, :-2], neg_inf))
    )
    prev_max_value, prev_max_idx = prev.max(dim=0)
    best_score[:, padding_num:] = log_probs[:, t].gather(-1, _t_a_r_g_e_t_s_) + prev_max_value
    backpointers[:, t, padding_num:] = prev_max_idx
```

这是标准的维特比（Viterbi）动态规划：

```
best_score[t][s] = log_probs[t][token_s] + max(
    best_score[t-1][s],    # 情况1：留在原位
    best_score[t-1][s-1],  # 情况2：从前一格来
    best_score[t-1][s-2]   # 情况3：跳两格（仅当 diff_labels 允许）
)
```

`backpointers` 记录每个 `(t, s)` 位置的最优前驱，用于后面回溯。

---

## 第四步：回溯——从终点找回最优路径

```python
# 找终点：最后一帧，扩展序列的倒数第一或第二个位置（末尾 blank 或最后一个字符）
path[bsz_indices, input_lengths - 1] = padding_num + target_lengths * 2 - 1 + l1l2.argmax(dim=-1)

# 从终点往回追
for t in range(input_time_size - 1, 0, -1):
    target_indices = path[:, t]
    prev_max_idx = backpointers[bsz_indices, t, target_indices]
    path[:, t - 1] += target_indices - prev_max_idx
```

从最后一帧沿着 `backpointers` 反向追踪，得到每一帧对应扩展序列中的哪个位置（是哪个字符还是 blank）。

---

## 第五步：把路径映射回真实 token

```python
alignments = _t_a_r_g_e_t_s_.gather(dim=-1, index=(path - padding_num).clamp(min=0))
```

`path` 记录的是扩展序列的下标，最终通过 `_t_a_r_g_e_t_s_` 映射回真实的 token ID，blank 帧对应 blank token。

---

## 整体数据流

```
输入：
  log_probs  [B, T, C]  — 每一帧上每个字符的对数概率
  targets    [B, L]     — 已知的目标文字序列（token ID）

      │
      ▼
扩展目标序列 [B, 2L+1]：  blank A blank B blank C blank

      │
      ▼
Viterbi 前向  [B, T, 2L+1]：
  每帧每个扩展位置的最优累计对数概率
  + backpointers 记录前驱

      │
      ▼
回溯  [B, T]：
  每帧对应扩展序列中的哪个位置

      │
      ▼
输出：
  alignments [B, T]  — 每一帧对应的 token ID（blank 帧 = 0）
```

---

## 如何从 alignments 得到时间戳

函数本身输出的是逐帧的 token 对齐，上层调用者通常这样用：

```python
# 找每个字符第一次出现的帧号，乘以帧移（通常 10ms/帧）就是开始时间
for token_id in unique_tokens:
    frames = (alignments == token_id).nonzero()
    start_time = frames[0].item() * 0.01  # 秒
    end_time   = frames[-1].item() * 0.01
```

---

## 一句话总结

把"音频帧序列"和"已知文字序列"之间的最优对齐路径找出来，记录每个字符精确出现在哪些帧。前向动态规划积累分数，`backpointers` 保存前驱，最后反向回溯还原完整路径。
