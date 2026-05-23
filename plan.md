# SSE 与 TF 融合方案整理

## 所有方案

1. **SSE + TF concat-linear 简单融合（当前）**
   - SSE 原始波形按 TF 的时间窗口方式切分，得到与 TF 对齐的时间 token。
   - 每个 SSE 窗口通过 `LayerNorm(200) -> Linear(200 -> 128) -> GELU -> Dropout` 压缩为 128 维。
   - SSE 和 TF 分别经过各自的特征提取模块后，在同一时间位置进行 concat。
   - concat 后通过 `LayerNorm + Linear(256 -> 128) + GELU + Dropout` 映射回统一维度。
   - 该方案作为当前优先实行方案，结构简单、参数量可控，适合作为第一版实验基线。

2. **SSE + TF 门控融合方案**
   - SSE 和 TF 分别提取特征后，由一个 gate 学习两类特征的融合比例。
   - 融合形式可以是：`gate * TF + (1 - gate) * SSE`。
   - 该方案允许模型根据不同样本、不同睡眠阶段动态选择更依赖频域信息还是原始波形信息。
   - 相比简单融合，表达能力更强，但需要额外参数和消融验证。

3. **SSE + TF Transformer Encoder 融合方案**
   - SSE 和 TF 分别提取特征后，将两类 token 组合起来送入一个 Transformer Encoder。
   - 可以将同一通道内的 TF token 和 SSE token 作为两组模态 token，让 Encoder 自动学习两者之间的关系。
   - 该方案交互能力最强，但计算量和训练难度也最高。
   - 建议在简单融合和门控融合验证有效后，再作为增强方案尝试。

# 当前方案详细内容

当前方案选择：**SSE + TF concat-linear 简单融合**。

该方案只改变 cross attention 之前的单通道特征构造方式。融合后的输出仍保持为 `[N, 3, 29, 128]`，用于兼容后续已有的三通道融合、multi-channel encoder 和分类器结构。

## 1. SSE 原始波形处理

每个 30 秒 epoch 的原始波形长度约为 3000 点，采样率为 100Hz。

为了与当前 TF 特征对齐，SSE 采用和 TF 相同的时间窗口方式：

```text
raw epoch: [3000]
-> 2 秒窗口，1 秒 overlap
-> 29 个窗口，每个窗口 200 个采样点
-> raw windows: [29, 200]
```

然后对每个窗口进行归一化、线性压缩、激活和 dropout：

```text
raw windows: [29, 200]
-> LayerNorm(200)
-> Linear(200 -> 128)
-> GELU
-> Dropout
-> SSE tokens: [29, 128]
```

注意事项：

- SSE 需要做数据级归一化，每个通道单独统计和处理。
- `LayerNorm(200)` 是模型内窗口级归一化，和数据级归一化作用不同，可以同时保留。
- 不使用 CNN，不将原始波形转换为图像。
- `Linear(200 -> 128)` 对同一通道内的所有时间窗口共享参数。
- 第一版中，三个通道各自使用独立的 SSE token 化模块，不共享参数。
- 与只取最大值相比，该方式能保留更多窗口内部的波形形态。

## 2. TF 特征处理

TF 分支保持当前已有处理方式：

```text
raw epoch
-> spectrogram / FFT
-> log amplitude
-> normalize
-> TF feature: [29, 128]
```

TF 特征本身已经与 SSE token 在时间维度和特征维度上对齐：

```text
TF feature:  [29, 128]
SSE tokens:  [29, 128]
```

输入数据结构需要明确区分 TF 和 SSE：

```text
TF input : [N, 3, 29, 128]
SSE input: [N, 3, 29, 128]
```

不要把 TF 和 SSE 直接堆叠成 `[N, 6, 29, 128]`，否则会把“模态维度”和“生理通道维度”混在一起，改变当前模型语义。

## 3. SSE 与 TF 特征提取

SSE 和 TF 分别进入各自的特征提取模块：

```text
TF feature [29, 128]
-> PositionalEncoding
-> TF Encoder
-> TF encoded feature [29, 128]

SSE tokens [29, 128]
-> PositionalEncoding
-> SSE Encoder
-> SSE encoded feature [29, 128]
```

注意事项：

- 第一版中，SSE Encoder 可以比 TF Encoder 更轻，避免模型复杂度过高。
- TF Encoder 和 SSE Encoder 第一版不共享参数。
- SSE 分支需要和 TF 分支一样加入位置编码，使 Transformer 能识别 29 个窗口的时间顺序。
- 三个通道保持独立特征提取结构，延续当前 baseline 中三个通道分别编码的设计。

## 4. SSE 与 TF concat-linear 简单融合

当前方案采用归一化、concat、Linear、激活和 dropout 的方式：

```text
TF encoded feature:  [29, 128]
SSE encoded feature: [29, 128]
-> TF LayerNorm(128)
-> SSE LayerNorm(128)
-> concat: [29, 256]
-> Linear(256 -> 128)
-> GELU
-> Dropout
-> fused feature: [29, 128]
```

该融合方式的优点：

- 比直接相加更灵活。
- 比门控融合更简单。
- 比 Transformer Encoder 融合更轻量。
- 输出维度仍为 `[29, 128]`，方便接入后续模型结构。

注意事项：

- 当前方案不是 element-wise add，统一命名为 `concat-linear 简单融合`。
- 第一版中，每个通道各自使用独立的 TF/SSE fusion module，不共享 `Linear(256 -> 128)` 参数。
- 暂不加入残差融合，避免把第一版融合模块复杂化。
- 如果后续需要对比，可单独增加 `TF + SSE element-wise add` 作为消融方案。

## 5. 实验顺序

建议按复杂度逐步验证：

1. 先实现并验证 **SSE + TF concat-linear 简单融合**。
2. 如果简单融合相对 TF-only baseline 有提升，再尝试 **门控融合**。
3. 如果门控融合仍有提升空间，再尝试 **Transformer Encoder 融合**。

当前计划只关注 SSE 与 TF 的融合，不讨论后续通道融合、cross attention 或分类器部分。

## 6. 当前模型数据流 ASCII 图

下面只画出 cross attention 之前的 SSE/TF 特征构造与融合流程。三个通道 `EEG_Fpz-Cz`、`EEG_Pz-Oz`、`EOG` 使用相同结构，但第一版参数彼此独立。

```text
                  One channel, one 30s epoch
                  ===========================

                            Raw waveform
                              [3000]
                                 |
                                 | 2s window, 1s overlap
                                 v
                         Raw windows [29, 200]
                                 |
                                 | data normalization
                                 | LayerNorm(200)
                                 | Linear(200 -> 128)
                                 | GELU
                                 | Dropout
                                 v
                         SSE tokens [29, 128]
                                 |
                                 | PositionalEncoding
                                 | SSE Transformer Encoder
                                 v
                    SSE encoded feature [29, 128]


                            Raw waveform
                              [3000]
                                 |
                                 | Spectrogram / FFT
                                 | Log amplitude
                                 | Normalize
                                 v
                         TF feature [29, 128]
                                 |
                                 | PositionalEncoding
                                 | TF Transformer Encoder
                                 v
                     TF encoded feature [29, 128]


          TF encoded feature [29, 128]      SSE encoded feature [29, 128]
                         |                                  |
                         | LayerNorm(128)                   | LayerNorm(128)
                         v                                  v
                  TF normalized                       SSE normalized
                         \                                  /
                          \                                /
                           \                              /
                            v                            v
                         concat on feature dimension [29, 256]
                                      |
                                      | Linear(256 -> 128)
                                      | GELU
                                      | Dropout
                                      v
                         fused channel feature [29, 128]


                 Three-channel output before cross attention
                 ===========================================

      EEG_Fpz-Cz fused [29, 128]
      EEG_Pz-Oz  fused [29, 128]     -> stack -> [3, 29, 128]
      EOG        fused [29, 128]

      Batch output:
      [N, 3, 29, 128]
```
