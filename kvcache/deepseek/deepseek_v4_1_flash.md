# DeepSeek-V4.1-Flash(552B-A16B)：CSA2 + CED w/ Top512

模型不含 MTP 共计 40 层：`2 SWA(w=128) + 18 CSA2(m=2)/SWA(w=128) + 20 CSA2(m=1)/SWA(w=128)`

> 注意：V4.1 与 V4 结构有差异，CSA(m=4)+HCA(m=128) 改为两级 CSA2(m=2)+CSA2(m=1)，且引入 CED 跨层复用

> CED 跨层复用：主 KV 仅 `kv_source_layers=[2,8,14,20]` 4 层存储（3 层 m=2 + 1 层 m=1），其余 CSA2 层复用最近源层缓存；Indexer K 仅 `index_source_layers=[2,8,14,20,24,28,32,36]` 8 层参与打分，其中 2/8/14/20 为 Full 层全量扫描，24/28/32/36 为 Reindex 层复用层 20 的 K 与 16384 候选池（2048 块 x 8）

## 一、缓存容量 (Bytes)

**128K 序列 = 131072 × 890 B + 2.70336 MB = 119.35744 MB**

### 1.1 Main KV 缓存 (CSA2/SWA) = 288 / 528 Bytes/token

压缩 Main KV 缓存 `[b, seq, kv_lora(512)]` NVFP4 (QAT 后训练量化) = 256B FP4 + 32B scale = 288B/token

> 精度说明 (论文 §2.4.4)：512 维 FP4 E2M1 + 每 16 通道 1 个 FP8 E4M3 scale（NVFP4 省略二级全局 scale）= `512 x 0.5 + 32 x 1 = 288 Bytes`；post-RoPE 量化，NoPE/RoPE 同格式

> SWA 窗口 KV 对量化敏感，论文保留 FP8：全 512 维（含 RoPE）FP8 E4M3 + 16 个 UE8M0 scale（每 32 维）= `512 + 16 = 528 Bytes`

> 备注 (vLLM 部署)：vLLM 未实现 NVFP4 record；SM100 写 MXFP8 record = 528B，其他架构回退 V4 record 584B (448 FP8 NoPE + 128B RoPE BF16 + 8B scale)

### 1.2 Indexer K 缓存 (CSA2 源层) = 68 Bytes/token

Indexer K 缓存 `[b, seq, 1, idx_dim(128)]` MXFP4 = `64 packed values + 4 UE8M0 scales = 68 Bytes`, 128 通道每 32 通道 1 个 UE8M0 scale

> Indexer 部分 `Hq=32`, `Hk=1`，因此 cache 的 k_idx 是单头的 MQA 模式；Q/K 均为 FP4 QAT，存储和计算都是 FP4

> 备注 (vLLM 部署)：MXFP4 需 `indexer_kv_dtype="mxfp4"` opt-in + SM100，且两级 sparse (Full+Reindex) 路径强制要求；vLLM 默认 dense 路径走 FP8 = `128 fp8 + 4 fp32 scale = 132 Bytes`

### 1.3 整模型 Global KV Cache Per Token（CED 源层 Main KV + Indexer K）

> 注意：DeepSeek 论文中将 SWA 的 128 token 的 KV 设定为 Local KV Cache，因此 1.3 不包含 SWA

```shell
# 1) 编码器 CSA2 Main KV (3 源层, m=2)
3 Layers × 288B/token ÷ 2 compress = 432 Bytes/token

# 2) 解码器 CSA2 Main KV (1 源层, m=1)
1 Layer × 288B/token ÷ 1 compress = 288 Bytes/token

# 3) 编码器 Indexer K (3 源层, m=2) in MXFP4
3 Layers × 68B/token ÷ 2 compress = 102 Bytes/token

# 4) 解码器 Indexer K (1 源层, m=1) in MXFP4
1 Layer × 68B/token ÷ 1 compress = 68 Bytes/token

# 合计
432 + 288 + 102 + 68 = 890 Bytes/token
```

### 1.4 整模型 Local KV Cache (SWA window = 128)

固定窗口 128 与序列长度无关

```shell
# SWA Window KV (FP8)
40 Layers x 128 window x 528B/token = 2.70336 MB
```

> 注意 (论文 §3.2，已部署)：SWA KV 不写入持久化缓存，仅存于 host DRAM 内存池（约 10% DRAM，分钟级 TTL），Global KV 才持久化（≥72 小时）；Encoder SWA 缺失时 Bounded Replay 仅重放尾部 n_win=128 token 重建 SWA（复用 Global KV 不重算）；Decoder SWA 从不缓存，每次 prefill 对尾部 128 token 做 bounded forward 重新生成（见第四节）；持久化 KV 因此降至 V4 的 1/8（去 SWA ≈ 减半 × Global 压至 1/4）。

## 二、访存量 (Bytes)

**128K 序列 = 5.308416 + 13.369344 + 8.912896 + 4.456448 + 2.70336 = 34.750464 MB**

- CSA2 Main KV：`38 Layers x Top(512) x 288B/token = 5.308416 MB`，这里是压缩后的 Top512 所以不需要除压缩比

- CSA2 Indexer K (Encoder Full)：`3 Layers x 131072 x 68B/token ÷ 2 compress = 13.369344 MB` 打分也是在压缩后KV段上进行

- CSA2 Indexer K (Decoder Full, 层 20)：`1 Layer x 131072 x 68B/token = 8.912896 MB`，m=1 不压缩全量扫描

- CSA2 Indexer K (Reindex 4 层)：`4 Layers x 16384 candidates x 68B/token = 4.456448 MB`，仅扫固定候选池 (2048 块 x 8)，常数开销

- SWA Local KV：`40 Layers x 128 window x 528B/token = 2.70336 MB`

## 三、Decoder 计算量 (FLOPs)

**128K 序列 = 16G + 2.55G + 0.671G + 131072 x 5120 Flops/token + 0.134G = 20.027 GFLOPs**

> 论文口径：DeepSeek-V4.1 Tech Report 的成本模型按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算

### 3.1 激活权重 GEMM

模型激活参数 A16B，MOE 采用 FP8 激活 + MXFP4 (FP4 E2M1) 权重即 W4A8 格式，但实际计算采用 FP8

计算量：`16B x 2 x FP8(0.5) = 16 GFlops`，Decoder `seq_q=1` 因此是常量

> 备注：论文中 MOE 按权重精度记 `FP4=0.25`, 按该口径激活 `10B x 2 x 0.25 + 5.1B x 2 x 0.5 = 10.1 GFlops`

### 3.2 CSA2 Attention

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，FP4 仅存储（attention 前反量化），论文 §2.4.4 有明确说明

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, kv_lora(512)]
k.shape = [b, top(512), kv_lora(512)]
p.shape = [b, Hq(64), seq_q, top(512)] # P = Q @ K.T
v.shape = [b, top(512), kv_lora(512)]
o.shape = [b, Hq(64), seq_q, kv_lora(512)] # O = P @ V
```

计算量如下，是常量，不随序列长度变化

```python
# P = Q @ K.T
38 Layers x Hq(64) x seq_q(1) x top(512) x kv_lora(512) x 2 x BF16(1) = 1275.068416 MFlops
# O = P @ V
38 Layers x Hq(64) x seq_q(1) x top(512) x kv_lora(512) x 2 x BF16(1) = 1275.068416 MFlops

# 合计
1275.068416 + 1275.068416 = 2550.136832 MFlops
```

### 3.3 SWA Attention

固定窗口 128 且不压缩，计算量是常量，不随序列长度变化

```python
# P = Q @ K.T
40 Layers x Hq(64) x seq_q(1) x seq_k(128) x kv_lora(512) x 2 x BF16(1) = 335.54432 MFlops
# O = P @ V
40 Layers x Hq(64) x seq_q(1) x seq_k(128) x kv_lora(512) x 2 x BF16(1) = 335.54432 MFlops

# 合计
335.54432 + 335.54432 = 671.08864 MFlops
```

### 3.4 CSA2 Indexer

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP4；Full 层计算量是变量随序列长度变化，Reindex 层仅扫候选池是常量

```python
# 输入输出 Shape
q_idx.shape = [b, idx_head(32), seq_q, idx_dim(128)]
k_idx.shape = [b, seq_k, idx_dim(128)]

# Full 层每 token 计算量 (3 Encoder m=2 + 1 Decoder m=1)
3 Layers x idx_head(32) x seq_q(1) x seq_k(?) / compress(2) x idx_dim(128) x 2 x FP4(0.25) = 3072 Flops/token
1 Layer  x idx_head(32) x seq_q(1) x seq_k(?) / compress(1) x idx_dim(128) x 2 x FP4(0.25) = 2048 Flops/token

# Reindex 层常量 (4 层仅扫 16384 候选池)
4 Layers x idx_head(32) x seq_q(1) x 16384 candidates x idx_dim(128) x 2 x FP4(0.25) = 134.217728 MFlops

# seq_k = 131072
131072 tokens x (3072 + 2048) Flops/token = 0.67108864 GFlops
```

### 3.5 计算总量

```python
常量 激活GEMM = 16 GFlops (论文口径 10.1G)
常量 CSA2 Attn = 2550.136832 MFlops
常量 SWA Attn = 671.08864 MFlops
变量 CSA2 Idx Full = 131072 tokens x 5120 Flops/token = 0.67108864 GFlops
常量 CSA2 Idx Reindex = 134.217728 MFlops

# 合计
16 + 2.550 + 0.671 + 0.671 + 0.134 = 20.027 GFlops
# 论文口径
10.1 + 2.550 + 0.671 + 0.671 + 0.134 = 14.127 GFlops
```


## 四、Prefill 计算量 (FLOPs)

**128K 序列 = 1048.577 + 158.502 + 44.023 + 26.439 = 1277.541 TFlops ≈ 1.3 PFlops**

> Decoder SWA Bounded Replay (论文 §3.2.2，已部署)：decoder global KV 由 Encoder 末层隐藏态投影生成，decoder 仅对尾部 n_win=128 token 做 bounded forward 产生 SWA KV，论文明确 "nearly halving total prefill computation"；本节按此已实现口径计算 = Encoder 全序列 + Decoder 128 token

> 因果掩码：Encoder Indexer Full 层稠密扫描只算下三角 ≈ seq²/2；CSA2 Top-K 天然因果（前 1024 个位置按 min(512, pos/2) 计，修正量 <0.5%，此处忽略）；Decoder bounded 部分为末尾 128 个 query，近似可见全量前缀，不计 causal 折减

> 论文口径：同第三节按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算，激活 GEMM 为 661.914 TFlops，合计 890.878 TFlops ≈ 0.9 PFlops

### 4.1 激活权重 GEMM

Bounded Replay 下 prefill 只完整运行 Encoder 20 层，对应论文 "only 8B parameters during prefill"（decode 为 16B），按真实计算的 FP8 计算，随序列长度线性增长 (≈ seq)

```python
# Encoder 20 层全序列
131072 tokens x 8B x 2 x FP8(0.5) = 1048.576 TFlops
# Decoder 20 层 bounded forward (n_win=128)
128 tokens x 8B x 2 x FP8(0.5) = 0.001 TFlops
```

> 备注：论文口径 MOE(FP4) `131072 x 5.05G = 661.914 TFlops`

### 4.2 CSA2 Attention

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，FP4 仅存储；Encoder 18 个 CSA2 层随序列长度线性增长 (≈ seq)，Decoder 20 层仅 128 query 为小项常量

```python
# Encoder: P = Q @ K.T / O = P @ V
18 Layers x Hq(64) x seq_q(?) x top(512) x kv_lora(512) x 2 x BF16(1) = 603.979776 MFlops/token
131072 tokens x (603.979776 + 603.979776) MFlops/token = 158.330 TFlops

# Decoder bounded: 128 query (top 部分)
128 tokens x 20 Layers x Hq(64) x top(512) x kv_lora(512) x 2 x 2 x BF16(1) = 0.172 TFlops
```

### 4.3 SWA Attention

固定窗口 128 且不压缩；Encoder 20 层随序列长度线性增长 (≈ seq)，Decoder 20 层仅 128 query 为小项常量

```python
# Encoder: 20 层 = 2 纯 SWA + 18 CSA2 的窗口部分
131072 tokens x 20 Layers x Hq(64) x seq_k(128) x kv_lora(512) x 2 x 2 x BF16(1) = 43.980 TFlops

# Decoder bounded: 128 query (window 部分)
128 tokens x 20 Layers x Hq(64) x seq_k(128) x kv_lora(512) x 2 x 2 x BF16(1) = 0.043 TFlops
```

### 4.4 CSA2 Indexer

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP4；Encoder Full 层随序列长度平方增长（因果下三角 ≈ seq²/2，K 为 m=2 压缩条目），Decoder Full/Reindex 层仅 128 query 为线性小项

```python
# Encoder Full (3 层, m=2, 因果下三角)
3 Layers x idx_head(32) x seq_q(?) x seq_k(?) / compress(2) x idx_dim(128) x 2 x FP4(0.25) x causal(0.5) = 1536 Flops/token²
131072² token² x 1536 Flops/token² = 26.388 TFlops

# Decoder Full (层 20, m=1): 128 query 全量前缀
128 tokens x 131072 x idx_head(32) x idx_dim(128) x 2 x FP4(0.25) = 0.034 TFlops

# Decoder Reindex (4 层): 128 query x 16384 候选池
128 tokens x 4 Layers x idx_head(32) x 16384 candidates x idx_dim(128) x 2 x FP4(0.25) = 0.017 TFlops
```

### 4.5 计算总量

```python
线性 激活GEMM = 1048.576 + 0.001 = 1048.577 TFlops (论文口径 661.914)
线性 CSA2 Attn = 158.330 + 0.172 = 158.502 TFlops
线性 SWA Attn = 43.980 + 0.043 = 44.023 TFlops
平方 CSA2 Idx Encoder Full = 131072² token² x 1536 Flops/token² = 26.388 TFlops
线性 CSA2 Idx Decoder = 0.034 + 0.017 = 0.051 TFlops

# 合计
1048.577 + 158.502 + 44.023 + 26.439 = 1277.541 TFlops ≈ 1.3 PFlops
# 论文口径
661.914 + 158.502 + 44.023 + 26.439 = 890.878 TFlops ≈ 0.9 PFlops
# 对比：未优化全量 prefill (Decoder 也全序列) = 2580.937 TFlops，Bounded Replay 使其近乎减半
```

## 五、论文 Figure 1(b) 关键数值 (890 / 3514 / 48068) 的计算

> 统计口径 (论文 §1)：Global KV = 主 KV + Indexer K；SWA 窗口 KV 属 Local KV 不计入（固定窗口与序列长度无关，摊到每 token 趋于 0）

### 5.1 DeepSeek-V4.1-Flash = 890 Bytes/token

即本文 1.3：CED 跨层复用后仅 4 个源层存储，每 token 摊销 2.5 条主 KV (NVFP4 288B) + 2.5 条 Indexer K (MXFP4 68B)

```shell
# 主 KV: 3 源层 m=2 + 1 源层 m=1
(3 ÷ 2 + 1 ÷ 1) x 288B = 2.5 x 288 = 720 Bytes/token
# Indexer K: 同样 2.5 条
2.5 x 68B = 170 Bytes/token

# 合计
720 + 170 = 890 Bytes/token
```

### 5.2 DeepSeek-V4-Flash = 3514 Bytes/token

无跨层复用，21 CSA(m=4) + 20 HCA(m=128) 每层各存一份；主 KV 584B (448 FP8 + 64x2 BF16 RoPE + 8 E8M0 scale)，Indexer K 为 FP4 68B（易误记为 FP8）

```shell
# CSA 主 KV
21 Layers ÷ 4 compress x 584B = 3066 Bytes/token
# HCA 主 KV
20 Layers ÷ 128 compress x 584B = 91.25 Bytes/token
# CSA Indexer K (FP4)
21 Layers ÷ 4 compress x 68B = 357 Bytes/token

# 合计
3066 + 91.25 + 357 = 3514.25 ≈ 3514 Bytes/token
```

### 5.3 DeepSeek-V3.2 = 48068 Bytes/token

MLA + Lightning Indexer (DSA)，61 层每层独立缓存完整 KV，无序列压缩无跨层复用；每层 788B = 主 KV 656B + Indexer K 132B

```shell
# 每层主 KV: 512 FP8 + 4 x fp32 scale + 64 x 2 BF16 RoPE
512 + 16 + 128 = 656 Bytes/token
# 每层 Indexer K: 128 FP8 + 1 x fp32 scale
128 + 4 = 132 Bytes/token

# 合计
61 Layers x (656 + 132) = 61 x 788 = 48068 Bytes/token
```

> 注意：V3.2 的 scale 为 fp32 (4B)，Indexer K 为 FP8（与 V4 的 FP4 区分），两个口径均为代码实证的精确值

### 5.4 三代对比与压缩路线

```shell
# V3.2 → V4-Flash (÷13.7): 序列维压缩 (61 条 → 5.4 条主 KV) + Indexer FP4 化
48068 / 3514.25 ≈ 13.7
# V4-Flash → V4.1-Flash (÷3.9): 跨层复用 (41 层 → 4 源层) + 主 KV FP8 → FP4
3514.25 / 890 ≈ 3.95，对应论文 "roughly 1/4 of DeepSeek-V4-Flash"
# 全程
48068 / 890 ≈ 54 倍
```

> 详细推导见 `DeepSeek_Global_KV_Cache_计算分析.md`

## 参考

- 模型参数 https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json
- 推理代码 https://github.com/deepseek-ai/DeepSeek-V4.1/tree/main/inference
- 技术报告 DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression (§2 CSA2/CED, §2.4.4 FP4 Main KV Cache)
- SWA bounded replay (SGLang/Miles Day-0) https://www.lmsys.org/blog/2026-09-10-deepseek-v41/
