# DeepSeek-V4-Flash(284B-A13B)：CSA + HCA w/ Top512

![alt](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/deepseek-v4-flash.webp)

模型不含 MTP 共计 43 层：`2 SWA(w=128) + 21 CSA(m=4)/SWA(w=128) + 20 HCA(m=128)/SWA(w=128)`

## 一、缓存容量 (Bytes)

**128K 序列 = 131072 × 3850.25 B + 3.214336 MB = 507.874304 MB**

### 1.1 Main KV 缓存 (CSA/HCA/SWA) = 584 Bytes/token

KVCache 缓存 `[b, seq, kv_lora(512)]` 448B NoPE + 128B RoPE + 8B fp8 scale = 584B/token

> 精度说明 (参考 [vllm](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/models/deepseek_v4/attention.py#L751))：前 448 维 FP8 E4M3 + 后 64 维 RoPE BF16 + FP8 scale 按完整 512 维分配 8 个 FP8 E8M0（每 64 维 1 个) 

> 注意：V3.2 缓存 `kv[b, seq, 512] + pe[b, seq, 64]` 两条；V4.0 合并为 `kv[b, seq, 512]` 即 `448 NoPE + 64 RoPE`

### 1.2 Indexer K 缓存 (仅 CSA 层) = 132 Bytes/token

Indexer K 缓存 `[b, seq, 1, idx_dim(128)]` 支持 MXFP4 和 FP8 两种格式，默认走 FP8 路径

- FP8 E4M3: `128 fp8 + 4 fp32 scale = 132 Bytes`

- MXFP4: `64 packed values + 4 UE8M0 scales = 68 Bytes`, 128 通道每 32 通道 1 个 UE8M0 scale

> Indexer 部分 `Hq=64`, `Hk=1`，因此 cache 的 k_idx 是单头的 MQA 模式；精度说明参考 [vllm](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/models/deepseek_v4/attention.py#L870) 的 `DeepseekV4Indexer` 。

### 1.3 整模型 Global KV Cache Per Token（CSA + HCA + Indexer）

> 注意：DeepSeek 论文中将 SWA 的 128 token 的 KV 设定为 Local KV Cache，因此 1.3 不包含 SWA

```shell
# 1) CSA Main KV
21 Layers × 584B/token ÷ 4 compress = 3066 Bytes/token

# 2) HCA Main KV
20 Layers × 584B/token ÷ 128 compress = 91.25 Bytes/token

# 3.1) CSA Indexer K in FP8
21 Layers × 132B/token ÷ 4 compress = 693 Bytes/token
# 3.2) CSA Indexer K in MXFP4
21 Layers × 68B/token ÷ 4 compress = 357 Bytes/token

# 合计 FP8
3066 + 91.25 + 693 = 3850.25 Bytes/token
# 合计 FP4
3066 + 91.25 + 357 = 3514.25 Bytes/token
```

### 1.4 整模型 Local KV Cache (SWA window = 128)

固定窗口 128 与序列长度无关

```shell
# SWA Main KV
43 Layers x 128 window x 584B/token = 3.214336 MB
```

> 注意：SWA bounded replay（窗口不持久化、缺失时重放重建）是 V4.1 的部署优化（SGLang/Miles），V4.0 仍需缓存。

## 二、访存量 (Bytes)

**128K 序列 = 6.279168 + 11.96032 + 90.832896 + 3.214336 = 112.28672 MB**

- CSA Main KV：`21 Layers x Top(512) x 584B/token = 6.279168 MB`，这里是压缩后的 Top512 所以不需要除压缩比

- HCA Main KV: `20 Layers x 131072 x 584B/token ÷ 128 compress = 11.96032 MB`，HCA 全量参与不做 TopK 选择

- CSA Indexer K: `21 Layers x 131072 x 132B/token ÷ 4 compress = 90.832896 MB` 打分也是在压缩后KV段上进行

- SWA Local KV: `43 Layers x 128 window x 584B/token = 3.214336 MB`

## 三、Decoder 计算量 (FLOPs)

**128K 序列 = 13G + 1.409G + 131072 x 20480 Flops/token + 0.721G + 131072 x 43008 Flops/token = 23.451 GFLOPs**

> 论文口径：DeepSeek-V4.1-Flash Tech Report Figure 2 的成本模型按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算

### 3.1 激活权重 GEMM

模型激活参数 A13B，MOE 采用 FP8 激活 + MXFP4 (FP4 E2M1) 权重即 W4A8 格式，但实际计算采用 FP8

计算量：`13B x 2 x FP8(0.5) = 13 GFlops`，Decoder `seq_q=1` 因此是常量

> 备注：论文中 MOE 按权重精度记 `FP4=0.25`, 按该口径激活 `7.6B x 2 x 0.25 + 5.1B x 2 x 0.5 = 8.9 GFlops`

### 3.2 CSA Attention

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，FP8/FP4 仅存储，论文中有明确说明

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
21 Layers x Hq(64) x seq_q(1) x top(512) x kv_lora(512) x 2 x BF16(1) = 704.643072 MFlops
# O = P @ V
21 Layers x Hq(64) x seq_q(1) x top(512) x kv_lora(512) x 2 x BF16(1) = 704.643072 MFlops

# 合计
704.643072 + 704.643072 = 1409.286144 MFlops
```

### 3.3 HCA Attention

计算量如下，是变量，随序列长度变化

```python
# P = Q @ K.T
20 Layers x Hq(64) x seq_q(1) x seq_k(?) / compress(128) x kv_lora(512) x 2 x BF16(1) = 10240 Flops/token
# O = P @ V
20 Layers x Hq(64) x seq_q(1) x seq_k(?) / compress(128) x kv_lora(512) x 2 x BF16(1) = 10240 Flops/token

# seq_k = 131072
131072 tokens x (10240 + 10240) Flops/token = 2.68435456 GFlops
```

### 3.4 SWA Attention

固定窗口 128 且不压缩，计算量是常量，不随序列长度变化

```python
# P = Q @ K.T
43 Layers x Hq(64) x seq_q(1) x seq_k(128) x kv_lora(512) x 2 x BF16(1) = 360.710144 MFlops
# O = P @ V
43 Layers x Hq(64) x seq_q(1) x seq_k(128) x kv_lora(512) x 2 x BF16(1) = 360.710144 MFlops

# 合计
360.710144 + 360.710144 = 721.420288 MFlops
```

### 3.5 CSA Indexer

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP8，计算量是变量，随序列长度变化

```python
# 输入输出 Shape
q_idx.shape = [b, idx_head(64), seq_q, idx_dim(128)]
k_idx.shape = [b, seq_k, idx_dim(128)]

# 每 token 计算量
21 Layers x idx_head(64) x seq_q(1) x seq_k(?) / compress(4) x idx_dim(128) x 2 x FP8(0.5) = 43008 Flops/token

# seq_k = 131072
131072 tokens x 43008 Flops/token = 5.637144576 GFlops
```

### 3.6 计算总量

```python
常量 激活GEMM = 13 GFlops (论文口径 8.9G)
常量 CSA Attn = 1409.286144 MFlops
变量 HCA Attn = 131072 tokens x 20480 Flops/token = 2.68435456 GFlops
常量 SWA Attn = 721.420288 MFlops
变量 CSA Idx = 131072 tokens x 43008 Flops/token = 5.637144576 GFlops 

# 合计
13 + 1.409 + 2.684 + 0.721 + 5.637 = 23.451 GFlops
# 论文口径
8.9 + 1.409 + 2.684 + 0.721 + 5.637 = 19.351 GFlops
```


## 四、Prefill 计算量 (FLOPs)

**128K 序列 = 1703.936 + 184.732 + 175.922 + 94.544 + 369.436 = 2528.570 TFlops ≈ 2.5 PFlops**

> 因果掩码：Indexer / HCA 稠密扫描只算下三角 ≈ seq²/2；CSA Top-K 天然因果（前 2048 个位置按 min(512, pos/4) 计，修正量 ≈0.8%，此处忽略）

> 论文口径：同第三节按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算，激活 GEMM 为 1166.541 TFlops，合计 1991.175 TFlops ≈ 2.0 PFlops

### 4.1 激活权重 GEMM

与 Decoder 3.1 相同，A13B 按真实计算的 FP8 计算，计算量随序列长度线性增长 (≈ seq)

`131072 tokens x 13B x 2 x FP8(0.5) = 1703.936 TFlops`

> 备注：论文口径 MOE(FP4) `131072 x 8.9G = 1166.541 TFlops`

### 4.2 CSA Attention

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，FP8/FP4 仅存储，论文中有明确说明

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, kv_lora(512)]
k.shape = [b, top(512), kv_lora(512)]
p.shape = [b, Hq(64), seq_q, top(512)] # P = Q @ K.T
v.shape = [b, top(512), kv_lora(512)]
o.shape = [b, Hq(64), seq_q, kv_lora(512)] # O = P @ V
```

计算量随序列长度线性增长 (≈ seq)

```python
# P = Q @ K.T
21 Layers x Hq(64) x seq_q(?) x top(512) x kv_lora(512) x 2 x BF16(1) = 704.643072 MFlops/token
# O = P @ V
21 Layers x Hq(64) x seq_q(?) x top(512) x kv_lora(512) x 2 x BF16(1) = 704.643072 MFlops/token

# 全序列 128K
131072 tokens x (704.643072 + 704.643072) MFlops/token = 184.732 TFlops
```

### 4.3 HCA Attention

计算量如下，是变量，随序列长度平方增长（因果下三角 ≈ seq²/2）

```python
# P = Q @ K.T
20 Layers x Hq(64) x seq_q(?) x seq_k(?) / compress(128) x kv_lora(512) x 2 x BF16(1) x causal(0.5) = 5120 Flops/token²
# O = P @ V
20 Layers x Hq(64) x seq_q(?) x seq_k(?) / compress(128) x kv_lora(512) x 2 x BF16(1) x causal(0.5) = 5120 Flops/token²

# seq_q = seq_k = 131072，下三角 ÷ 2
131072² token² x (5120 + 5120) Flops/token² = 175.922 TFlops
```

### 4.4 SWA Attention

固定窗口 128 且不压缩，计算量如下，随序列长度线性增长 (≈ seq)

```python
# P = Q @ K.T
43 Layers x Hq(64) x seq_q(?) x seq_k(128) x kv_lora(512) x 2 x BF16(1) = 360.710144 MFlops/token
# O = P @ V
43 Layers x Hq(64) x seq_q(?) x seq_k(128) x kv_lora(512) x 2 x BF16(1) = 360.710144 MFlops/token

# 全序列
131072 tokens x (360.710144 + 360.710144) MFlops/token = 94.544 TFlops
```

### 4.5 CSA Indexer

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP8，计算量随序列长度平方增长（因果下三角 ≈ seq²/2，K 为 m=4 压缩条目）

```python
# 输入输出 Shape
q_idx.shape = [b, idx_head(64), seq_q, idx_dim(128)]
k_idx.shape = [b, seq_k, idx_dim(128)]

# 每 token 计算量
21 Layers x idx_head(64) x seq_q(?) x seq_k(?) / compress(4) x idx_dim(128) x 2 x FP8(0.5) x causal(0.5) = 21504 Flops/token²

# seq_q = seq_k = 131072，下三角 ÷ 2
131072² token² x 21504 Flops/token² = 369.436 TFlops
```

### 4.6 计算总量

```python
线性 激活GEMM = 131072 tokens x 13 GFlops = 1703.936 TFlops (论文口径 1166.541)
线性 CSA Attn = 131072 tokens x 1409.286144 MFlops = 184.732 TFlops
平方 HCA Attn = 131072² token² x 10240 Flops/token² = 175.922 TFlops
线性 SWA Attn = 131072 tokens x 721.420288 MFlops = 94.544 TFlops
平方 CSA Idx = 131072² token² x 21504 Flops/token² = 369.436 TFlops

# 合计
1703.936 + 184.732 + 175.922 + 94.544 + 369.436 = 2528.570 TFlops ≈ 2.5 PFlops
# 论文口径
1166.541 + 184.732 + 175.922 + 94.544 + 369.436 = 1991.175 TFlops ≈ 2.0 PFlops
```

## 参考

- 模型卡片 https://sebastianraschka.com/llm-architecture-gallery/#card-deepseek-v4-flash
- 模型结构 https://github.com/CalvinXKY/InfraTech/blob/main/models/deepseek_v4/README.md
- 模型参数 https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json
