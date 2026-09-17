# GLM-5.3-Flash(320B-A18B)：KDA + DSA(MLA-NoPE) w/ Top2048@kpool4 + mHC

模型不含 MTP 共计 45 层：`3 Dense + 42 MoE`，注意力为 `34 KDA(linear) + 11 DSA(每 4 层 1 个，层号 3,7,11,...,43)`；GLM 系列首个原生多模态模型（本文仅计 text tower，不含 24 层 vision tower）

> 与 GLM-5.3 的关系：全新训练的 base model，非同一底座。首次引入 sparse+linear 混合注意力、mHC (Manifold-Constrained Hyper-Connections, 4 残差流)；MLA 改为 NoPE (`qk_rope_head_dim=0`)；Indexer 引入 kpool=4 压缩（每 4 个 K 条目池化为 1 个存储/扫描条目，topk 在池粒度选择 2048/4=512 池）；权重同为 FP8 E4M3 block-wise (128×128) + 动态激活 (W8A8)

## 一、缓存容量 (Bytes)

**128K 序列 = 131072 × 11627 B + 147.62 MB(KDA 状态) = 1.672 GB**

### 1.1 Main KV 缓存 (DSA/MLA-NoPE，仅 11 层) = 1024 Bytes/token

KVCache 缓存 `[seq, 1, kv_lora(512)]` BF16 = 1024B/token

> 精度说明：`mla_use_nope=True` 且 `qk_rope_head_dim=0`，主 KV 无 RoPE 段（对比 GLM-5.3 的 1152B = 1024B + 128B rope）；模型 `dtype=bfloat16`（FP8 仅量化权重），vllm `cache_dtype=auto` 时主 KV 缓存为 BF16 = 512×2 = 1024 Bytes/token

### 1.2 Indexer K 缓存 (kpool=4 压缩，仅 11 层) = 33 Bytes/token

Indexer K 缓存每池 `[1, 1, idx_dim(128)]` FP8 E4M3 = 128B + 4B scale = 132B/池，`index_kpool=4` 即每 4 token 存 1 池 = 33B/token

- vllm `Glm5NextIndexerCache`：KV spec `tokens_per_state=index_kpool(4)`，缓存分配每 4 token 一个 state；topk 在池粒度运行（`select_k = index_topk // index_kpool = 512` 池 = 2048 token）

- Tail cache（每请求常数）：in-progress pool 的 raw BF16 K + gate score = `11层 × 4 slots × 2 × 128 × 2B = 22.5 KB`，不计入 per-token

> 注意：GLM-5.3-Flash 的 11 个 DSA 层每层都有完整 indexer（`indexer_types` 全 full），没有 GLM-5.2/5.3 的 IndexShare；压缩靠的是 kpool 而不是跨层共享

### 1.3 整模型 Global KV Cache Per Token（MLA + Indexer）

```shell
# 1) Main KV (MLA-NoPE, 11 DSA 层)
11 Layers × 1024B/token = 11264 Bytes/token

# 2) Indexer K (11 层, FP8, kpool=4)
11 Layers × 132B/pool ÷ 4 pool = 363 Bytes/token

# 合计
11264 + 363 = 11627 Bytes/token
```

> 注：支持最长 1M (1048576) 上下文，1M 序列 Global KV ≈ 8 × 1.524 = 12.19 GB（对比 GLM-5.3 的 97.13 GB）

### 1.4 整模型 Local State (KDA，与序列长度无关)

KDA 为线性注意力，不存 KV，只存固定大小状态

```shell
# Recurrent state (FP32, auto)
34 Layers × 64 heads × 128 × 128 × 4B = 142.606336 MB

# Conv state (kernel=4, BF16)
34 Layers × (3 × 64×128) × (4-1) × 2B = 5.013504 MB

# 合计
142.606336 + 5.013504 = 147.619840 MB
```

> 精度说明 (vllm `mamba_utils.py` `kda_state_dtype`)：conv state 随模型 dtype = BF16，recurrent state `mamba_ssm_cache_dtype=auto` 恒为 FP32

## 二、访存量 (Bytes)

**128K 序列 = 23.068672 + 47.611776 + 147.619840 = 218.300288 MB**

- Main KV (DSA)：`11 Layers x Top(2048) x 1024B/token = 23.068672 MB`，每层只选 Top2048（512 池）

- Indexer K (kpool)：`11 Layers x (131072÷4) x 132B/pool = 47.611776 MB` 全量池扫描

- KDA Local State：`34 Layers x 4.341760 MB = 147.619840 MB`，每 token 读/写 recurrent state（读计一次）

## 三、Decoder 计算量 (FLOPs)

**128K 序列 = 18G + 2.953G + 0.214G + 131072 x 11264 Flops/token = 22.6 GFLOPs**

> 论文口径：计算量按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算

### 3.1 激活权重 GEMM

模型激活参数 A18B，权重 FP8 E4M3 block-wise (128×128) + 动态激活（W8A8），实际计算采用 FP8

计算量：`18B x 2 x FP8(0.5) = 18 GFlops`，Decoder `seq_q=1` 因此是常量

> 备注：layernorm / mlp.gate(路由) / 小投影不量化保持 BF16（config `modules_to_not_convert`），MoE 路由为 FP32；mHC 的流混合矩阵参数量极小，已含在激活口径内

### 3.2 DSA MLA Attention (11 层)

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，FP8 仅存储/权重；decode 走吸收形态（NoPE，latent 512 维）

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, kv_lora(512)]
k.shape = [b, top(2048), kv_lora(512)]
p.shape = [b, Hq(64), seq_q, top(2048)] # P = Q @ K.T
v.shape = [b, top(2048), kv_lora(512)]
o.shape = [b, Hq(64), seq_q, kv_lora(512)] # O = P @ V
```

计算量如下，是常量，不随序列长度变化

```python
# P = Q @ K.T
11 Layers x Hq(64) x seq_q(1) x top(2048) x kv_lora(512) x 2 x BF16(1) = 1.476 GFlops
# O = P @ V
11 Layers x Hq(64) x seq_q(1) x top(2048) x kv_lora(512) x 2 x BF16(1) = 1.476 GFlops

# 合计
1.476 + 1.476 = 2.953 GFlops
```

### 3.3 KDA Linear Attention (34 层)

线性注意力 decode 为递推形式，计算量是常量，不随序列长度变化；BF16 计算（state 更新 FP32，此处按 BF16 口径 = 1 折算）

```python
# 每 token 每层: 状态衰减 + delta rule 更新 + 输出 ≈ 3 次 state matvec
34 Layers x 3 x H(64) x d(128) x d(128) x 2 = 213.909504 MFlops

# 合计 ≈ 0.214 GFlops (conv1d kernel=4 约 0.2M/层, 忽略)
```

### 3.4 DSA Indexer (11 层, kpool=4)

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP8，在池粒度扫描（扫描量 = seq/4），计算量是变量，随序列长度变化

```python
# 输入输出 Shape
q_idx.shape = [b, idx_head(32), seq_q, idx_dim(128)]
k_idx.shape = [b, seq_k/4 (池数), idx_dim(128)]

# 每 token 计算量
11 Layers x idx_head(32) x seq_q(1) x (seq_k/4) x idx_dim(128) x 2 x FP8(0.5) = 11264 Flops/token

# seq_k = 131072
131072 tokens x 11264 Flops/token = 1.476 GFlops
```

### 3.5 计算总量

```python
常量 激活GEMM = 18 GFlops
常量 DSA MLA = 2.953 GFlops
常量 KDA Attn = 0.214 GFlops
变量 DSA Idx = 131072 tokens x 11264 Flops/token = 1.476 GFlops

# 合计
18 + 2.953 + 0.214 + 1.476 = 22.643 GFlops
```

## 四、Prefill 计算量 (FLOPs)

**128K 序列 = 2359.3 + 193.5 + 28.1 + 96.8 = 2677.7 TFlops ≈ 2.7 PFlops**

> 因果掩码：Indexer 池扫描只算下三角 ≈ (seq/4)²/2 池对；DSA Top-K 天然因果（前 2048 个位置按 min(2048, pos) 计，修正量 <1%，此处忽略）
> MLA prefill 采用 MHA 展开形态（对齐参考实现）：每 head K = 256 维、V = 256 维（NoPE）

### 4.1 激活权重 GEMM

与 Decoder 3.1 相同，A18B 按真实计算的 FP8 计算，计算量随序列长度线性增长 (≈ seq)

`131072 tokens x 18B x 2 x FP8(0.5) = 2359.296 TFlops`

### 4.2 DSA MLA Attention (11 层)

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，prefill 采用 MHA 展开形态

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, qk_nope(256)]
k.shape = [b, Hq(64), top(2048), qk_nope(256)]
p.shape = [b, Hq(64), seq_q, top(2048)] # P = Q @ K.T
v.shape = [b, Hq(64), top(2048), v_head_dim(256)]
o.shape = [b, Hq(64), seq_q, v_head_dim(256)] # O = P @ V
```

计算量随序列长度线性增长 (≈ seq)

```python
# P = Q @ K.T + O = P @ V
11 Layers x Hq(64) x seq_q(?) x top(2048) x (256+256) x 2 x BF16(1) = 1.476 GFlops/token

# 全序列 128K
131072 tokens x 1.476 GFlops/token = 193.5 TFlops
```

> 注：若 prefill 沿用 decode 的吸收形态 (512+512=1024 维/head)，此项为 387.0 TFlops，是展开形态的 2 倍

### 4.3 KDA Linear Attention (34 层)

prefill 为分块 (chunk) 并行扫描，计算量随序列长度线性增长，per-token 与递推形态同量级

```python
# 每 token
34 Layers x 3 x H(64) x d(128) x d(128) x 2 = 213.909504 MFlops/token

# 全序列
131072 tokens x 0.214 GFlops/token = 28.1 TFlops
```

### 4.4 DSA Indexer (11 层, kpool=4)

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP8，计算量随序列长度平方增长（因果下三角，K 侧为 kpool=4 压缩池）

```python
# 每 token 计算量
11 Layers x idx_head(32) x seq_q(?) x (seq_k/4) x idx_dim(128) x 2 x FP8(0.5) x causal(0.5) = 5632 Flops/token²

# seq_q = seq_k = 131072，池对数 = seq²/4，下三角 ÷ 2
131072² token² ÷ 4 pool x 5632 x 2 Flops = 96.8 TFlops
```

### 4.5 计算总量

```python
线性 激活GEMM = 131072 tokens x 18 GFlops = 2359.296 TFlops
线性 DSA MLA = 131072 tokens x 1.476 GFlops/token = 193.5 TFlops
线性 KDA Attn = 131072 tokens x 0.214 GFlops/token = 28.1 TFlops
平方 DSA Idx = 131072² token² x 5632 Flops/token² = 96.8 TFlops

# 合计
2359.3 + 193.5 + 28.1 + 96.8 = 2677.7 TFlops ≈ 2.7 PFlops
```

## 参考

- 模型卡片 (320B-A18B, sparse+linear 混合, mHC) https://modelscope.cn/models/ZhipuAI/GLM-5.3-Flash (`README.md`)
- 模型参数 https://modelscope.cn/models/ZhipuAI/GLM-5.3-Flash (`config.json`: `text_config`)
- kpool=4 压缩与池粒度 topk vllm `vllm/transformers_utils/configs/glm5_next.py` ("Every index_kpool indexer K entries pool into one stored entry")
- Indexer K 缓存池存储 (`tokens_per_state=index_kpool`) vllm `vllm/models/glm5next/common/attention.py` (`Glm5NextIndexerCache`)
- Tail cache (in-progress pool raw K+gate) vllm `vllm/models/glm5next/common/attention.py` (`Glm5NextTailCache`)
- KDA state 形状与精度 (recurrent FP32 / conv BF16) vllm `vllm/model_executor/layers/mamba/mamba_utils.py` (`kda_state_shape`/`kda_state_dtype`)
- Indexer Q 经 FWHT-128 旋转后 FP8 量化 vllm `vllm/models/glm5next/common/attention.py` (`fwht128_quant_fp8`)
