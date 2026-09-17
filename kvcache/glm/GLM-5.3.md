# GLM-5.3(744B-A40B)：MLA + DSA w/ Top2048 + IndexShare(每4层共享) + FP8 权重

模型不含 MTP 共计 78 层：`3 Dense + 75 MoE`，每层均为 `MLA(kv_lora=512) + DSA(topk=2048)`；Indexer 为 `21 Full + 57 Shared`（每 4 层共享 topk 索引）

> 与 GLM-5.2 的关系：同一 base model（config 除 `quantization_config` 外完全一致），官方说明所有提升来自 post-training；唯一结构差异是发布 checkpoint 经 **FP8 E4M3 block-wise (128×128) 权重量化 + 动态激活 (W8A8)**，GLM-5.2 为 BF16 无量化

## 一、缓存容量 (Bytes)

**128K 序列 = 131072 × 92628 B = 12.141 GB**

### 1.1 Main KV 缓存 (MLA) = 1152 Bytes/token × 78层

KVCache 缓存 `[seq, 1, kv_lora(512)]` 1024B + `[seq, 1, qk_rope(64)]` 128B = 1152B/token

> 精度说明：模型 `dtype=bfloat16`（FP8 仅量化权重，不改变模型 dtype），vllm `cache_dtype=auto` 时主 KV 缓存为 BF16 = 512×2 + 64×2 = 1152 Bytes/token；可选 `fp8_ds_mla` 压缩布局 = `512 FP8 + 4×4 scale + 128 BF16 rope` = 656 Bytes/token（与 DeepSeek-V3.2 一致）

### 1.2 Indexer K 缓存 (DSA，仅 21 个 Full 层) = 132 Bytes/token

Indexer K 缓存 `[seq, 1, idx_dim(128)]` FP8 E4M3 = 128B + 4B scale = 132B/token

- FP8 E4M3: `128 fp8 + 4 fp32 scale = 132 Bytes`，128 个 FP8 元素共享一个 FP32 scale factor

- vllm 中 DSA Indexer K 缓存恒为量化存储（`auto` 即 fp8，V3.2 布局），与主 KV 缓存精度无关

> IndexShare：`index_topk_freq=4, index_skip_topk_offset=3`，每 4 层仅 Full 层计算 top-k 并持有 indexer 与 K 缓存，3 个 Shared 层复用其索引（`skip_topk` 层不建 indexer、不分配 K 缓存）；78 层中 Full 层 = 0,1,2,6,10,...,74 共 21 层

### 1.3 整模型 Global KV Cache Per Token（MLA + Indexer）

```shell
# 1) Main KV (MLA)
78 Layers × 1152B/token = 89856 Bytes/token

# 2) Indexer K (仅 21 Full 层, FP8)
21 Layers × 132B/token = 2772 Bytes/token

# 合计
89856 + 2772 = 92628 Bytes/token
```

> 注：GLM-5.3 支持最长 1M (1048576) 上下文（`rope_theta=8M`），1M 序列缓存 = 8 × 12.141 ≈ 97.13 GB

## 二、访存量 (Bytes)

**128K 序列 = 184.049664 + 363.331584 = 547.381248 MB ≈ 0.547 GB**

- Main KV (MLA)：`78 Layers x Top(2048) x 1152B/token = 184.049664 MB`，每层只选 Top2048（78 层全部参与注意力）

- Indexer K (DSA)：`21 Layers x 131072 x 132B/token = 363.331584 MB` 全量 Seq 扫描（仅 21 个 Full 层，Shared 层复用索引不扫描）

## 三、Decoder 计算量 (FLOPs)

**128K 序列 = 39.3G + 22.2G + 131072 x 86016 Flops/token = 72.8 GFLOPs**

> 论文口径：计算量按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算

### 3.1 激活权重 GEMM

模型激活参数 A40B（indexer 权重仅 21 层，激活参数 39.3B），权重 FP8 E4M3 block-wise (128×128) + 动态激活（W8A8），实际计算采用 FP8

计算量：`39.3B x 2 x FP8(0.5) = 39.3 GFlops`，Decoder `seq_q=1` 因此是常量

> 备注：GLM-5.2 同构但权重 BF16，此项为 `39.3B x 2 x BF16(1.0) = 78.7 GFlops`；vllm 中 FP8 走 block-wise Fp8 GEMM，layernorm / mlp.gate(路由) / indexers_proj / embed / lm_head 不量化保持 BF16（config `modules_to_not_convert`），MoE 路由为 FP32

### 3.2 MLA Attention

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，FP8 仅存储/权重，78 层全部参与（Shared 层复用索引后仍各自执行稀疏注意力）

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, kv_lora(512)+qk_rope(64)]
k.shape = [b, top(2048), kv_lora(512)+qk_rope(64)]
p.shape = [b, Hq(64), seq_q, top(2048)] # P = Q @ K.T
v.shape = [b, top(2048), kv_lora(512)]
o.shape = [b, Hq(64), seq_q, kv_lora(512)] # O = P @ V
```

计算量如下，是常量，不随序列长度变化

```python
# P = Q @ K.T
78 Layers x Hq(64) x seq_q(1) x top(2048) x (512+64) x 2 x BF16(1) = 11.123 GFlops
# O = P @ V
78 Layers x Hq(64) x seq_q(1) x top(2048) x 512 x 2 x BF16(1) = 11.123 GFlops

# 合计
78 x 64 x 1 x 2048 x (512+64+512) x 2 x BF16(1) = 22.247 GFlops
```

### 3.3 DSA Indexer

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP8，仅 21 个 Full 层，计算量是变量，随序列长度变化

```python
# 输入输出 Shape
q_idx.shape = [b, idx_head(32), seq_q, idx_dim(128)]
k_idx.shape = [b, seq_k, idx_dim(128)]

# 每 token 计算量
21 Layers x idx_head(32) x seq_q(1) x seq_k(?) x idx_dim(128) x 2 x FP8(0.5) = 86016 Flops/token

# seq_k = 131072
131072 tokens x 86016 Flops/token = 11.274 GFlops
```

### 3.4 计算总量

```python
常量 激活GEMM = 39.3 GFlops (FP8 口径; GLM-5.2 BF16 为 78.7G)
常量 MLA Attn = 22.247 GFlops
变量 DSA Idx = 131072 tokens x 86016 Flops/token = 11.274 GFlops

# 合计
39.3 + 22.247 + 11.274 = 72.821 GFlops
# GLM-5.2 (BF16 权重) 对比
78.7 + 22.247 + 11.274 = 112.221 GFlops
```

## 四、Prefill 计算量 (FLOPs)

**128K 序列 = 5151.1 + 1372.2 + 738.9 = 7262.2 TFlops ≈ 7.3 PFlops**

> 因果掩码：Indexer 稠密扫描只算下三角 ≈ seq²/2；MLA 经 DSA top-k 后每 query 仅 2048 个 key，天然因果（前 2048 个位置按 min(2048, pos) 计，修正量 <1%，此处忽略）
> MLA prefill 采用 MHA 展开形态（对齐参考实现）：每 head K = 192+64 = 256 维、V = 256 维

### 4.1 激活权重 GEMM

与 Decoder 3.1 相同，A40B 按真实计算的 FP8 计算，计算量随序列长度线性增长 (≈ seq)

`131072 tokens x 39.3B x 2 x FP8(0.5) = 5151.1 TFlops`

> 备注：GLM-5.2 BF16 口径 `131072 x 78.7G = 10314.4 TFlops`

### 4.2 MLA Attention

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，prefill 采用 MHA 展开形态

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, qk_nope(192)+qk_rope(64)]
k.shape = [b, Hq(64), top(2048), qk_nope(192)+qk_rope(64)]
p.shape = [b, Hq(64), seq_q, top(2048)] # P = Q @ K.T
v.shape = [b, Hq(64), top(2048), v_head_dim(256)]
o.shape = [b, Hq(64), seq_q, v_head_dim(256)] # O = P @ V
```

计算量随序列长度线性增长 (≈ seq)

```python
# P = Q @ K.T + O = P @ V
78 Layers x Hq(64) x seq_q(?) x top(2048) x (256+256) x 2 x BF16(1) = 10.469 GFlops/token

# 全序列 128K
131072 tokens x 10.469 GFlops/token = 1372.2 TFlops
```

> 注：若 prefill 沿用 decode 的吸收形态 (512+64+512=1088 维/head)，此项为 2915.9 TFlops，是展开形态的 2.1 倍——这正是参考实现 prefill 走 MHA 展开形态的原因

### 4.3 DSA Indexer

`q_idx @ k_idx.T` 的存储精度和计算精度都是 FP8，仅 21 个 Full 层，计算量随序列长度平方增长（因果下三角 ≈ seq²/2）

```python
# 输入输出 Shape
q_idx.shape = [b, idx_head(32), seq_q, idx_dim(128)]
k_idx.shape = [b, seq_k, idx_dim(128)]

# 每 token 计算量
21 Layers x idx_head(32) x seq_q(?) x seq_k(?) x idx_dim(128) x 2 x FP8(0.5) x causal(0.5) = 43008 Flops/token²

# seq_q = seq_k = 131072，下三角 ÷ 2
131072² token² x 43008 Flops/token² = 738.9 TFlops
```

### 4.4 计算总量

```python
线性 激活GEMM = 131072 tokens x 39.3 GFlops = 5151.1 TFlops (FP8)
线性 MLA Attn = 131072 tokens x 10.469 GFlops/token = 1372.2 TFlops
平方 DSA Idx = 131072² token² x 43008 Flops/token² = 738.9 TFlops

# 合计
5151.1 + 1372.2 + 738.9 = 7262.2 TFlops ≈ 7.3 PFlops
# GLM-5.2 (BF16 权重) 对比
10314.4 + 1372.2 + 738.9 = 12425.5 TFlops ≈ 12.4 PFlops
```

## 参考

- 模型参数 https://modelscope.cn/models/ZhipuAI/GLM-5.3 (`config.json`)
- GLM-5.3 与 GLM-5.2 同一 base model https://modelscope.cn/models/ZhipuAI/GLM-5.3 (`README.md`: "same base model as GLM-5.2 — every gain comes from post-training")
- FP8 E4M3 block-wise (128×128) + 动态激活量化 GLM-5.3 `config.json` (`quantization_config`)
- IndexShare（indexer 每 4 层共享，1M 上下文 per-token FLOPs 降 2.9×）https://arxiv.org/abs/2603.12201
- Shared 层 skip_topk 不建 indexer/K 缓存 vllm `vllm/models/deepseek_v32/attention.py` (`index_topk_freq`/`index_skip_topk_offset`)
- Indexer K 缓存恒为 FP8 ("auto" means fp8, V3.2 layout) vllm `vllm/v1/attention/backends/mla/indexer.py`
- MLA KV 缓存布局 (BF16 1152B / fp8_ds_mla 656B) vllm `vllm/model_executor/layers/attention/mla_attention.py`
