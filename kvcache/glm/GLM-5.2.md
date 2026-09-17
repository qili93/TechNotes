## GLM-5.2(744B-A40B)：MLA + DSA w/ Top2048 + IndexShare(每4层共享)

> 单位约定：容量 Bytes → KB → MB → GB 与计算量 FLOPs → GFLOPs → TFLOPs 均按 1000 换算。

### ① 缓存容量 (Bytes)

**128K 序列 = 131072 × 92628 Bytes = 12.141 GB**

**Main KV 缓存 (MLA) = 1152 Bytes/token × 78层**

- `[seq, 1, qk_rope(64)]` BF16 位置编码需高精 = 128 Bytes/token

- `[seq, 1, kv_lora(512)]` BF16 = 1024 Bytes/token

- 注：GLM-5.2 权重为 BF16 无量化，vllm 默认 `cache_dtype=auto` 时主 KV 缓存为 BF16；可选 `fp8_ds_mla` 压缩布局 = `512 FP8 + 4×4 scale + 128 BF16 rope` = 656 Bytes/token（与 DeepSeek-V3.2 一致）

**Indexer K 缓存 (DSA) = 132 Bytes/token × 21层（仅 Full 层）**

- `[seq, 1, idx_dim(128)]` FP8 E4M3 = 128 + 4 = 132 Bytes/token

  - vllm 中 DSA Indexer K 缓存恒为量化存储（`auto` 即 fp8，V3.2 布局），128 FP8 元素共享一个 FP32 的 scale factor = 4 Bytes

- IndexShare：`index_topk_freq=4, index_skip_topk_offset=3`，每 4 层仅 Full 层计算 top-k 并持有 indexer 与 K 缓存，3 个 Shared 层复用其索引（`skip_topk` 层不建 indexer、不分配 K 缓存）；78 层中 Full 层 = 0,1,2,6,10,...,74 共 21 层

**整模型**：`Global KV Cache Per Token = 78 × 1152 + 21 × 132 = 92628 Bytes/token`

- 注：GLM-5.2 支持最长 1M (1048576) 上下文（`rope_theta=8M`），1M 序列缓存 = 8 × 12.141 ≈ 97.13 GB

### ② 访存量 (Bytes)

**128K 序列 = 78 × 2048×1152 + 21 × 131072×132 = 0.547 GB**

Main KV 缓存 (MLA) = `1152 Bytes/token` 每层只选 Top2048（78 层全部参与注意力）

Indexer K 缓存 (DSA) = `132 Bytes/token` 全量 Seq 扫描（仅 21 个 Full 层，Shared 层复用索引不扫描）

### ③ Decoder 计算量 (FLOPs)

> 对齐 DeepSeek 论文，计算量以 BF16 为基准，BF16 是 1.0, FP8 是 0.5，FP4 是 0.25

**128K 序列 = 78.7G + 22.2G + 131072×86016 (11.3G) = 112.2 GFlops**

激活权重 `GEMM = 39.3B × 2 × BF16(1.0) = 78.7 GFlops 常量`（indexer 权重仅 21 层，激活参数较 GLM-5.1 的 39.9B 下降）

GLM-5.2 权重为 BF16 无量化（vllm 中 glm_moe_dsa 走 UnquantizedLinearMethod，GLM 低延迟 GEMM 仅支持 BF16），不同于 DeepSeek-V3.2 的 FP8 权重 (0.5)；MoE 路由为 FP32

MLA 计算 Q@K.T 和 P@V 计算精度为 BF16

- Q@K.T = Q `[b, Hq(64), seq_q, kv_lora(512)+qk_rope(64)]` @ K `[b, topk(2048), kv_lora(512)+qk_rope(64)]`.T = P `[b, Hq(64), seq_q, topk(2048)]`

- P@V = P `[b, Hq(64), seq_q, topk(2048)]` @ V `[b, topk(2048), kv_lora(512)]` = O `[b, Hq(64), seq_q, kv_lora(512)]`

- 计算量：`78层 × Hq(64) × seq_q(1) × topk(2048) × (512+64+512) × 2 × BF16(1) = 22.2 GFlops 常量`（Shared 层复用索引后仍各自执行稀疏注意力）

Indexer 打分计算：仅 21 个 Full 层，计算量随序列增长，计算和存储格式都是 FP8

- q_idx @ k_idx.T = q_idx `[b, idx_head(32), seq_q, idx_dim(128)]` @ k_idx `[b, seq_k, idx_dim(128)]`.T = logits `[b, idx_head(32), seq_q, seq_k]`

- 计算量：`21层 × idx_head(32) × seq_q(1) × seq_k × idx_dim(128) × 2 × FP8(0.5) = 86016 FLOPs/token`（为 GLM-5.1 每层打分量的 21/78 ≈ 1/3.7）

### ④ Prefill 计算量 (FLOPs)

> 因果掩码：Indexer 只算下三角 ≈ seq²/2；MLA 经 DSA top-k 后每 query 仅 2048 个 key，天然因果（前 2048 个位置按 min(2048, pos) 计，修正量 <1%，此处忽略）
> MLA prefill 采用 MHA 展开形态（对齐参考实现）：每 head K = 192+64 = 256 维、V = 256 维

**128K 序列 = 10314.4 + 1372.2 + 738.9 = 12425.5 TFlops ≈ 12.4 PFlops**

激活权重 `GEMM = 131072 × 39.3B × 2 × BF16(1.0) = 10314.4 TFlops`

MLA 计算 (MLA prefill 采用 MHA 展开形态, BF16)：

- Q@K.T = Q `[b, Hq(64), seq_q, qk_nope(192)+qk_rope(64)]` @ K `[b, Hq(64), topk(2048), qk_nope(192)+qk_rope(64)]`.T = P `[b, Hq(64), seq_q, topk(2048)]`

- P@V = P `[b, Hq(64), seq_q, topk(2048)]` @ V `[b, Hq(64), topk(2048), v_head_dim(256)]` = O `[b, Hq(64), seq_q, v_head_dim(256)]`

- 计算量：`78层 × Hq(64) × seq_q × topk(2048) × (256+256) × 2 × BF16(1) = 10.469 GFlops/token`，全序列 = `10.469G × 131072 ≈ 1372.2 TFlops`

- 注：若 prefill 沿用 decode 的吸收形态 (512+64+512=1088 维/head)，此项为 2915.9 TFlops，是展开形态的 2.1 倍——这正是参考实现 prefill 走 MHA 展开形态的原因

Indexer 打分 (FP8, 因果下三角, 仅 21 个 Full 层)：

- 计算量：`21层 × idx_head(32) × seq_q × seq_k × idx_dim(128) × 2 × FP8(0.5) = 86016 FLOPs/(token·token)`

- 全序列 = `86016 × 131072² / 2 = 738.9 TFlops`

- 注：Prefill 全序列计算采用 casual 因果掩码，Indexer 只算下三角 ≈ seq²/2

### 参考

- 模型参数 https://modelscope.cn/models/ZhipuAI/GLM-5.2 (`config.json`)
- IndexShare（indexer 每 4 层共享，1M 上下文 per-token FLOPs 降 2.9×）https://arxiv.org/abs/2603.12201
- Shared 层 skip_topk 不建 indexer/K 缓存 vllm `vllm/models/deepseek_v32/attention.py` (`index_topk_freq`/`index_skip_topk_offset`)
- 权重 BF16 无量化 & GLM 低延迟 GEMM vllm `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py`
- Indexer K 缓存恒为 FP8 ("auto" means fp8, V3.2 layout) vllm `vllm/v1/attention/backends/mla/indexer.py`
- MLA KV 缓存布局 (BF16 1152B / fp8_ds_mla 656B) vllm `vllm/model_executor/layers/attention/mla_attention.py`
