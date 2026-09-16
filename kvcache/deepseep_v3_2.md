## DeepSeek-V3.2(671B-A37B)：MLA + DSA w/ Top2048

> 单位约定：容量 Bytes → KB → MB → GB 与计算量 FLOPs → GFLOPs → TFLOPs 均按 1000 换算。

### ① 缓存容量 (Bytes)

**128K 序列 = 131072 × 48068 Bytes = 6.300 GB**

**Main KV 缓存 (MLA) = 656 Bytes/token**

- `[seq, 1, qk_rope(64)]` BF16 位置编码需高精 = 128 Bytes/token

- `[seq, 1, kv_lora(512)]` FP8 E4M3 = 512 + 4×4 = 528 Bytes/token

  - `kv_lora(512)` 分为 4×128 tiles，每 tile 一个 FP32 的 scale factor = 4×4 Bytes

**Indexer K 缓存 (DSA) = 132 Bytes/token**

- `[seq, 1, idx_dim(128)]` FP8 E4M3 = 128 + 4 = 132 Bytes/token

  - 同上，128 FP8 元素共享一个 FP32 的 scale factor = 4 Bytes

**整模型**：`Global KV Cache Per Token = 61 × (656 + 132) = 48068 Bytes/token`

### ② 访存量 (Bytes)

**128K 序列 = 61 × (2048×656 + 131072×132) = 1.137 GB**

Main KV 缓存 (MLA) = `656 Bytes/token` 只选 Top2048

Indexer K 缓存 (DSA) = `132 Bytes/token` 全量 Seq 扫描

### ③ Decoder 计算量 (FLOPs)

> 对齐 DeepSeek 论文，计算量以 BF16 为基准，BF16 是 1.0, FP8 是 0.5，FP4 是 0.25

**128K 序列 = 37G + 34.8G + 131072×499712 (65.5G) = 137.3 GFlops**

激活权重 `GEMM = 37B × 2 × FP8(0.5) = 37 GFlops 常量`

MLA 计算 Q@K.T 和 P@V 计算精度为 BF16，FP8 仅存储，论文中有明确说明

- Q@K.T = Q `[b, Hq(128), seq_q, kv_lora(512)+qk_rope(64)]` @ K `[b, topk(2048), kv_lora(512)+qk_rope(64)]`.T = P `[b, Hq(128), seq_q, topk(2048)]`

- P@V = P `[b, Hq(128), seq_q, topk(2048)]` @ V `[b, topk(2048), kv_lora(512)]` = O `[b, Hq(128), seq_q, kv_lora(512)]`

- 计算量：`61层 × Hq(128) x seq_q(1) × topk(2048) × (512+64+512) × 2 × BF16(1) = 34.8 GFlops 常量`

Indexer 打分计算：计算量随序列增长，计算和存储格式都是 FP8

- q_idx @ k_idx.T = q_idx `[b, idx_head, seq_q, idx_dim]` @ k_idx `[b, seq_k, idx_dim]`.T = logits `[b, idx_head, seq_q, seq_k]`

- 计算量：`61层 × idx_head(64) × seq_q(1) x seq_k x idx_dim(128) × 2 × FP8(0.5) = 499712 FLOPs/token`

### ④ Prefill 计算量 (FLOPs)

> 因果掩码：Indexer 只算下三角 ≈ seq²/2；MLA 经 DSA top-k 后每 query 仅 2048 个 key，天然因果（前 2048 个位置按 min(2048, pos) 计，修正量 <1%，此处忽略）
> MLA prefill 采用 MHA 展开形态（对齐参考实现）：每 head K = 128+64 = 192 维、V = 128 维

**128K 序列 = 4849.66 + 1341.40 + 4292.49 = 10483.56 TFlops ≈ 10.5 PFlops**

激活权重 `GEMM = 131072 × 37B × 2 × FP8(0.5) = 4849.66 TFlops`

MLA 计算 (MLA prefill 采用 MHA 展开形态, BF16)：

- Q@K.T = Q `[b, Hq(128), seq_q, qk_nope(128)+qk_rope(64)]` @ K `[b, Hq(128), topk(2048), qk_nope(128)+qk_rope(64)]`.T = P `[b, Hq(128), seq_q, topk(2048)]`

- P@V = P `[b, Hq(128), seq_q, topk(2048)]` @ V `[b, Hq(128), topk(2048), v_head_dim(128)]` = O `[b, Hq(128), seq_q, v_head_dim(128)]`

- 计算量：`61层 × Hq(128) × seq_q × topk(2048) × (192+128) × 2 × BF16(1) = 10.234 GFlops/token`，全序列 = `10.234G × 131072 ≈ 1341.404 TFlops`

- 注：若 prefill 沿用 decode 的吸收形态 (512+64+512=1088 维/head)，此项为 4560.774 TFlops，是展开形态的 3.4 倍——这正是参考实现 prefill 走 MHA 展开形态的原因

Indexer 打分 (FP8, 因果下三角)：

- 计算量：`61层 × idx_head(64) × seq_q × seq_k × idx_dim(128) × 2 × FP8(0.5) = 499712 FLOPs/(token·token)`

- 全序列 = `499712 × 131072² / 2 = 4292.49 TFlops`

- 注：Prefill 全序列计算采用 casual 因果掩码，Indexer 只算下三角 ≈ seq²/2

### 参考

- 模型卡片 https://sebastianraschka.com/llm-architecture-gallery/#card-deepseek-v3-2
- 模型结构https://github.com/CalvinXKY/InfraTech/blob/main/models/deepseek_v3_2/deepseek_v3_2_architecture.jpg
- 模型参数 https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json
- MLA 量化精度 https://github.com/flashinfer-ai/flashinfer/issues/2426
