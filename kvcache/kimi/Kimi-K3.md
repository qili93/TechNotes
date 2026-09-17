# Kimi-K3(2.8T-A104B)：KDA + Gated MLA(NoPE) 全量注意力 + LatentMoE + MXFP4 专家

模型不含 MTP 共计 93 层：`1 Dense + 92 LatentMoE`，注意力为 `69 KDA(linear) + 24 Gated MLA(全量，1-indexed 层号 4,8,...,92,93)`；原生多模态（本文仅计 text tower，不含 27 层 ViT）

> 与 GLM-5.3-Flash 同为 KDA+全注意力混合架构，但设计点不同：全注意力层无稀疏 indexer（每 4 层 1 个 Gated MLA 做**全量**注意力，序列平方项不可约）；MLA 同样 NoPE（`mla_use_nope=True`，`rotary_emb=None`）；MoE 为 LatentMoE（token 先降维到 3584 再路由，896 选 16 + 2 共享）；权重量化为 **MXFP4 (E2M1, group=32) 仅作用于路由专家 Linear**，注意力/共享专家/Dense MLP/lm_head 保持 BF16（config `quantization_config.ignore`）

## 一、缓存容量 (Bytes)

**128K 序列 = 131072 × 27648 B + 449.372160 MB(KDA 状态) = 4.073 GB**

### 1.1 Main KV 缓存 (Gated MLA，仅 24 层) = 1152 Bytes/token

KVCache 缓存 `[seq, 1, kv_lora(512)+qk_rope(64)]` BF16 = 1152B/token

> 精度说明：`mla_use_nope=True`（modeling_kimi_linear.py `assert self.use_nope; self.rotary_emb=None`），64 维 rope 段不施加旋转但仍随 `kv_a_proj`(输出 512+64=576) 一并缓存；模型 `dtype=bfloat16`，MXFP4 仅量化路由专家权重且 `kv_cache_scheme=None`，vllm `cache_dtype=auto` 时主 KV 缓存为 BF16 = 576×2 = 1152 Bytes/token

### 1.2 整模型 Global KV Cache Per Token（MLA）

```shell
# Main KV (Gated MLA, 24 层)
24 Layers × 1152B/token = 27648 Bytes/token
```

> 注：Kimi-K3 无 DSA indexer，全注意力层对全序列做稠密注意力；支持最长 1M (1048576) 上下文，1M 序列 Global KV = 8 × 3.624 ≈ 28.99 GB

### 1.3 整模型 Local State (KDA，与序列长度无关)

KDA 为线性注意力（69 层），不存 KV，只存固定大小状态

```shell
# Recurrent state (FP32, auto)
69 Layers × 96 heads × 128 × 128 × 4B = 434.110464 MB

# Conv state (kernel=4, BF16)
69 Layers × (3 × 96×128) × (4-1) × 2B = 15.261696 MB

# 合计
434.110464 + 15.261696 = 449.372160 MB
```

> 精度说明 (vllm `mamba_utils.py` `kda_state_dtype`)：conv state 随模型 dtype = BF16，recurrent state `mamba_ssm_cache_dtype=auto` 恒为 FP32（与 GLM-5.3-Flash 的 KDA 同一实现族 `KimiGatedDeltaNetAttention`）

## 二、访存量 (Bytes)

**128K 序列 = 3623.878656 + 449.372160 = 4073.250816 MB ≈ 4.073 GB**

- Main KV (MLA)：`24 Layers x 131072 x 1152B/token = 3623.878656 MB` 全量 Seq 扫描（无 topk 稀疏）

- KDA Local State：`69 Layers x 6.512640 MB = 449.372160 MB`，每 token 读/写 recurrent state（读计一次）

## 三、Decoder 计算量 (FLOPs)

**128K 序列 = 135.1G + 131072 x 5013504 Flops/token + 0.651G = 792.9 GFLOPs**

> 论文口径：计算量按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算

### 3.1 激活权重 GEMM

模型激活参数 A104B，其中路由专家 ≈ 48.6B 为 MXFP4 (W4)，其余（注意力/KDA/共享专家/Dense/投影）≈ 55.4B 为 BF16

```python
# 路由专家 (MXFP4): 16 experts x 92 layers x 3 x 3584 x 3072 = 48.6B
48.6B x 2 x FP4(0.25) = 24.3 GFlops
# 非路由部分 (BF16)
55.4B x 2 x BF16(1.0) = 110.8 GFlops

# 合计 = 135.1 GFlops，Decoder seq_q=1 因此是常量
```

> 备注：若 MXFP4 走反量化 BF16 GEMM（如 marlin 路径）而非原生 FP4，则此项为 `104B x 2 = 208 GFlops`；MoE 路由 sigmoid + noaux_tc，LatentMoE 先降维 7168→3584 再路由

### 3.2 Gated MLA Attention (24 层, 全量)

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，decode 走吸收形态；无稀疏 topk，K/V 为全序列，计算量是变量，随序列长度**线性**增长（decode）

输入输出 Shape 如下：

```python
q.shape = [b, Hq(96), seq_q, kv_lora(512)+qk_rope(64)]
k.shape = [b, seq_k, kv_lora(512)+qk_rope(64)]
p.shape = [b, Hq(96), seq_q, seq_k] # P = Q @ K.T
v.shape = [b, seq_k, kv_lora(512)]
o.shape = [b, Hq(96), seq_q, kv_lora(512)] # O = P @ V
```

```python
# 每 token 计算量
24 Layers x Hq(96) x seq_q(1) x seq_k(?) x (576+512) x 2 x BF16(1) = 5013504 Flops/token

# seq_k = 131072
131072 tokens x 5013504 Flops/token = 657.1 GFlops
```

### 3.3 KDA Linear Attention (69 层)

线性注意力 decode 为递推形式，计算量是常量，不随序列长度变化

```python
# 每 token 每层: 状态衰减 + delta rule 更新 + 输出 ≈ 3 次 state matvec
69 Layers x 3 x H(96) x d(128) x d(128) x 2 = 651.165696 MFlops

# 合计 ≈ 0.651 GFlops (conv1d kernel=4 约 0.3M/层, 忽略)
```

### 3.4 计算总量

```python
常量 激活GEMM = 135.1 GFlops (MXFP4 口径; 全 BF16 反量化口径 208G)
变量 Gated MLA = 131072 tokens x 5013504 Flops/token = 657.1 GFlops
常量 KDA Attn = 0.651 GFlops

# 合计
135.1 + 657.1 + 0.651 = 792.851 GFlops
```

## 四、Prefill 计算量 (FLOPs)

**128K 序列 = 17707.8 + 12667.7 + 85.4 = 30460.9 TFlops ≈ 30.5 PFlops**

> 因果掩码：MLA 全量注意力只算下三角 ≈ seq²/2——Kimi-K3 无稀疏 topk，**平方项不可约**（对比 GLM-5.3 经 DSA top-2048 后 MLA 退化为线性项）
> MLA prefill 采用 MHA 展开形态（对齐参考实现）：每 head K = 128+64 = 192 维、V = 128 维

### 4.1 激活权重 GEMM

与 Decoder 3.1 相同，计算量随序列长度线性增长 (≈ seq)

`131072 tokens x (55.4B x 2 x BF16(1) + 48.6B x 2 x FP4(0.25)) = 17707.8 TFlops`

> 备注：全 BF16 反量化口径 `131072 x 208G = 27262.98 TFlops`

### 4.2 Gated MLA Attention (24 层, 全量)

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，prefill 采用 MHA 展开形态，计算量随序列长度**平方**增长（因果下三角 ≈ seq²/2）

输入输出 Shape 如下：

```python
q.shape = [b, Hq(96), seq_q, qk_nope(128)+qk_rope(64)]
k.shape = [b, Hq(96), seq_k, qk_nope(128)+qk_rope(64)]
p.shape = [b, Hq(96), seq_q, seq_k] # P = Q @ K.T
v.shape = [b, Hq(96), seq_k, v_head_dim(128)]
o.shape = [b, Hq(96), seq_q, v_head_dim(128)] # O = P @ V
```

```python
# 每 token 计算量
24 Layers x Hq(96) x seq_q(?) x seq_k(?) x (192+128) x 2 x BF16(1) x causal(0.5) = 737280 Flops/token²

# seq_q = seq_k = 131072，下三角 ÷ 2
131072² token² x 737280 Flops/token² = 12667.7 TFlops
```

> 注：若 prefill 沿用 decode 的吸收形态 (576+512=1088 维/head)，此项为 43067.4 TFlops，是展开形态的 3.4 倍——这正是参考实现 prefill 走 MHA 展开形态的原因

### 4.3 KDA Linear Attention (69 层)

prefill 为分块 (chunk) 并行扫描，计算量随序列长度线性增长，per-token 与递推形态同量级

```python
# 每 token
69 Layers x 3 x H(96) x d(128) x d(128) x 2 = 651.165696 MFlops/token

# 全序列
131072 tokens x 0.651 GFlops/token = 85.4 TFlops
```

### 4.4 计算总量

```python
线性 激活GEMM = 131072 tokens x 135.1 GFlops = 17707.8 TFlops
平方 Gated MLA = 131072² token² x 737280 Flops/token² = 12667.7 TFlops
线性 KDA Attn = 131072 tokens x 0.651 GFlops/token = 85.4 TFlops

# 合计
17707.8 + 12667.7 + 85.4 = 30460.9 TFlops ≈ 30.5 PFlops
```

## 参考

- 模型卡片 (2.8T-A104B, 69 KDA + 24 Gated MLA, LatentMoE 896 选 16) https://modelscope.cn/models/moonshotai/Kimi-K3 (`README.md`)
- 模型参数 https://modelscope.cn/models/moonshotai/Kimi-K3 (`config.json`: `text_config`)
- MXFP4 (E2M1, group=32) 仅路由专家量化 Kimi-K3 `config.json` (`quantization_config`: compressed-tensors, `ignore` 含 self_attn/shared_experts/dense mlp/lm_head/vision)
- MLA NoPE (`assert self.use_nope; self.rotary_emb=None`) Kimi-K3 `modeling_kimi_linear.py`
- KDA state 形状与精度 (recurrent FP32 / conv BF16) vllm `vllm/model_executor/layers/mamba/mamba_utils.py` (`kda_state_shape`/`kda_state_dtype`)、vllm `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`
- Kimi Linear 架构论文 (KDA + MLA 3:1 混合) https://arxiv.org/abs/2510.26692
