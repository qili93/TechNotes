# Kimi-K2.7-Code(1T-A32B)：MLA 全量注意力 + INT4 路由专家

模型不含 MTP 共计 61 层：`1 Dense + 60 MoE`，61 层全部为标准 MLA **全量**注意力（无 DSA indexer、无线性注意力）；多模态 kimi_k25 封装（本文仅计 text tower，不含 27 层 400M ViT）

> 与 Kimi-K2.6 的关系：**built upon Kimi K2.6 的 coding 后训练变体**，官方 README 明确 "has the same architecture as Kimi-K2.5/Kimi-K2.6"；两侧 `config.json` 逐字段完全一致（含 INT4 quantization_config），所有提升来自 post-training（thinking token 用量较 K2.6 降约 30%）。因此本文全部数字与 Kimi-K2.6.md 相同，类似 GLM-5.3 之于 GLM-5.2——但 K2.7-Code 连量化格式都未变

## 一、缓存容量 (Bytes)

**128K 序列 = 131072 × 70272 B = 9.211 GB**

### 1.1 Main KV 缓存 (MLA，61 层) = 1152 Bytes/token

KVCache 缓存 `[seq, 1, kv_lora(512)+qk_rope(64)]` BF16 = 1152B/token

> 精度说明：模型 `dtype=bfloat16`，INT4 仅量化路由专家权重且 `kv_cache_scheme=None`，vllm `cache_dtype=auto` 时主 KV 缓存为 BF16 = 576×2 = 1152 Bytes/token；vllm 中 `kimi_k2` 直接映射 DeepseekV3Config，走标准 DeepSeek-V3 MLA 路径（RoPE 施加，YaRN `factor=64 × original 4096 → 262144`）

### 1.2 整模型 Global KV Cache Per Token（MLA）

```shell
# Main KV (MLA, 61 层)
61 Layers × 1152B/token = 70272 Bytes/token
```

> 注：上下文上限 256K (262144)，256K 序列 Global KV = 2 × 9.211 ≈ 18.42 GB；无 indexer 缓存、无线性注意力状态

## 二、访存量 (Bytes)

**128K 序列 = 61 × 131072 × 1152 = 9210.691584 MB ≈ 9.211 GB**

- Main KV (MLA)：`61 Layers x 131072 x 1152B/token = 9210.691584 MB` 全量 Seq 扫描（无 topk 稀疏）

## 三、Decoder 计算量 (FLOPs)

**128K 序列 = 64G + 131072 x 8495104 Flops/token = 1177.5 GFLOPs**

> 论文口径：计算量按 `BF16/FP8/FP4 = 1/0.5/0.25` 折算

### 3.1 激活权重 GEMM

模型激活参数 A32B，INT4 为 weight-only 量化（W4A16），vllm compressed-tensors 走 Marlin 内核**反量化为 BF16 后做 BF16 GEMM**，故真实计算精度为 BF16

计算量：`32B x 2 x BF16(1.0) = 64 GFlops`，Decoder `seq_q=1` 因此是常量

> 备注：若按权重精度口径折算（路由专家 21.1B INT4 + 其余 10.9B BF16），此项为 `21.1B x 2 x FP4(0.25) + 10.9B x 2 x BF16(1) = 32.3 GFlops`；MoE 路由 sigmoid + noaux_tc，384 选 8 + 1 共享

### 3.2 MLA Attention (61 层, 全量)

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，decode 走吸收形态；无稀疏 topk，K/V 为全序列，计算量是变量，随序列长度**线性**增长（decode）

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, kv_lora(512)+qk_rope(64)]
k.shape = [b, seq_k, kv_lora(512)+qk_rope(64)]
p.shape = [b, Hq(64), seq_q, seq_k] # P = Q @ K.T
v.shape = [b, seq_k, kv_lora(512)]
o.shape = [b, Hq(64), seq_q, kv_lora(512)] # O = P @ V
```

```python
# 每 token 计算量
61 Layers x Hq(64) x seq_q(1) x seq_k(?) x (576+512) x 2 x BF16(1) = 8495104 Flops/token

# seq_k = 131072
131072 tokens x 8495104 Flops/token = 1113.5 GFlops
```

### 3.3 计算总量

```python
常量 激活GEMM = 64 GFlops (Marlin 反量化 BF16 口径; 权重口径 32.3G)
变量 MLA Attn = 131072 tokens x 8495104 Flops/token = 1113.5 GFlops

# 合计
64 + 1113.5 = 1177.5 GFlops
```

## 四、Prefill 计算量 (FLOPs)

**128K 序列 = 8388.6 + 21462.6 = 29851.2 TFlops ≈ 29.9 PFlops**

> 因果掩码：MLA 全量注意力只算下三角 ≈ seq²/2——K2.7-Code 无稀疏 topk，**平方项不可约**（对比 GLM-5.3 经 DSA top-2048 后 MLA 退化为线性项）
> MLA prefill 采用 MHA 展开形态（对齐参考实现）：每 head K = 128+64 = 192 维、V = 128 维

### 4.1 激活权重 GEMM

与 Decoder 3.1 相同，A32B 按真实计算的 BF16 计算，计算量随序列长度线性增长 (≈ seq)

`131072 tokens x 32B x 2 x BF16(1.0) = 8388.608 TFlops`

> 备注：权重精度口径 `131072 x 32.3G = 4233.6 TFlops`

### 4.2 MLA Attention (61 层, 全量)

计算 `Q@K.T` 和 `P@V` 计算精度为 BF16，prefill 采用 MHA 展开形态，计算量随序列长度**平方**增长（因果下三角 ≈ seq²/2）

输入输出 Shape 如下：

```python
q.shape = [b, Hq(64), seq_q, qk_nope(128)+qk_rope(64)]
k.shape = [b, Hq(64), seq_k, qk_nope(128)+qk_rope(64)]
p.shape = [b, Hq(64), seq_q, seq_k] # P = Q @ K.T
v.shape = [b, Hq(64), seq_k, v_head_dim(128)]
o.shape = [b, Hq(64), seq_q, v_head_dim(128)] # O = P @ V
```

```python
# 每 token 计算量
61 Layers x Hq(64) x seq_q(?) x seq_k(?) x (192+128) x 2 x BF16(1) x causal(0.5) = 1249280 Flops/token²

# seq_q = seq_k = 131072，下三角 ÷ 2
131072² token² x 1249280 Flops/token² = 21462.6 TFlops
```

> 注：若 prefill 沿用 decode 的吸收形态 (576+512=1088 维/head)，此项为 72969.9 TFlops，是展开形态的 3.4 倍——这正是参考实现 prefill 走 MHA 展开形态的原因

### 4.3 计算总量

```python
线性 激活GEMM = 131072 tokens x 64 GFlops = 8388.608 TFlops
平方 MLA Attn = 131072² token² x 1249280 Flops/token² = 21462.6 TFlops

# 合计
8388.6 + 21462.6 = 29851.2 TFlops ≈ 29.9 PFlops
```

## 参考

- 模型卡片 (built upon Kimi K2.6, thinking token 降 30%) https://modelscope.cn/models/moonshotai/Kimi-K2.7-Code (`README.md`: "same architecture as Kimi-K2.5/Kimi-K2.6")
- 模型参数 https://modelscope.cn/models/moonshotai/Kimi-K2.7-Code (`config.json`: `text_config`, model_type `kimi_k2`，与 K2.6 逐字段一致)
- INT4 (group=32, 对称) 仅路由专家量化 Kimi-K2.7-Code `config.json` (`quantization_config`: compressed-tensors pack-quantized)
- vllm 架构映射 `kimi_k2 → DeepseekV3Config` vllm `vllm/transformers_utils/config.py`
- W4A16 Marlin 反量化 BF16 GEMM vllm `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
