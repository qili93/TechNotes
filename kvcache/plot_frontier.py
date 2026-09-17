# -*- coding: utf-8 -*-
"""跨系列旗舰模型推理开销对比：缓存量 / 访存量 / Decode / Prefill 随序列长度变化

数据来源（my-docs 下同口径公式，计算量按真实计算精度 BF16=1 / FP8=0.5 / FP4=0.25）：
- DeepSeek-V4-Pro.md      1.6T-A49B  CSA + HCA w/ Top1024
- DeepSeek-V4.1-Flash.md  552B-A16B  CSA2 + CED w/ Top512（Prefill 含 Decoder SWA Bounded Replay）
- GLM-5.2.md              744B-A40B  MLA + DSA w/ Top2048 + IndexShare（BF16 权重）
- GLM-5.3.md              744B-A40B  同 GLM-5.2 base model，权重 FP8 量化（缓存/访存与 5.2 重合）
- GLM-5.3-Flash.md        320B-A18B  KDA + DSA(MLA-NoPE) w/ Top2048@kpool4（FP8 权重）
- Kimi-K3.md              2.8T-A104B KDA + Gated MLA 全量注意力（MXFP4 路由专家，Prefill 含 seq² 项）
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEQS = np.array([2048, 8192, 32768, 131072, 524288])
SEQ_LABELS = ["2K", "8K", "32K", "128K", "512K"]

MLA_DECODE_G = 22.246588416   # GLM-5.2/5.3: 78 x 64 x 2048 x (512+64+512) x 2 BF16
MLA_PREFILL_G = 10.468982784  # GLM-5.2/5.3: 78 x 64 x 2048 x (256+256) x 2 BF16


def v4_pro(N):
    """DeepSeek-V4-Pro。"""
    cache = N * 5511.4375 + 61 * 128 * 584
    io = (30 * 1024 * 584 + 61 * 128 * 584) + N * (31 * 584 / 128 + 30 * 132 / 4)
    decode = (49 + 8.05306368 + 2.046820352) * 1e9 + N * (63488 + 61440)
    prefill = N * (49 + 8.05306368 + 2.046820352) * 1e9 + N**2 * (31744 + 30720)
    return cache, io, decode, prefill


def v41_flash(N):
    """DeepSeek-V4.1-Flash（Prefill 按已实现口径：Encoder 全序列 + Decoder bounded 128）。"""
    cache = N * 890 + 40 * 128 * 528
    io = (38 * 512 * 288 + 4 * 16384 * 68 + 40 * 128 * 528) + N * (3 * 68 / 2 + 68)
    decode = (16 + 2.550136832 + 0.67108864 + 0.134217728) * 1e9 + N * 5120
    prefill = (
        (N * 8 + 128 * 8) * 1e9
        + N * 1207.959552e6 + 128 * 1342.17728e6
        + (N + 128) * 335.54432e6
        + N**2 * 1536 + 128 * N * 2048 + 128 * 134.217728e6
    )
    return cache, io, decode, prefill


def glm52(N):
    """GLM-5.2：IndexShare 21 Full indexer，BF16 权重（激活 39.3B）。"""
    cache = N * (78 * 1152 + 21 * 132)
    io = 78 * 2048 * 1152 + 21 * 132 * N
    decode = (78.7 + MLA_DECODE_G) * 1e9 + N * 86016
    prefill = N * (78.7 + MLA_PREFILL_G) * 1e9 + N**2 * 43008
    return cache, io, decode, prefill


def glm53(N):
    """GLM-5.3：同 5.2 结构，权重 FP8（GEMM 0.5 口径）；缓存/访存与 5.2 完全一致。"""
    cache = N * (78 * 1152 + 21 * 132)
    io = 78 * 2048 * 1152 + 21 * 132 * N
    decode = (39.3 + MLA_DECODE_G) * 1e9 + N * 86016
    prefill = N * (39.3 + MLA_PREFILL_G) * 1e9 + N**2 * 43008
    return cache, io, decode, prefill


def glm53_flash(N):
    """GLM-5.3-Flash：11 DSA(kpool=4) + 34 KDA；KDA 状态为常数项。"""
    kda_state = 147619840
    cache = N * (11 * 1024 + 11 * 132 / 4) + kda_state
    io = 11 * 2048 * 1024 + kda_state + N * (11 * 132 / 4)
    decode = (18 + 2.952790016 + 0.213909504) * 1e9 + N * 11264
    prefill = N * (18 + 1.476395008 + 0.213909504) * 1e9 + N**2 * 5632
    return cache, io, decode, prefill


def kimi_k3(N):
    """Kimi-K3：24 Gated MLA 全量（无稀疏）+ 69 KDA；MXFP4 路由专家（48.6B FP4 + 55.4B BF16）。"""
    kda_state = 449372160
    cache = N * (24 * 1152) + kda_state
    io = 24 * 1152 * N + kda_state
    decode = (135.1 + 0.651165696) * 1e9 + N * 5013504      # 24x96x(576+512)x2 BF16
    prefill = N * (135.1 + 0.651165696) * 1e9 + N**2 * 737280  # 24x96x(192+128)x2x0.5
    return cache, io, decode, prefill


MODELS = [
    ("DeepSeek-V4-Pro", "1.6T-A49B", v4_pro, "#d62728", "^", "-"),
    ("DeepSeek-V4.1-Flash", "552B-A16B", v41_flash, "#2ca02c", "D", "-"),
    ("GLM-5.2", "744B-A40B BF16", glm52, "#1f77b4", "s", "-"),
    ("GLM-5.3", "744B-A40B FP8", glm53, "#9467bd", "v", "--"),
    ("GLM-5.3-Flash", "320B-A18B FP8", glm53_flash, "#8c564b", "P", "-"),
    ("Kimi-K3", "2.8T-A104B MXFP4", kimi_k3, "#ff7f0e", "X", "-"),
]

PANELS = [
    (0, "(a) 缓存容量 (GB)", 1e9),
    (1, "(b) 访存量 (MB)", 1e6),
    (2, "(c) Decode 计算量 (GFLOPs)", 1e9),
    (3, "(d) Prefill 计算量 (TFLOPs)", 1e12),
]


def fmt_value(v):
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.2f}"
    return f"{v:.3f}"


# 数值标签相对点的位置 (dx, dy, ha)：六条线向不同象限散开；GLM-5.2/5.3 在 (a)(b) 重合，分别取右上/左下
LABEL_OFFSETS = {
    "DeepSeek-V4-Pro": (8, 8, "left"),
    "DeepSeek-V4.1-Flash": (-8, -12, "right"),
    "GLM-5.2": (-8, 8, "right"),
    "GLM-5.3": (8, -10, "left"),
    "GLM-5.3-Flash": (-8, 12, "right"),
    "Kimi-K3": (10, -12, "left"),
}
# 面板级覆盖：Decode 中 V4-Pro 与 GLM-5.3 在 128K 接近，V4-Pro 标签上移避让
PANEL_LABEL_OVERRIDES = {2: {"DeepSeek-V4-Pro": (-8, 12, "right")}}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), sharex=True)

    for ax, (idx, title, scale) in zip(axes.flat, PANELS):
        for name, params, fn, color, marker, ls in MODELS:
            values = fn(SEQS)[idx] / scale
            ax.plot(SEQS, values, marker=marker, markersize=6, lw=2, ls=ls, color=color,
                    label=f"{name} ({params})", zorder=3)
            # 每个数据点标注具体数值
            off = PANEL_LABEL_OVERRIDES.get(idx, {}).get(name, LABEL_OFFSETS[name])
            for x, y in zip(SEQS, values):
                ax.annotate(fmt_value(y), xy=(x, y), xytext=off[:2],
                            textcoords="offset points", fontsize=7.5, color=color,
                            ha=off[2], va="bottom" if off[1] > 0 else "top",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.65),
                            annotation_clip=False, zorder=5)
            # 线条末端标注模型名
            short = name.replace("DeepSeek-", "DS-").replace("GLM-", "G")
            ax.annotate(short, xy=(SEQS[-1], values[-1]),
                        xytext=(12, 0), textcoords="offset points", color=color,
                        fontsize=9, fontweight="bold", va="center",
                        annotation_clip=False, zorder=4)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=13, pad=8)
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.35, zorder=0)
        ax.tick_params(labelsize=10)

    for ax in axes[1]:
        ax.set_xticks(SEQS)
        ax.set_xticklabels(SEQ_LABELS)
        ax.set_xlabel("序列长度", fontsize=11)
    axes[0, 0].set_xlim(SEQS[0] * 0.75, SEQS[-1] * 1.05)
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("跨系列旗舰模型推理开销对比 (DeepSeek / GLM / Kimi)", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "口径：计算量按真实计算精度（BF16=1 / FP8=0.5 / FP4=0.25）；"
             "GLM-5.3 与 GLM-5.2 缓存/访存重合（5.3 虚线）；V4.1-Flash Prefill 含 Decoder SWA Bounded Replay；"
             "Kimi-K3 全注意力层无稀疏，Decode 线性项 / Prefill 平方项不可约",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    out = Path(__file__).with_name("frontier_comp.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
