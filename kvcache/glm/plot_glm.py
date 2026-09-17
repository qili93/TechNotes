# -*- coding: utf-8 -*-
"""GLM 系列模型推理开销对比：缓存量 / 访存量 / Decode / Prefill 随序列长度变化

数据来源（my-docs 下同口径公式，计算量按真实计算精度 BF16=1 / FP8=0.5 / FP4=0.25）：
- GLM-5.1.md        744B-A40B  MLA + DSA w/ Top2048（78 层全 indexer，BF16 权重）
- GLM-5.2.md        744B-A40B  MLA + DSA w/ Top2048 + IndexShare（21 Full indexer，BF16 权重）
- GLM-5.3.md        744B-A40B  同 GLM-5.2 base model，权重 FP8 量化（缓存/访存与 5.2 重合）
- GLM-5.3-Flash.md  320B-A18B  KDA + DSA(MLA-NoPE) w/ Top2048@kpool4（FP8 权重）
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEQS = np.array([2048, 8192, 32768, 131072, 524288])
SEQ_LABELS = ["2K", "8K", "32K", "128K", "512K"]

MLA_DECODE_G = 22.246588416   # 78 x 64 x 2048 x (512+64+512) x 2 BF16
MLA_PREFILL_G = 10.468982784  # 78 x 64 x 2048 x (256+256) x 2 BF16


def glm51(N):
    """GLM-5.1：78 层全 indexer，BF16 权重（激活 39.9B）。"""
    cache = N * (78 * 1152 + 78 * 132)
    io = 78 * 2048 * 1152 + 78 * 132 * N
    decode = (79.8 + MLA_DECODE_G) * 1e9 + N * 319488      # 78x32x128x2xFP8(0.5)
    prefill = N * (79.8 + MLA_PREFILL_G) * 1e9 + N**2 * 159744
    return cache, io, decode, prefill


def glm52(N):
    """GLM-5.2：IndexShare 21 Full indexer，BF16 权重（激活 39.3B）。"""
    cache = N * (78 * 1152 + 21 * 132)
    io = 78 * 2048 * 1152 + 21 * 132 * N
    decode = (78.7 + MLA_DECODE_G) * 1e9 + N * 86016       # 21x32x128x2xFP8(0.5)
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
    kda_state = 147619840  # 34 x (96... 64x128x128 FP32 + conv BF16)
    cache = N * (11 * 1024 + 11 * 132 / 4) + kda_state
    io = 11 * 2048 * 1024 + kda_state + N * (11 * 132 / 4)
    decode = (18 + 2.952790016 + 0.213909504) * 1e9 + N * 11264   # 11x32x128x2x0.5/4
    prefill = N * (18 + 1.476395008 + 0.213909504) * 1e9 + N**2 * 5632
    return cache, io, decode, prefill


MODELS = [
    ("GLM-5.1", "744B-A40B BF16", glm51, "#7f7f7f", "o", "-"),
    ("GLM-5.2", "744B-A40B BF16", glm52, "#1f77b4", "s", "-"),
    ("GLM-5.3", "744B-A40B FP8", glm53, "#d62728", "^", "--"),
    ("GLM-5.3-Flash", "320B-A18B FP8", glm53_flash, "#2ca02c", "D", "-"),
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


# 数值标签相对点的位置 (dx, dy, ha)：四条线分别向 左上/右上/左下/右下 扇形散开，避免同 x 处重叠
# GLM-5.2 与 GLM-5.3 在 (a)(b) 完全重合：5.2 标签在点右上、5.3 在点左下，互相错开
LABEL_OFFSETS = {"GLM-5.1": (-8, 8, "right"), "GLM-5.2": (8, 8, "left"),
                 "GLM-5.3": (-8, -10, "right"), "GLM-5.3-Flash": (8, -10, "left")}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    for ax, (idx, title, scale) in zip(axes.flat, PANELS):
        for name, params, fn, color, marker, ls in MODELS:
            values = fn(SEQS)[idx] / scale
            ax.plot(SEQS, values, marker=marker, markersize=6, lw=2, ls=ls, color=color,
                    label=f"{name} ({params})", zorder=3)
            # 每个数据点标注具体数值
            off = LABEL_OFFSETS[name]
            for x, y in zip(SEQS, values):
                ax.annotate(fmt_value(y), xy=(x, y), xytext=off[:2],
                            textcoords="offset points", fontsize=7.5, color=color,
                            ha=off[2], va="bottom" if off[1] > 0 else "top",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.65),
                            annotation_clip=False, zorder=5)
            # 线条末端标注模型名
            ax.annotate(name.replace("GLM-", ""), xy=(SEQS[-1], values[-1]),
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
    axes[0, 0].legend(loc="upper left", fontsize=9.5, framealpha=0.9)

    fig.suptitle("GLM 系列模型推理开销对比", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "口径：计算量按真实计算精度（BF16=1 / FP8=0.5 / FP4=0.25）；"
             "GLM-5.3 与 GLM-5.2 同 base model 仅权重 FP8 量化，缓存/访存曲线重合（5.3 虚线）；"
             "5.3-Flash 缓存含 KDA 状态常数项；四模型上下文均支持 1M",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    out = Path(__file__).with_name("glm_model_comp.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
