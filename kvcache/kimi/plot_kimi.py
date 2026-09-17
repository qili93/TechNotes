# -*- coding: utf-8 -*-
"""Kimi 系列模型推理开销对比：缓存量 / 访存量 / Decode / Prefill 随序列长度变化

数据来源（my-docs 下同口径公式，计算量按真实计算精度 BF16=1 / FP8=0.5 / FP4=0.25）：
- Kimi-K2.6.md       1T-A32B  MLA 全量注意力（61 层，无稀疏），INT4 路由专家（Marlin 反量化按 BF16 计）
- Kimi-K2.7-Code.md  1T-A32B  built upon K2.6 的后训练变体，config 逐字段一致（四图曲线完全重合，虚线）
- Kimi-K3.md         2.8T-A104B  KDA(69) + Gated MLA(24) 全量，MXFP4 路由专家（FP4=0.25 口径）

上下文上限：K2.6 / K2.7-Code 为 256K（YaRN 64×4096），曲线画到 128K；K3 为 1M，画到 512K
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEQS = np.array([2048, 8192, 32768, 131072, 524288])
SEQ_LABELS = ["2K", "8K", "32K", "128K", "512K"]


def k26(N):
    """Kimi-K2.6：61 层全量 MLA（64 头），无稀疏；INT4 Marlin 反量化 BF16 口径。"""
    cache = N * (61 * 1152)
    io = 61 * 1152 * N
    decode = 64 * 1e9 + N * 8495104          # 61x64x(576+512)x2 BF16
    prefill = N * 64 * 1e9 + N**2 * 1249280  # 61x64x(192+128)x2x0.5
    return cache, io, decode, prefill


def k27_code(N):
    """Kimi-K2.7-Code：与 K2.6 同 base model（config 逐字段一致），曲线完全重合。"""
    return k26(N)


def kimi_k3(N):
    """Kimi-K3：24 Gated MLA 全量 + 69 KDA；MXFP4 路由专家（48.6B FP4 + 55.4B BF16）。"""
    kda_state = 449372160
    cache = N * (24 * 1152) + kda_state
    io = 24 * 1152 * N + kda_state
    decode = (135.1 + 0.651165696) * 1e9 + N * 5013504
    prefill = N * (135.1 + 0.651165696) * 1e9 + N**2 * 737280
    return cache, io, decode, prefill


MODELS = [
    ("Kimi-K2.6", "1T-A32B INT4", k26, "#7f7f7f", "o", "-", 262144),
    ("Kimi-K2.7-Code", "1T-A32B INT4", k27_code, "#1f77b4", "s", "--", 262144),
    ("Kimi-K3", "2.8T-A104B MXFP4", kimi_k3, "#ff7f0e", "X", "-", 1048576),
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


# 数值标签相对点的位置 (dx, dy, ha)：K2.6 与 K2.7-Code 四图完全重合，分别取左上/右下错开
LABEL_OFFSETS = {"Kimi-K2.6": (-8, 8, "right"), "Kimi-K2.7-Code": (8, -10, "left"),
                 "Kimi-K3": (8, 8, "left")}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    for ax, (idx, title, scale) in zip(axes.flat, PANELS):
        for name, params, fn, color, marker, ls, max_seq in MODELS:
            seqs = SEQS[SEQS <= max_seq]
            values = fn(seqs)[idx] / scale
            ax.plot(seqs, values, marker=marker, markersize=6, lw=2, ls=ls, color=color,
                    label=f"{name} ({params})", zorder=3)
            # 每个数据点标注具体数值
            off = LABEL_OFFSETS[name]
            for x, y in zip(seqs, values):
                ax.annotate(fmt_value(y), xy=(x, y), xytext=off[:2],
                            textcoords="offset points", fontsize=7.5, color=color,
                            ha=off[2], va="bottom" if off[1] > 0 else "top",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.65),
                            annotation_clip=False, zorder=5)
            # 线条末端标注模型名
            offset = (12, 0) if seqs[-1] == SEQS[-1] else (12, 6)
            ax.annotate(name.replace("Kimi-", ""), xy=(seqs[-1], values[-1]),
                        xytext=offset, textcoords="offset points", color=color,
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

    fig.suptitle("Kimi 系列模型推理开销对比", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "口径：计算量按真实计算精度（BF16=1 / FP8=0.5 / FP4=0.25），K2.x INT4 Marlin 反量化按 BF16 计；"
             "K2.7-Code 与 K2.6 同 base model（四图曲线重合，虚线）；K2.x 上限 256K、K3 上限 1M",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    out = Path(__file__).with_name("kimi_model_comp.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
