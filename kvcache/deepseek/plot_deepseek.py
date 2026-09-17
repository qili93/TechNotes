# -*- coding: utf-8 -*-
"""DeepSeek 系列模型推理开销对比：缓存量 / 访存量 / Decode / Prefill 随序列长度变化

数据来源（my-docs 下同口径公式，计算量按真实计算精度 BF16=1 / FP8=0.5 / FP4=0.25）：
- DeepSeek-V3.2.md        671B-A37B  MLA + DSA w/ Top2048
- DeepSeek-V4-Flash.md    284B-A13B  CSA + HCA w/ Top512
- DeepSeek-V4-Pro.md      1.6T-A49B  CSA + HCA w/ Top1024
- DeepSeek-V4.1-Flash.md  552B-A16B  CSA2 + CED w/ Top512（Prefill 含 Decoder SWA Bounded Replay）
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEQS = np.array([2048, 8192, 32768, 131072, 524288])
SEQ_LABELS = ["2K", "8K", "32K", "128K", "512K"]


def v32(N):
    """DeepSeek-V3.2，N 为 token 数（numpy 数组）。"""
    cache = N * 48068
    io = 61 * 2048 * 656 + 61 * 132 * N
    decode = (37 + 34.795945984) * 1e9 + N * 499712
    prefill = N * (37 + 10.23410176) * 1e9 + N**2 / 2 * 499712
    return cache, io, decode, prefill


def v4_flash(N):
    """DeepSeek-V4-Flash。"""
    cache = N * 3850.25 + 43 * 128 * 584
    io = (21 * 512 * 584 + 43 * 128 * 584) + N * (20 * 584 / 128 + 21 * 132 / 4)
    decode = (13 + 1.409286144 + 0.721420288) * 1e9 + N * (20480 + 43008)
    prefill = N * (13 + 1.409286144 + 0.721420288) * 1e9 + N**2 * (10240 + 21504)
    return cache, io, decode, prefill


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


MODELS = [
    ("DeepSeek-V3.2", "671B-A37B", v32, "#7f7f7f", "o", 131072),
    ("DeepSeek-V4-Flash", "284B-A13B", v4_flash, "#1f77b4", "s", 524288),
    ("DeepSeek-V4-Pro", "1.6T-A49B", v4_pro, "#d62728", "^", 524288),
    ("DeepSeek-V4.1-Flash", "552B-A16B", v41_flash, "#2ca02c", "D", 524288),
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
LABEL_OFFSETS = {"DeepSeek-V3.2": (-8, 8, "right"), "DeepSeek-V4-Pro": (8, 8, "left"),
                 "DeepSeek-V4-Flash": (-8, -10, "right"), "DeepSeek-V4.1-Flash": (8, -10, "left")}
# 个别面板的覆盖：Prefill 中 V4-Pro 与 V3.2 在 128K 交叉，红线标签改到下方
PANEL_LABEL_OVERRIDES = {3: {"DeepSeek-V4-Pro": (8, -10, "left")}}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    for ax, (idx, title, scale) in zip(axes.flat, PANELS):
        for name, params, fn, color, marker, max_seq in MODELS:
            seqs = SEQS[SEQS <= max_seq]
            values = fn(seqs)[idx] / scale
            ax.plot(seqs, values, marker=marker, markersize=6, lw=2, color=color,
                    label=f"{name} ({params})", zorder=3)
            # 每个数据点标注具体数值
            off = PANEL_LABEL_OVERRIDES.get(idx, {}).get(name, LABEL_OFFSETS[name])
            for x, y in zip(seqs, values):
                ax.annotate(fmt_value(y), xy=(x, y), xytext=off[:2],
                            textcoords="offset points", fontsize=7.5, color=color,
                            ha=off[2], va="bottom" if off[1] > 0 else "top",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                      ec="none", alpha=0.65),
                            annotation_clip=False, zorder=5)
            # 线条末端标注模型名
            offset = (12, 0) if seqs[-1] == SEQS[-1] else (12, 6)
            ax.annotate(name.replace("DeepSeek-", ""), xy=(seqs[-1], values[-1]),
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

    fig.suptitle("DeepSeek 系列模型推理开销对比", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "口径：计算量按真实计算精度（BF16=1 / FP8=0.5 / FP4=0.25）；"
             "V4.1-Flash Prefill 含 Decoder SWA Bounded Replay；V3.2 上下文上限 128K",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    out = Path(__file__).with_name("deepseek_comp.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
