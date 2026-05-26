"""Draw the C/E causal CNN architecture with the correct residual path."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("artifacts/figures/ce_model_architecture_blocks")

COLORS = {
    "bg": "#F8FAFC",
    "ink": "#0F172A",
    "muted": "#64748B",
    "line": "#334155",
    "input": "#DCFCE7",
    "input_edge": "#16A34A",
    "norm": "#DBEAFE",
    "norm_edge": "#2563EB",
    "cnn_panel": "#FFFBEB",
    "cnn_edge": "#D97706",
    "conv": "#FEF3C7",
    "head": "#FFE4E6",
    "head_edge": "#E11D48",
    "output": "#E0F2FE",
    "output_edge": "#0284C7",
    "residual": "#7C3AED",
}


def add_box(ax, x, y, w, h, title, subtitle, fill, edge, title_size=10.5):
    shadow = FancyBboxPatch(
        (x + 0.03, y - 0.03),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        linewidth=0,
        facecolor="#CBD5E1",
        alpha=0.28,
        zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        linewidth=1.7,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=COLORS["ink"], zorder=3)
    ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=8.0, color=COLORS["muted"], zorder=3)


def add_arrow(ax, start, end, color=None, dashed=False, lw=2.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        color=color or COLORS["line"],
        linestyle="--" if dashed else "-",
        connectionstyle="angle3,angleA=0,angleB=90" if start[1] != end[1] else "arc3,rad=0",
        zorder=4,
    )
    ax.add_patch(arrow)


def add_plus(ax, x, y):
    circ = Circle((x, y), 0.18, facecolor="#FFFFFF", edgecolor=COLORS["residual"], linewidth=2.0, zorder=5)
    ax.add_patch(circ)
    ax.text(x, y - 0.005, "+", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["residual"], zorder=6)


def add_pill(ax, x, y, text, fill):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8,
        color="#FFFFFF",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=fill, edgecolor="none"),
        zorder=7,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 180, "savefig.bbox": "tight"})

    fig, ax = plt.subplots(figsize=(16.2, 6.4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(0.55, 5.95, "C/E Phase Segmentation Model", fontsize=20, fontweight="bold", color=COLORS["ink"], ha="left")
    ax.text(0.55, 5.58, "Raw6 IMU window -> normalized input -> 5-layer causal CNN with input residual projection -> per-sample C/E logits.", fontsize=10.5, color=COLORS["muted"], ha="left")

    main_y = 2.90
    add_box(ax, 0.65, main_y, 1.55, 1.00, "Raw6 IMU", "[6 x 300]", COLORS["input"], COLORS["input_edge"], 10.5)
    add_box(ax, 2.65, main_y, 1.65, 1.00, "Normalize", "z-score", COLORS["norm"], COLORS["norm_edge"], 10.5)

    # CNN panel with five explicit convolution blocks on the main path.
    panel_x, panel_y, panel_w, panel_h = 4.85, 1.55, 6.15, 3.45
    panel = FancyBboxPatch(
        (panel_x, panel_y),
        panel_w,
        panel_h,
        boxstyle="round,pad=0.05,rounding_size=0.16",
        linewidth=2.0,
        edgecolor=COLORS["cnn_edge"],
        facecolor=COLORS["cnn_panel"],
        zorder=1,
    )
    ax.add_patch(panel)
    ax.text(panel_x + panel_w / 2, 4.68, "5-Layer Causal CNN Encoder", fontsize=14.5, fontweight="bold", color=COLORS["ink"], ha="center", zorder=3)
    ax.text(panel_x + panel_w / 2, 4.38, "Conv1d k=5, hidden=64, GroupNorm + ReLU + Dropout", fontsize=8.8, color=COLORS["muted"], ha="center", zorder=3)

    conv_y = main_y
    conv_w, conv_h = 0.82, 1.00
    conv_gap = 0.22
    conv_x0 = panel_x + 0.38
    convs = []
    for idx, dilation in enumerate([1, 2, 4, 8, 16]):
        x = conv_x0 + idx * (conv_w + conv_gap)
        add_box(ax, x, conv_y, conv_w, conv_h, f"Conv{idx + 1}", f"d={dilation}", COLORS["conv"], COLORS["cnn_edge"], 8.8)
        convs.append((x, conv_y, conv_w, conv_h))
        if idx > 0:
            prev_x = convs[idx - 1][0]
            add_arrow(ax, (prev_x + conv_w, main_y + conv_h / 2), (x, main_y + conv_h / 2), COLORS["cnn_edge"], lw=1.5)

    # Correct residual path: normalized input -> 1x1 projection -> add after Conv5.
    proj_x, proj_y = panel_x + 0.65, 2.02
    add_box(ax, proj_x, proj_y, 1.18, 0.58, "1x1", "6 -> 64", "#F5F3FF", COLORS["residual"], 8.8)
    plus_x, plus_y = panel_x + 5.45, main_y + conv_h / 2
    add_plus(ax, plus_x, plus_y)
    add_arrow(ax, (4.30, main_y + 0.50), (proj_x, proj_y + 0.29), COLORS["residual"], dashed=True, lw=1.5)
    add_arrow(ax, (proj_x + 1.18, proj_y + 0.29), (plus_x, proj_y + 0.29), COLORS["residual"], dashed=True, lw=1.5)
    add_arrow(ax, (plus_x, proj_y + 0.29), (plus_x, plus_y - 0.18), COLORS["residual"], dashed=True, lw=1.5)
    add_arrow(ax, (convs[-1][0] + conv_w, plus_y), (plus_x - 0.18, plus_y), COLORS["cnn_edge"], lw=1.5)
    ax.text(panel_x + panel_w / 2, 1.82, "Residual: normalized input is projected with 1x1 Conv and added after Conv5", fontsize=8.3, color=COLORS["residual"], ha="center", zorder=3)

    add_box(ax, 11.60, main_y, 1.45, 1.00, "1x1 Head", "64 -> 2", COLORS["head"], COLORS["head_edge"], 10.5)
    add_box(ax, 13.50, main_y, 1.55, 1.00, "C/E Logits", "[2 x 300]", COLORS["output"], COLORS["output_edge"], 10.5)

    add_arrow(ax, (2.20, main_y + 0.50), (2.65, main_y + 0.50), COLORS["input_edge"])
    add_arrow(ax, (4.30, main_y + 0.50), (4.85, main_y + 0.50), COLORS["norm_edge"])
    add_arrow(ax, (plus_x + 0.18, plus_y), (11.60, main_y + 0.50), COLORS["cnn_edge"])
    add_arrow(ax, (13.05, main_y + 0.50), (13.50, main_y + 0.50), COLORS["output_edge"])

    add_pill(ax, 1.42, 4.55, "ax ay az gx gy gz", COLORS["input_edge"])
    add_pill(ax, 3.48, 4.55, "train-fold stats", COLORS["norm_edge"])
    add_pill(ax, 7.92, 5.18, "no future samples", COLORS["cnn_edge"])
    add_pill(ax, 12.32, 4.55, "model only", COLORS["head_edge"])
    ax.text(14.28, 2.55, "eccentric\nconcentric", fontsize=9, color=COLORS["muted"], ha="center", linespacing=1.3)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"ce_model_architecture_blocks.{ext}", dpi=240)
    plt.close(fig)
    print(f"Saved C/E model block figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
