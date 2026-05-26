"""Draw a high-level C/E phase segmentation model architecture diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("artifacts/figures/phase_segment_model_architecture")


COLORS = {
    "bg": "#F8FAFC",
    "ink": "#0F172A",
    "muted": "#64748B",
    "line": "#334155",
    "input": "#DCFCE7",
    "input_edge": "#16A34A",
    "norm": "#DBEAFE",
    "norm_edge": "#2563EB",
    "model": "#FEF3C7",
    "model_edge": "#D97706",
    "decode": "#FFE4E6",
    "decode_edge": "#E11D48",
    "output": "#E0F2FE",
    "output_edge": "#0284C7",
    "tag": "#64748B",
}


def add_box(ax, x, y, w, h, title, subtitle, fill, edge):
    shadow = FancyBboxPatch(
        (x + 0.04, y - 0.04),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.12",
        linewidth=0,
        facecolor="#CBD5E1",
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.12",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["ink"], zorder=3)
    ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=9, color=COLORS["muted"], zorder=3)


def add_arrow(ax, start, end, color=None, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.1,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=0,
    )
    ax.add_patch(arrow)


def add_tag(ax, x, y, text, fill):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8,
        color="#FFFFFF",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.33", facecolor=fill, edgecolor="none"),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 180, "savefig.bbox": "tight"})

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(0.55, 5.95, "C/E Phase Segmentation Model", fontsize=19, fontweight="bold", color=COLORS["ink"], ha="left")
    ax.text(0.55, 5.58, "Compact view of the validated raw6 causal CNN phase model and its lightweight decoder.", fontsize=10.5, color=COLORS["muted"], ha="left")

    add_box(ax, 0.60, 2.65, 1.95, 1.10, "Active IMU", "6-axis window", COLORS["input"], COLORS["input_edge"])
    add_box(ax, 3.10, 2.65, 2.05, 1.10, "Normalize", "train-fold z-score", COLORS["norm"], COLORS["norm_edge"])
    add_box(ax, 5.80, 2.65, 2.35, 1.10, "Causal CNN", "temporal encoder", COLORS["model"], COLORS["model_edge"])
    add_box(ax, 8.85, 2.65, 1.95, 1.10, "C/E Logits", "per-sample phase", COLORS["model"], COLORS["model_edge"])
    add_box(ax, 11.25, 2.65, 1.65, 1.10, "Decoder", "smooth + Viterbi", COLORS["decode"], COLORS["decode_edge"])

    add_box(ax, 4.65, 0.75, 2.20, 1.00, "Phase Segments", "concentric / eccentric", COLORS["output"], COLORS["output_edge"])
    add_box(ax, 7.45, 0.75, 2.20, 1.00, "Rep Structure", "boundaries + C/E ratio", COLORS["output"], COLORS["output_edge"])

    add_arrow(ax, (2.55, 3.20), (3.10, 3.20), COLORS["input_edge"])
    add_arrow(ax, (5.15, 3.20), (5.80, 3.20), COLORS["norm_edge"])
    add_arrow(ax, (8.15, 3.20), (8.85, 3.20), COLORS["model_edge"])
    add_arrow(ax, (10.80, 3.20), (11.25, 3.20), COLORS["decode_edge"])
    add_arrow(ax, (12.08, 2.65), (5.75, 1.75), COLORS["output_edge"], rad=0.12)
    add_arrow(ax, (6.85, 1.25), (7.45, 1.25), COLORS["output_edge"])

    add_tag(ax, 2.02, 4.70, "input: ax ay az gx gy gz", COLORS["input_edge"])
    add_tag(ax, 6.98, 4.70, "causal: no future samples", COLORS["model_edge"])
    add_tag(ax, 10.55, 4.70, "current core model", COLORS["decode_edge"])

    ax.text(6.98, 2.25, "5 temporal conv blocks", fontsize=8.5, color=COLORS["muted"], ha="center")
    ax.text(6.98, 2.03, "dilations capture short-to-long rep dynamics", fontsize=8.5, color=COLORS["muted"], ha="center")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"phase_segment_model_architecture.{ext}", dpi=240)
    plt.close(fig)
    print(f"Saved phase model figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
