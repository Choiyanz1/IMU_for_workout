"""Draw a high-level IMU workout pipeline architecture diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("artifacts/figures/current_model_architecture")


COLORS = {
    "bg": "#F8FAFC",
    "ink": "#0F172A",
    "muted": "#64748B",
    "line": "#334155",
    "input": "#DCFCE7",
    "input_edge": "#16A34A",
    "gate": "#DBEAFE",
    "gate_edge": "#2563EB",
    "action": "#EDE9FE",
    "action_edge": "#7C3AED",
    "phase": "#FEF3C7",
    "phase_edge": "#D97706",
    "decoder": "#FFE4E6",
    "decoder_edge": "#E11D48",
    "output": "#E0F2FE",
    "output_edge": "#0284C7",
    "pending": "#F1F5F9",
    "pending_edge": "#64748B",
}


def add_box(ax, x, y, w, h, title, subtitle, fill, edge, dashed=False):
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
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["ink"], zorder=3)
    ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=9, color=COLORS["muted"], zorder=3)


def add_arrow(ax, start, end, color=None, dashed=False, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.1,
        color=color or COLORS["line"],
        linestyle="--" if dashed else "-",
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

    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(0.55, 6.52, "IMU Workout Recognition + Rep Segmentation", fontsize=19, fontweight="bold", color=COLORS["ink"], ha="left")
    ax.text(0.55, 6.15, "High-level streaming architecture: action context runs in parallel with C/E phase segmentation.", fontsize=10.5, color=COLORS["muted"], ha="left")

    add_box(ax, 0.60, 3.10, 1.85, 1.10, "6-axis IMU", "100 Hz stream", COLORS["input"], COLORS["input_edge"])
    add_box(ax, 3.05, 3.10, 2.10, 1.10, "Active Gate", "remove rest/prep", COLORS["gate"], COLORS["gate_edge"])

    add_box(ax, 5.95, 4.55, 2.55, 1.05, "Action Recognition", "set/action context", COLORS["action"], COLORS["action_edge"], dashed=True)
    add_box(ax, 5.95, 2.00, 2.55, 1.05, "C/E Phase Model", "causal IMU segmentation", COLORS["phase"], COLORS["phase_edge"])

    add_box(ax, 9.25, 3.10, 2.05, 1.10, "Rep Decoder", "count + boundaries", COLORS["decoder"], COLORS["decoder_edge"])
    add_box(ax, 11.85, 3.10, 1.25, 1.10, "Outputs", "live feedback", COLORS["output"], COLORS["output_edge"])

    add_arrow(ax, (2.45, 3.65), (3.05, 3.65), COLORS["input_edge"])
    add_arrow(ax, (5.15, 3.65), (5.95, 5.05), COLORS["action_edge"], dashed=True, rad=0.18)
    add_arrow(ax, (5.15, 3.65), (5.95, 2.55), COLORS["phase_edge"], rad=-0.12)
    add_arrow(ax, (8.50, 5.05), (9.25, 3.85), COLORS["action_edge"], dashed=True, rad=-0.15)
    add_arrow(ax, (8.50, 2.55), (9.25, 3.45), COLORS["phase_edge"], rad=0.12)
    add_arrow(ax, (11.30, 3.65), (11.85, 3.65), COLORS["output_edge"])

    ax.text(7.20, 5.86, "pending integration", fontsize=8.3, color=COLORS["action_edge"], ha="center", fontweight="bold")
    ax.text(7.22, 1.68, "validated core", fontsize=8.3, color=COLORS["phase_edge"], ha="center", fontweight="bold")
    ax.text(10.28, 2.70, "uses action context", fontsize=8.3, color=COLORS["decoder_edge"], ha="center")

    add_tag(ax, 3.85, 5.85, "rest-aware gate still needs full-session validation", COLORS["pending_edge"])
    add_tag(ax, 10.35, 5.85, "rep-first action recognition is not the primary path", COLORS["action_edge"])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"current_model_architecture.{ext}", dpi=240)
    plt.close(fig)
    print(f"Saved architecture figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
