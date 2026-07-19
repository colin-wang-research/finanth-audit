from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from tueplots import bundles


FULL_WIDTH = 6.95
SINGLE_WIDTH = 3.35

PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light_gray": "#D1D5DB",
    "dark": "#1F2937",
    "white": "#FFFFFF",
}

RULE_COLORS = {
    "No Action": PALETTE["gray"],
    "Direct Prior": PALETTE["dark"],
    "Confidence Gate": PALETTE["sky"],
    "Uncertainty Gate": PALETTE["purple"],
    "Risk Filter": PALETTE["orange"],
    "Cost-Aware Gate": PALETTE["vermillion"],
    "Hard Role Gate": PALETTE["green"],
    "Lifecycle Checklist": PALETTE["blue"],
    "No Role Gate": PALETTE["dark"],
    "Shared Threshold": PALETTE["orange"],
    "Soft Penalty": PALETTE["yellow"],
    "Provenance Hard Gate": PALETTE["green"],
    "Provenance Learned Gate": PALETTE["sky"],
    "EPV Adapter": PALETTE["purple"],
}


def apply_style() -> None:
    settings = bundles.neurips2024(
        usetex=False, rel_width=1.0, nrows=1, ncols=2, family="sans-serif"
    )
    settings.update(
        {
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.facecolor": PALETTE["white"],
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.24,
            "lines.linewidth": 1.5,
            "lines.markersize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "finauth-audit-v0.6.0",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    mpl.rcParams.update(settings)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.10,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9.5,
        va="bottom",
        ha="left",
    )


def save_figure(fig: plt.Figure, output_base: Path) -> dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 360}),
    ):
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, **kwargs)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        hashes[suffix] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=PALETTE["light_gray"], zorder=0)
    ax.set_axisbelow(True)


def figure_text_qa(fig: plt.Figure, minimum_font_points: float = 8.0) -> dict[str, object]:
    """Fail when text, legends, data marks, or plot boundaries collide."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    too_small = [
        {"text": artist.get_text(), "font_points": float(artist.get_fontsize())}
        for artist in visible_text
        if float(artist.get_fontsize()) + 1e-9 < minimum_font_points
    ]
    qa_labels = [artist for artist in visible_text if artist.get_gid() == "qa_label"]
    overlaps: list[dict[str, str]] = []
    for index, left in enumerate(visible_text):
        left_bbox = Text.get_window_extent(left, renderer=renderer)
        for right in visible_text[index + 1 :]:
            right_bbox = Text.get_window_extent(right, renderer=renderer)
            if left_bbox.overlaps(right_bbox):
                overlaps.append({"left": left.get_text(), "right": right.get_text()})
    point_overlaps: list[dict[str, object]] = []
    for label in qa_labels:
        axes = label.axes
        if axes is None:
            continue
        label_bbox = Text.get_window_extent(label, renderer=renderer).padded(1.5)
        for point in getattr(axes, "_finauth_qa_points", []):
            display_x, display_y = axes.transData.transform(point)
            marker_bbox = Bbox.from_bounds(display_x - 4.5, display_y - 4.5, 9.0, 9.0)
            if label_bbox.overlaps(marker_bbox):
                point_overlaps.append(
                    {"label": label.get_text(), "point": [float(point[0]), float(point[1])]}
                )
    boundary_violations: list[dict[str, str]] = []
    for label in qa_labels:
        axes = label.axes
        if axes is None:
            continue
        label_bbox = Text.get_window_extent(label, renderer=renderer)
        axes_bbox = axes.get_window_extent(renderer=renderer)
        margin = 1.5
        if not (
            label_bbox.x0 >= axes_bbox.x0 + margin
            and label_bbox.x1 <= axes_bbox.x1 - margin
            and label_bbox.y0 >= axes_bbox.y0 + margin
            and label_bbox.y1 <= axes_bbox.y1 - margin
        ):
            boundary_violations.append(
                {"label": label.get_text(), "axes_title": axes.get_title()}
            )
    legend_axes_overlaps: list[dict[str, str]] = []
    for legend in fig.findobj(match=Legend):
        if not legend.get_visible() or legend.axes is None:
            continue
        legend_bbox = legend.get_window_extent(renderer=renderer)
        axes_bbox = legend.axes.get_window_extent(renderer=renderer)
        intersection = Bbox.intersection(legend_bbox, axes_bbox)
        if intersection is not None and intersection.width * intersection.height > 1.0:
            legend_axes_overlaps.append(
                {"axes_title": legend.axes.get_title(), "legend": " | ".join(text.get_text() for text in legend.get_texts())}
            )
    if too_small or overlaps or point_overlaps or boundary_violations or legend_axes_overlaps:
        raise ValueError(
            "Figure text QA failed: "
            f"too_small={too_small}, text_overlaps={overlaps}, "
            f"label_point_overlaps={point_overlaps}, "
            f"label_boundary_violations={boundary_violations}, "
            f"legend_axes_overlaps={legend_axes_overlaps}"
        )
    return {
        "minimum_font_points": min(float(artist.get_fontsize()) for artist in visible_text),
        "visible_text_count": len(visible_text),
        "qa_label_count": len(qa_labels),
        "all_text_overlap_count": 0,
        "qa_label_overlap_count": 0,
        "qa_label_point_overlap_count": 0,
        "qa_label_boundary_violation_count": 0,
        "legend_axes_overlap_count": 0,
    }
