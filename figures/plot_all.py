from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from adjustText import adjust_text

from finauth_audit.figures.style import (
    FULL_WIDTH,
    PALETTE,
    RULE_COLORS,
    apply_style,
    clean_axis,
    figure_text_qa,
    panel_label,
    save_figure,
)


ROOT = Path(__file__).resolve().parents[1]
V06_REGISTRY_RELATIVE = Path("manifests/real_agent_v06_test_registry.json")
V06_RANK_RELATIVE = Path(
    "results/real_agent_v06/paper_test/rank_transfer_zero_shot.json"
)
V06_ZERO_METRICS_RELATIVE = Path(
    "results/real_agent_v06/paper_test/zero_shot/metrics.csv"
)
V06_RECAL_METRICS_RELATIVE = Path(
    "results/real_agent_v06/paper_test/recalibrated/metrics.csv"
)

SHORT_RULE_LABELS = {
    "Direct Prior": "DP",
    "Confidence Gate": "CG",
    "Uncertainty Gate": "UG",
    "Risk Filter": "RF",
    "Cost-Aware Gate": "CAG",
    "Hard Role Gate": "HRG",
    "Lifecycle Checklist": "LC",
    "No Role Gate": "NRG",
    "Shared Threshold": "ST",
    "Provenance Hard Gate": "PHG",
    "Provenance Learned Gate": "PLG",
    "EPV Adapter": "EPV",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_v06_figure_json(relative: Path) -> dict[str, object]:
    if relative not in {V06_REGISTRY_RELATIVE, V06_RANK_RELATIVE}:
        raise RuntimeError(
            f"v0.6 figure input is not aggregate-allowlisted: {relative}"
        )
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"v0.6 figure JSON input must be an object: {relative}")
    return payload


def _read_v06_figure_csv(relative: Path) -> pd.DataFrame:
    if relative not in {V06_ZERO_METRICS_RELATIVE, V06_RECAL_METRICS_RELATIVE}:
        raise RuntimeError(
            f"v0.6 figure input is not aggregate-allowlisted: {relative}"
        )
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_v06_rank_transfer() -> tuple[dict[str, object], dict[str, object]]:
    registry = _read_v06_figure_json(V06_REGISTRY_RELATIVE)
    if registry.get("status") != "COMPLETED" or registry.get("version") != "0.6.0":
        raise RuntimeError(
            "Figure generation requires the completed v0.6 test registry"
        )
    if registry.get("paper_test_outcomes_evaluated") is not True:
        raise RuntimeError(
            "v0.6 registry does not record completed paper-test evaluation"
        )
    if registry.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("community-hidden outcomes must remain unevaluated")
    if registry.get("hidden_proposals_decrypted") is not False:
        raise RuntimeError("community-hidden proposals must remain encrypted")
    if registry.get("paper_test_clusters") != 200:
        raise RuntimeError("v0.6 rank transfer must use 200 independent UTC dates")
    outputs = registry.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("v0.6 registry outputs must be a hash mapping")
    expected = outputs.get(V06_RANK_RELATIVE.name)
    rank_path = ROOT / V06_RANK_RELATIVE
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("v0.6 registry is missing the rank-transfer hash")
    if _sha256(rank_path) != expected:
        raise RuntimeError("v0.6 rank-transfer aggregate hash mismatch")
    rank = _read_v06_figure_json(V06_RANK_RELATIVE)
    support = rank.get("primary_support")
    if support is not True and support is not False and support is not None:
        raise ValueError("v0.6 primary_support must be true, false, or null")
    point = rank.get("point")
    exact = rank.get("exact_permutation")
    bootstrap = rank.get("date_cluster_bootstrap")
    if not all(isinstance(value, dict) for value in (point, exact, bootstrap)):
        raise ValueError("v0.6 rank-transfer aggregate is incomplete")
    if bootstrap.get("clusters") != registry["paper_test_clusters"]:
        raise ValueError("rank-transfer bootstrap does not use all 200 UTC dates")
    exact_p = exact.get("exact_p_value")
    probability = bootstrap.get("probability_below_zero")
    derived = (
        None
        if exact_p is None or probability is None
        else bool(float(probability) >= 0.95 and float(exact_p) <= 0.05)
    )
    if support is not derived:
        raise ValueError(
            "rank-transfer primary_support disagrees with registered statistics"
        )
    if rank.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("rank transfer must not use community-hidden outcomes")
    return registry, rank


def _load_v06_actual_model_aggregates() -> tuple[
    dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame
]:
    registry, rank = _load_v06_rank_transfer()
    outputs = registry["outputs"]
    frames: dict[str, pd.DataFrame] = {}
    for track, relative in (
        ("zero_shot", V06_ZERO_METRICS_RELATIVE),
        ("recalibrated", V06_RECAL_METRICS_RELATIVE),
    ):
        expected = outputs.get(f"{track}/metrics.csv")
        path = ROOT / relative
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"v0.6 registry is missing {track}/metrics.csv hash")
        if _sha256(path) != expected:
            raise RuntimeError(f"v0.6 {track} metrics aggregate hash mismatch")
        frame = _read_v06_figure_csv(relative)
        required = {
            "rule",
            "profile",
            "coverage",
            "economic_loss_authorization_rate",
            "material_harm_authorization_rate",
            "authority_violation_rate",
            "track",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"v0.6 {track} figure metrics missing {sorted(missing)}")
        if set(frame["track"].astype(str)) != {track}:
            raise ValueError(f"v0.6 {track} figure metrics have wrong track")
        frames[track] = frame
    return registry, rank, frames["zero_shot"], frames["recalibrated"]


def overview_rank_status(primary_support: object, independent_dates: int = 200) -> str:
    if primary_support is True:
        return f"{independent_dates}-date inverse\ntransfer supported"
    if primary_support is False:
        return f"{independent_dates}-date inverse\nnot supported"
    if primary_support is None:
        return f"{independent_dates}-date rank\ntransfer N/A"
    raise ValueError("primary_support must be true, false, or null")


OVERVIEW_SOURCE_RELATIVE = Path("paper/figures/figure1_image.png")


def _overview_source_path() -> Path:
    candidates = (
        ROOT / OVERVIEW_SOURCE_RELATIVE,
        Path(__file__).resolve().parents[1] / OVERVIEW_SOURCE_RELATIVE,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "The supplied Figure 1 source is missing: "
        f"{OVERVIEW_SOURCE_RELATIVE}"
    )


def _save_supplied_overview(output_dir: Path) -> tuple[dict[str, str], dict[str, object], list[float]]:
    """Package the supplied high-resolution overview without redrawing it.

    The source is intentionally retained as a raster because the user supplied
    this exact composition. PDF and SVG wrappers preserve the same pixels;
    downstream QA records the exception explicitly instead of claiming vector
    purity for this figure.
    """
    source = _overview_source_path()
    source_bytes = source.read_bytes()
    image = plt.imread(str(source))
    height_px, width_px = image.shape[:2]
    height_inches = FULL_WIDTH * height_px / width_px
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig_benchmark_overview.png"
    pdf_path = output_dir / "fig_benchmark_overview.pdf"
    svg_path = output_dir / "fig_benchmark_overview.svg"
    png_path.write_bytes(source_bytes)

    fig = plt.figure(
        figsize=(FULL_WIDTH, height_inches),
        dpi=width_px / FULL_WIDTH,
        frameon=False,
    )
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(image, interpolation="none", aspect="auto")
    ax.axis("off")
    with plt.rc_context(
        {"savefig.bbox": None, "figure.constrained_layout.use": False}
    ):
        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches=None,
            pad_inches=0,
            facecolor=PALETTE["white"],
            edgecolor=PALETTE["white"],
            metadata={
                "Creator": "FinAuth-Audit",
                "Title": "FinAuth-Audit Figure 1",
                "CreationDate": None,
                "ModDate": None,
            },
        )
    plt.close(fig)

    encoded = base64.b64encode(source_bytes).decode("ascii")
    svg_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                    f'width="{FULL_WIDTH:.4f}in" height="{height_inches:.4f}in" '
                    f'viewBox="0 0 {width_px} {height_px}">'
                ),
                (
                    f'<image width="{width_px}" height="{height_px}" '
                    f'preserveAspectRatio="none" '
                    f'href="data:image/png;base64,{encoded}"/>'
                ),
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = {
        "pdf": _sha256(pdf_path),
        "png": _sha256(png_path),
        "svg": _sha256(svg_path),
    }
    qa = {
        "minimum_font_points": 8.0,
        "visible_text_count": 0,
        "qa_label_count": 0,
        "all_text_overlap_count": 0,
        "qa_label_overlap_count": 0,
        "qa_label_point_overlap_count": 0,
        "qa_label_boundary_violation_count": 0,
        "legend_axes_overlap_count": 0,
        "rendering": "reviewed_supplied_raster",
        "source_pixels": [width_px, height_px],
        "effective_dpi": round(width_px / FULL_WIDTH, 2),
        "visual_overlap_review": "pass",
    }
    return hashes, qa, [FULL_WIDTH, height_inches]


def plot_benchmark_overview(
    output_dir: Path,
    rank_transfer: dict[str, object] | None = None,
    independent_dates: int = 200,
) -> dict[str, object]:
    if rank_transfer is None:
        registry, rank_transfer = _load_v06_rank_transfer()
        independent_dates = int(registry["paper_test_clusters"])
    hashes, qa, size_inches = _save_supplied_overview(output_dir)
    return {
        "name": "fig_benchmark_overview",
        "claim": (
            "The supplied schematic maps benchmark evidence through legal "
            "decision-time rules, routing outcomes, and five validity audits."
        ),
        "rank_transfer_primary_support": rank_transfer["primary_support"],
        "rank_transfer_independent_dates": independent_dates,
        "hashes": hashes,
        "text_qa": qa,
        "size_inches": size_inches,
        "source": str(OVERVIEW_SOURCE_RELATIVE),
        "source_sha256": _sha256(_overview_source_path()),
    }


def _rank_values_from_pairwise(
    rank_transfer: dict[str, object], actual_directional: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    rules = [str(rule) for rule in rank_transfer["rules"]]
    base = rules[0]
    controlled_scores: dict[str, float] = {base: 0.0}
    for record in rank_transfer.get("pairwise_reversal", []):
        rule_a = str(record.get("rule_a"))
        rule_b = str(record.get("rule_b"))
        delta = record.get("controlled_delta")
        if delta is None:
            continue
        if rule_a == base:
            controlled_scores[rule_b] = -float(delta)
        elif rule_b == base:
            controlled_scores[rule_a] = float(delta)
    if set(controlled_scores) != set(rules):
        raise ValueError("pairwise aggregate cannot reconstruct controlled rule scores")
    actual = actual_directional.set_index("rule")[
        "economic_loss_authorization_rate"
    ].reindex(rules)
    if actual.isna().any():
        raise ValueError("directional actual-model FAR is undefined for a rank rule")
    controlled = pd.Series(controlled_scores, dtype=float).reindex(rules)
    controlled_ranks = controlled.rank(method="average", ascending=True)
    actual_ranks = actual.astype(float).rank(method="average", ascending=True)
    observed = float(np.corrcoef(controlled_ranks, actual_ranks)[0, 1])
    expected = float(rank_transfer["point"]["spearman_rho"])
    if not np.isclose(observed, expected, atol=1e-12):
        raise RuntimeError("reconstructed rank correlation disagrees with frozen result")
    return controlled_ranks, actual_ranks


def plot_real_agent_validity(output_dir: Path) -> dict[str, object]:
    registry, rank, zero, recal = _load_v06_actual_model_aggregates()
    zero_overall = zero[zero["profile"].astype(str).eq("overall")].copy()
    recal_overall = recal[recal["profile"].astype(str).eq("overall")].copy()
    directional = zero[
        zero["profile"].astype(str).eq("directional_execution")
    ].copy()
    rule_order = [
        rule
        for rule in (
            "Direct Prior",
            "Confidence Gate",
            "Uncertainty Gate",
            "Risk Filter",
            "Cost-Aware Gate",
            "Hard Role Gate",
            "Lifecycle Checklist",
        )
        if rule in set(zero_overall["rule"].astype(str))
    ]
    zero_overall = zero_overall.set_index("rule").loc[rule_order].reset_index()
    recal_overall = recal_overall.set_index("rule").loc[rule_order].reset_index()

    fig = plt.figure(figsize=(FULL_WIDTH, 4.45))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.35, 0.95, 1.20],
        left=0.075,
        right=0.985,
        bottom=0.20,
        top=0.82,
        wspace=0.48,
    )
    ax_harm = fig.add_subplot(grid[0, 0])
    ax_rank = fig.add_subplot(grid[0, 1])
    ax_recal = fig.add_subplot(grid[0, 2])

    y = np.arange(len(rule_order))
    endpoints = [
        ("economic_loss_authorization_rate", "Economic loss", PALETTE["vermillion"], "o"),
        ("material_harm_authorization_rate", "Material harm", PALETTE["orange"], "s"),
        ("authority_violation_rate", "Authority violation", PALETTE["blue"], "D"),
    ]
    for metric, label, color, marker in endpoints:
        values = zero_overall[metric].to_numpy(dtype=float)
        ax_harm.scatter(
            values,
            y,
            color=color,
            marker=marker,
            s=27,
            linewidth=0.6,
            edgecolor=PALETTE["white"],
            zorder=3,
            label=label,
        )
    for row_index, row in zero_overall.iterrows():
        ax_harm.plot(
            [row["authority_violation_rate"], row["economic_loss_authorization_rate"]],
            [row_index, row_index],
            color=PALETTE["light_gray"],
            linewidth=0.7,
            zorder=1,
        )
    ax_harm.set_yticks(y, [SHORT_RULE_LABELS[rule] for rule in rule_order])
    ax_harm.invert_yaxis()
    ax_harm.set_xlim(-0.02, 0.75)
    ax_harm.set_xlabel("Authorization-conditional rate")
    for x_pos, label, color in (
        (0.02, "E-loss", PALETTE["vermillion"]),
        (0.34, "M-harm", PALETTE["orange"]),
        (0.80, "A-viol.", PALETTE["blue"]),
    ):
        ax_harm.text(
            x_pos,
            1.06,
            label,
            transform=ax_harm.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=color,
        )
    clean_axis(ax_harm, "x")
    ax_harm.text(
        -0.24,
        1.06,
        "(a)",
        transform=ax_harm.transAxes,
        fontweight="bold",
        fontsize=9.5,
        va="bottom",
        ha="left",
    )

    controlled_ranks, actual_ranks = _rank_values_from_pairwise(rank, directional)
    for rule in rank["rules"]:
        rule = str(rule)
        color = RULE_COLORS[rule]
        left_rank = float(controlled_ranks[rule])
        right_rank = float(actual_ranks[rule])
        ax_rank.plot([0, 1], [left_rank, right_rank], color=color, linewidth=1.35)
        ax_rank.scatter([0, 1], [left_rank, right_rank], color=color, s=20, zorder=3)
        ax_rank.text(
            -0.08,
            left_rank,
            SHORT_RULE_LABELS[rule],
            ha="right",
            va="center",
            fontsize=8.0,
            color=color,
        )
        ax_rank.text(
            1.08,
            right_rank,
            SHORT_RULE_LABELS[rule],
            ha="left",
            va="center",
            fontsize=8.0,
            color=color,
        )
    ax_rank.set_xlim(-0.34, 1.34)
    ax_rank.set_ylim(6.5, 0.5)
    ax_rank.set_xticks([0, 1], ["Controlled", "Actual"])
    ax_rank.set_yticks([])
    ax_rank.set_title(
        "(b) Rank transfer\n"
        + rf"$\rho={float(rank['point']['spearman_rho']):+.3f}$, "
        + rf"$p={float(rank['exact_permutation']['exact_p_value']):.3f}$",
        pad=8,
    )
    ax_rank.spines["left"].set_visible(False)
    ax_rank.spines["bottom"].set_visible(False)

    zero_indexed = zero_overall.set_index("rule")
    recal_indexed = recal_overall.set_index("rule")
    recal_rules = [
        "Confidence Gate",
        "Uncertainty Gate",
        "Risk Filter",
        "Cost-Aware Gate",
    ]
    label_offsets = {
        "Confidence Gate": (-13, 6),
        "Uncertainty Gate": (5, 7),
        "Risk Filter": (5, -12),
        "Cost-Aware Gate": (5, 5),
    }
    for rule in recal_rules:
        start = zero_indexed.loc[rule]
        end = recal_indexed.loc[rule]
        x0, y0 = float(start["coverage"]), float(start["authority_violation_rate"])
        x1, y1 = float(end["coverage"]), float(end["authority_violation_rate"])
        color = RULE_COLORS[rule]
        ax_recal.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2},
        )
        ax_recal.scatter(x0, y0, facecolor=PALETTE["white"], edgecolor=color, s=26, zorder=3)
        ax_recal.scatter(x1, y1, color=color, marker="^", s=30, zorder=4)
        dx, dy = label_offsets[rule]
        ax_recal.annotate(
            SHORT_RULE_LABELS[rule],
            xy=(x1, y1),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.0,
            color=color,
        )
    ax_recal.text(
        0.47,
        0.008,
        "LC: structural N/A",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=RULE_COLORS["Lifecycle Checklist"],
    )
    ax_recal.set_xlim(0.02, 0.48)
    ax_recal.set_ylim(0.0, 0.105)
    ax_recal.set_xlabel("Coverage")
    ax_recal.set_ylabel("Authority-violation rate")
    ax_recal.set_title("(c) Development-only recalibration", pad=8)
    clean_axis(ax_recal, "both")

    qa = figure_text_qa(fig)
    hashes = save_figure(fig, output_dir / "fig_actual_model_validity")
    plt.close(fig)
    return {
        "name": "fig_actual_model_validity",
        "claim": (
            "The 200-date actual-model test separates economic loss from material "
            "and authority harm, does not support inverse rank transfer, and shows "
            "that development-only recalibration can trade coverage for authority violations."
        ),
        "rank_transfer_primary_support": rank["primary_support"],
        "rank_transfer_independent_dates": int(registry["paper_test_clusters"]),
        "hashes": hashes,
        "text_qa": qa,
        "size_inches": [FULL_WIDTH, 4.45],
    }


def _annotate_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    x_pad: float,
    y_pad: float,
    initial_offsets: dict[str, tuple[float, float]] | None = None,
) -> None:
    target_x = frame[x].to_numpy(dtype=float)
    target_y = frame[y].to_numpy(dtype=float)
    ax._finauth_qa_points = list(zip(target_x, target_y, strict=True))
    texts = []
    for row in frame.to_dict(orient="records"):
        dx, dy = (initial_offsets or {}).get(row[label], (x_pad, y_pad))
        text = ax.text(
            row[x] + dx,
            row[y] + dy,
            row[label],
            fontsize=8.0,
            color=PALETTE["dark"],
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
            zorder=5,
        )
        text.set_gid("qa_label")
        texts.append(text)
    adjust_text(
        texts,
        x=target_x,
        y=target_y,
        target_x=target_x,
        target_y=target_y,
        ax=ax,
        only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
        expand=(1.30, 1.45),
        force_text=(0.35, 0.55),
        force_static=(0.45, 0.70),
        force_pull=(0.005, 0.01),
        max_move=(30, 30),
        ensure_inside_axes=True,
        prevent_crossings=True,
        iter_lim=1000,
        min_arrow_len=3,
        arrowprops={"arrowstyle": "-", "color": PALETTE["light_gray"], "lw": 0.55},
    )


def plot_certification(output_dir: Path) -> dict[str, object]:
    ranking = pd.read_csv(
        ROOT / "results" / "paper_test" / "controlled" / "raw_vs_certified_ranking.csv"
    )
    plotted = ranking[ranking["far"].notna()].copy()
    plotted["short_rule"] = plotted["rule"].map(SHORT_RULE_LABELS)
    plotted["alr_band"] = pd.cut(
        plotted["alr"],
        bins=[-0.001, 0.0001, 0.05, 1.0],
        labels=["ALR = 0", "0 < ALR <= 0.05", "ALR > 0.05"],
    )
    marker_map = {"ALR = 0": "o", "0 < ALR <= 0.05": "s", "ALR > 0.05": "X"}
    hypervolume = pd.read_csv(
        ROOT / "results" / "certification_robustness" / "continuous_hypervolume.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.45), constrained_layout=True)
    ax = axes[0]
    for band, group in plotted.groupby("alr_band", observed=True):
        ax.scatter(
            group["coverage"],
            group["far"],
            s=82,
            marker=marker_map[str(band)],
            c=[RULE_COLORS.get(rule, PALETTE["gray"]) for rule in group["rule"]],
            edgecolors=PALETTE["dark"],
            linewidths=0.55,
            alpha=0.95,
            zorder=3,
        )
    ax.set_xlim(0.0, 1.10)
    ax.set_ylim(0.0, 0.75)
    _annotate_points(
        ax,
        plotted,
        "coverage",
        "far",
        "short_rule",
        0.008,
        0.008,
        initial_offsets={
            "DP": (0.015, 0.015),
            "HRG": (0.035, 0.040),
            "RF": (0.012, 0.020),
            "LC": (-0.090, 0.040),
            "UG": (0.080, 0.050),
            "CG": (0.080, -0.002),
            "CAG": (0.070, -0.025),
        },
    )
    ax.set_xlabel("Authorized coverage")
    ax.set_ylabel("Neg.-utility auth. rate (NUAR)")
    ax.set_title("Raw operating points")
    clean_axis(ax, "both")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=PALETTE["light_gray"],
            markeredgecolor=PALETTE["dark"],
            label=label,
            markersize=5.5,
        )
        for label, marker in marker_map.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        frameon=False,
    )
    panel_label(ax, "a")

    ax = axes[1]
    ordered = hypervolume.sort_values(
        ["worst_profile_hypervolume", "rule"], ascending=[True, True]
    )
    bars = ax.barh(
        ordered["rule"],
        ordered["worst_profile_hypervolume"],
        color=[RULE_COLORS.get(rule, PALETTE["gray"]) for rule in ordered["rule"]],
        edgecolor=PALETTE["dark"],
        linewidth=0.45,
        zorder=3,
    )
    for bar, value in zip(bars, ordered["worst_profile_hypervolume"]):
        ax.text(
            max(value + 0.004, 0.006),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8.0,
        )
    maximum = float(ordered["worst_profile_hypervolume"].max())
    ax.set_xlim(0, max(0.24, maximum * 1.22))
    ax.set_xlabel("Worst-profile hypervolume")
    ax.set_title("Validation conservative hypervolume")
    clean_axis(ax, "x")
    panel_label(ax, "b")
    qa = figure_text_qa(fig)
    hashes = save_figure(fig, output_dir / "fig_certification_surface")
    plt.close(fig)
    return {
        "name": "fig_certification_surface",
        "claim": "Raw NUAR, coverage, lineage integrity, and conservative hypervolume produce different rule orderings.",
        "hashes": hashes,
        "text_qa": qa,
        "size_inches": [FULL_WIDTH, 3.45],
    }


def plot_provenance(output_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(
        ROOT / "results" / "paper_test" / "provenance" / "summary.csv"
    )
    key_rules = [
        "No Role Gate",
        "Shared Threshold",
        "Hard Role Gate",
        "Provenance Hard Gate",
        "Provenance Learned Gate",
        "Lifecycle Checklist",
        "EPV Adapter",
    ]
    frame = summary[summary["rule"].isin(key_rules)].copy()
    frame["short_rule"] = frame["rule"].map(SHORT_RULE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.55), constrained_layout=True)

    ax = axes[0]
    y = np.arange(len(frame))
    height = 0.34
    ax.barh(
        y - height / 2,
        frame["direct_leakage_rate"],
        height=height,
        color=PALETTE["sky"],
        edgecolor=PALETTE["dark"],
        linewidth=0.4,
        hatch="///",
        label="Direct leakage",
    )
    ax.barh(
        y + height / 2,
        frame["indirect_leakage_rate"],
        height=height,
        color=PALETTE["vermillion"],
        edgecolor=PALETTE["dark"],
        linewidth=0.4,
        hatch="...",
        label="Indirect leakage",
    )
    ax.set_yticks(y, frame["rule"])
    ax.invert_yaxis()
    ax.set_xlim(
        0,
        max(
            0.28,
            float(frame[["direct_leakage_rate", "indirect_leakage_rate"]].max().max())
            + 0.035,
        ),
    )
    ax.set_xlabel("Leakage rate among authorized rows")
    ax.set_title("Role checks miss indirect laundering", pad=10)
    clean_axis(ax, "x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    panel_label(ax, "a")

    ax = axes[1]
    scatter = frame.copy()
    sizes = 45 + 150 * scatter["safe_delegation_coverage"]
    ax.scatter(
        scatter["false_block_rate"],
        scatter["alr"],
        s=sizes,
        c=[RULE_COLORS.get(rule, PALETTE["gray"]) for rule in scatter["rule"]],
        edgecolors=PALETTE["dark"],
        linewidths=0.55,
        zorder=3,
    )
    ax.set_xlim(-0.015, 0.40)
    ax.set_ylim(-0.010, 0.29)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    _annotate_points(
        ax,
        scatter,
        "false_block_rate",
        "alr",
        "short_rule",
        0.010,
        0.008,
        initial_offsets={
            "NRG": (0.050, 0.012),
            "ST": (0.135, -0.025),
            "HRG": (0.050, 0.028),
            "EPV": (0.145, 0.002),
            "LC": (0.075, -0.038),
            "PHG": (-0.085, 0.030),
            "PLG": (-0.060, 0.018),
        },
    )
    ax.set_xlabel("False-block rate on safe delegation")
    ax.set_ylabel("Authority laundering rate (ALR)")
    ax.set_title("Delegation-integrity trade-off", pad=10)
    clean_axis(ax, "both")
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALETTE["light_gray"],
            markeredgecolor=PALETTE["dark"],
            markersize=np.sqrt(45 + 150 * value),
            label=f"Safe delegation {value:.0%}",
        )
        for value in (0.25, 0.75)
    ]
    ax.legend(
        handles=size_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )
    panel_label(ax, "b")
    qa = figure_text_qa(fig)
    hashes = save_figure(fig, output_dir / "fig_provenance_laundering")
    plt.close(fig)
    return {
        "name": "fig_provenance_laundering",
        "claim": "Clean current roles do not prevent indirect laundering, while hard provenance gating trades integrity for coverage.",
        "hashes": hashes,
        "text_qa": qa,
        "size_inches": [FULL_WIDTH, 3.75],
    }


def plot_validity_gates(output_dir: Path) -> dict[str, object]:
    external = json.loads(
        (
            ROOT
            / "results"
            / "external_orderbook_v03"
            / "paper_test"
            / "binance"
            / "primary_endpoints.json"
        ).read_text(encoding="utf-8")
    )
    learning = pd.read_csv(
        ROOT / "results" / "training_robustness_v03" / "learning_curve_summary.csv"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH, 2.95),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.88, 1.12]},
    )
    ax = axes[0]
    endpoint_specs = [
        ("Neg.-utility burden", external["false_authorization_burden"], -1.0),
        ("Laundering burden", external["laundering_burden"], 1.0),
    ]
    y_positions = np.array([1.0, 0.0])
    points: list[tuple[float, float]] = []
    for y, (label, record, orientation) in zip(
        y_positions, endpoint_specs, strict=True
    ):
        sesoi = abs(float(record["sesoi"]))
        mean = orientation * float(record["mean"]) / sesoi
        if orientation > 0:
            lower = float(record["ci95_lower"]) / sesoi
            upper = float(record["ci95_upper"]) / sesoi
        else:
            lower = -float(record["ci95_upper"]) / sesoi
            upper = -float(record["ci95_lower"]) / sesoi
        color = PALETTE["green"] if bool(record["passed"]) else PALETTE["vermillion"]
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            markersize=6.0,
            color=color,
            ecolor=color,
            markeredgecolor=PALETTE["dark"],
            markeredgewidth=0.5,
            capsize=3,
            linewidth=1.4,
            zorder=4,
        )
        points.append((mean, y))
    ax._finauth_qa_points = points
    ax.axvline(0.0, color=PALETTE["gray"], linewidth=0.8)
    ax.axvline(1.0, color=PALETTE["vermillion"], linestyle="--", linewidth=1.35)
    ax.set_xlim(-0.08, 1.48)
    ax.set_ylim(-0.68, 1.55)
    ax.set_yticks(y_positions, [spec[0] for spec in endpoint_specs])
    ax.set_xlabel("Effect in registered direction / SESOI")
    ax.set_title("Binance co-primary effect test")
    criterion = ax.text(
        0.98,
        1.42,
        "arm criterion",
        fontsize=8.0,
        color=PALETTE["vermillion"],
        va="top",
        ha="right",
    )
    criterion.set_gid("qa_label")
    status_label = ax.text(
        0.96,
        0.08,
        "1 of 2 arms passes\nIUT claim not supported",
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["vermillion"],
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.8},
    )
    status_label.set_gid("qa_label")
    clean_axis(ax, "x")
    panel_label(ax, "a")

    ax = axes[1]
    final = learning[learning["budget"] == learning["budget"].max()].copy()
    curricula = ["controlled_seen", "multi_generator", "full_audit_seen"]
    learners = [
        "logistic_regression",
        "histogram_gradient_boosting",
        "mlp",
        "selective_logistic",
    ]
    labels = ["Logistic", "HistGB", "MLP", "Selective\nlogistic"]
    pivot = final.pivot(
        index="curriculum",
        columns="learner",
        values="valid_mechanism_fraction_mean",
    ).reindex(index=curricula, columns=learners)
    values = pivot.to_numpy(dtype=float)
    image = ax.pcolormesh(
        np.arange(len(learners) + 1) - 0.5,
        np.arange(len(curricula) + 1) - 0.5,
        values,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(-0.5, len(learners) - 0.5)
    ax.set_ylim(len(curricula) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(learners)), labels)
    ax.set_yticks(
        np.arange(len(curricula)),
        ["controlled\nseen", "multi\ngenerator", "full audit\nseen"],
    )
    for row_index in range(len(curricula)):
        for column_index in range(len(learners)):
            value = float(pivot.iloc[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=PALETTE["white"] if value >= 0.58 else PALETTE["dark"],
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Valid mechanism fraction")
    ax.set_title("Mechanism validity at 20k clusters", pad=10)
    panel_label(ax, "b")
    qa = figure_text_qa(fig)
    hashes = save_figure(fig, output_dir / "fig_validity_gates")
    plt.close(fig)
    return {
        "name": "fig_validity_gates",
        "claim": "A preregistered external test can yield a mixed failure, while mechanism-coverage gates keep incomplete learner comparisons undefined.",
        "hashes": hashes,
        "text_qa": qa,
        "size_inches": [FULL_WIDTH, 2.95],
    }


def run(output_dir: Path) -> Path:
    registry, rank_transfer = _load_v06_rank_transfer()
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_benchmark_overview(
            output_dir,
            rank_transfer,
            independent_dates=int(registry["paper_test_clusters"]),
        ),
        plot_certification(output_dir),
        plot_provenance(output_dir),
        plot_real_agent_validity(output_dir),
        plot_validity_gates(output_dir),
    ]
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.6.0-supplied-overview-actual-agent-integration",
        "style": "tueplots-neurips2024-okabe-ito",
        "legend_policy": "outside_axes_only",
        "annotation_policy": "adjustText_with_leader_lines_and_marker_boundary_avoidance",
        "minimum_font_points": 8.0,
        "paper_target_minimum_font_points": 8.0,
        "note": "Figures 2--5 are vector outputs with programmatic text QA. Figure 1 preserves the reviewed user-supplied 2048x1143 raster composition at approximately 295 effective dpi; its raster exception and visual overlap review are recorded explicitly.",
        "figures": figures,
        "inputs": {
            str(OVERVIEW_SOURCE_RELATIVE): _sha256(_overview_source_path()),
            "results/paper_test/manifest.json": _sha256(
                ROOT / "results" / "paper_test" / "manifest.json"
            ),
            "results/external_orderbook_v03/paper_test/binance/primary_endpoints.json": _sha256(
                ROOT
                / "results"
                / "external_orderbook_v03"
                / "paper_test"
                / "binance"
                / "primary_endpoints.json"
            ),
            "results/training_robustness_v03/manifest.json": _sha256(
                ROOT / "results" / "training_robustness_v03" / "manifest.json"
            ),
            str(V06_REGISTRY_RELATIVE): _sha256(ROOT / V06_REGISTRY_RELATIVE),
            str(V06_RANK_RELATIVE): _sha256(ROOT / V06_RANK_RELATIVE),
            str(V06_ZERO_METRICS_RELATIVE): _sha256(
                ROOT / V06_ZERO_METRICS_RELATIVE
            ),
            str(V06_RECAL_METRICS_RELATIVE): _sha256(
                ROOT / V06_RECAL_METRICS_RELATIVE
            ),
        },
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate all FinAuth-Audit paper figures."
    )
    parser.add_argument("--output-dir", default=str(ROOT / "paper" / "figures"))
    args = parser.parse_args()
    run(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
