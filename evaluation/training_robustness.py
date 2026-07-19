from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finauth_audit.evaluation.seeds import derive_seed
from finauth_audit.evaluation.training_utility import _feature_frame, _read_split
from finauth_audit.figures.style import (
    FULL_WIDTH,
    PALETTE,
    apply_style,
    clean_axis,
    figure_text_qa,
    panel_label,
    save_figure,
)
from finauth_audit.generators.training_corpora import (
    _authorization_label,
    _normalize,
    _one_row_per_cluster,
)


ROOT = Path(__file__).resolve().parents[1]

LEARNER_LABELS = {
    "logistic_regression": "Logistic",
    "histogram_gradient_boosting": "HistGB",
    "mlp": "MLP",
    "selective_logistic": "Selective logistic",
}

LEARNER_COLORS = {
    "logistic_regression": PALETTE["blue"],
    "histogram_gradient_boosting": PALETTE["orange"],
    "mlp": PALETTE["purple"],
    "selective_logistic": PALETTE["green"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_columns(features: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                "row_id",
                "event_cluster_id",
                "split",
                "opportunity_slice",
                "attack_type",
                "full_utility",
                "original_source_eligible",
                "authority_laundering",
            ]
            + features
        )
    )


def _read_source(
    path: Path,
    split: str,
    features: list[str],
    layer: str,
) -> pd.DataFrame:
    frame = _read_split(path, split, _required_columns(features))
    normalized = _normalize(frame, features, layer)
    normalized["evaluation_cluster_id"] = normalized["training_cluster_id"]
    normalized["authorization_harm"] = 1 - _authorization_label(normalized)
    normalized["authority_laundering"] = (
        normalized["authority_laundering"].fillna(False).astype(bool)
    )
    return normalized


def _exclude_registered_holdouts(
    frame: pd.DataFrame,
    layer: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    if layer == "controlled":
        return frame[
            ~frame["opportunity_slice"].isin(config["controlled_test_holdouts"])
        ].copy()
    if layer == "provenance":
        return frame[
            ~frame["attack_type"].isin(config["provenance_test_holdouts"])
        ].copy()
    return frame.copy()


def _ordered_unique_clusters(
    frame: pd.DataFrame,
    seed: int,
    namespace: str,
) -> pd.DataFrame:
    one_per_cluster = _one_row_per_cluster(
        frame, derive_seed(seed, f"training-robustness/{namespace}/row")
    )
    rng = np.random.default_rng(
        derive_seed(seed, f"training-robustness/{namespace}/order")
    )
    order = rng.permutation(len(one_per_cluster))
    return one_per_cluster.iloc[order].reset_index(drop=True)


def _curriculum_quotas(
    budget: int,
    weights: dict[str, float],
) -> dict[str, int]:
    if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise ValueError(f"curriculum weights must sum to one: {weights}")
    names = list(weights)
    quotas: dict[str, int] = {}
    assigned = 0
    for name in names[:-1]:
        quota = int(round(budget * float(weights[name])))
        quotas[name] = quota
        assigned += quota
    quotas[names[-1]] = budget - assigned
    if any(value <= 0 for value in quotas.values()):
        raise ValueError(f"all curriculum sources need positive quotas: {quotas}")
    return quotas


def _training_sample(
    ordered_pools: dict[tuple[str, int], pd.DataFrame],
    curriculum: str,
    budget: int,
    seed: int,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quotas = _curriculum_quotas(budget, weights)
    pieces: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    for source, quota in quotas.items():
        pool = ordered_pools[(source, seed)]
        if len(pool) < quota:
            raise ValueError(
                f"{curriculum} requires {quota} {source} clusters; pool has {len(pool)}"
            )
        selected = pool.iloc[:quota].copy()
        pieces.append(selected)
        selection_rows.append(
            pd.DataFrame(
                {
                    "curriculum": curriculum,
                    "budget": budget,
                    "seed": seed,
                    "source_layer": source,
                    "training_cluster_id": selected["training_cluster_id"].astype(str),
                    "selected_row_id": selected["row_id"].astype(str),
                }
            )
        )
    sample = pd.concat(pieces, ignore_index=True)
    sample["training_label"] = _authorization_label(sample)
    if sample["training_label"].nunique() < 2:
        raise ValueError(
            f"{curriculum} budget={budget} seed={seed} contains one label class"
        )
    return sample, pd.concat(selection_rows, ignore_index=True)


def _preprocessor(config: dict[str, Any], dense: bool) -> ColumnTransformer:
    numeric = list(config["features"]["numeric"])
    categorical = list(config["features"]["categorical"])
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def build_learner(
    learner: str,
    config: dict[str, Any],
    seed: int,
) -> Pipeline:
    model_config = config["model"]
    if learner in {"logistic_regression", "selective_logistic"}:
        classifier = LogisticRegression(
            C=float(model_config["logistic_c"]),
            class_weight="balanced",
            max_iter=int(model_config["max_iter"]),
            random_state=int(seed),
            solver="liblinear",
        )
        return Pipeline(
            [
                ("preprocessor", _preprocessor(config, dense=False)),
                ("classifier", classifier),
            ]
        )
    if learner == "histogram_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=int(model_config["histogram_max_iter"]),
            max_leaf_nodes=int(model_config["histogram_max_leaf_nodes"]),
            learning_rate=0.08,
            l2_regularization=0.05,
            early_stopping=True,
            validation_fraction=0.10,
            random_state=int(seed),
        )
        return Pipeline(
            [
                ("preprocessor", _preprocessor(config, dense=True)),
                ("classifier", classifier),
            ]
        )
    if learner == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(
                int(value) for value in model_config["mlp_hidden_layer_sizes"]
            ),
            alpha=float(model_config["mlp_alpha"]),
            batch_size=128,
            learning_rate_init=0.001,
            early_stopping=bool(model_config["mlp_early_stopping"]),
            validation_fraction=0.10,
            max_iter=int(model_config["max_iter"]),
            random_state=int(seed),
        )
        return Pipeline(
            [
                ("preprocessor", _preprocessor(config, dense=True)),
                ("classifier", classifier),
            ]
        )
    raise KeyError(learner)


def _cluster_metrics_with_bounds(
    frame: pd.DataFrame,
    authorized: np.ndarray,
    z_value: float,
) -> dict[str, float | int]:
    evaluated = frame[
        [
            "evaluation_cluster_id",
            "authorization_harm",
            "authority_laundering",
            "full_utility",
        ]
    ].copy()
    evaluated["authorized"] = authorized.astype(bool)
    evaluated["authorized_harm"] = (
        evaluated["authorized"].astype(float)
        * evaluated["authorization_harm"].astype(float)
    )
    evaluated["authorized_laundering"] = (
        evaluated["authorized"].astype(float)
        * evaluated["authority_laundering"].astype(float)
    )
    evaluated["authorized_utility"] = (
        evaluated["authorized"].astype(float)
        * evaluated["full_utility"].astype(float)
    )
    clusters = evaluated.groupby("evaluation_cluster_id", sort=False).agg(
        row_count=("authorized", "size"),
        authorized_count=("authorized", "sum"),
        harm_count=("authorized_harm", "sum"),
        laundering_count=("authorized_laundering", "sum"),
        cau=("authorized_utility", "sum"),
    )
    clusters["coverage"] = clusters["authorized_count"] / clusters["row_count"]
    clusters["far"] = np.divide(
        clusters["harm_count"],
        clusters["authorized_count"],
        out=np.full(len(clusters), np.nan, dtype=float),
        where=clusters["authorized_count"].to_numpy() > 0,
    )
    clusters["alr"] = np.divide(
        clusters["laundering_count"],
        clusters["authorized_count"],
        out=np.full(len(clusters), np.nan, dtype=float),
        where=clusters["authorized_count"].to_numpy() > 0,
    )

    def mean_bound(series: pd.Series, direction: str) -> tuple[float, float]:
        values = series.dropna().to_numpy(dtype=float)
        if not len(values):
            return np.nan, np.nan
        mean = float(values.mean())
        if len(values) == 1:
            return mean, mean
        standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
        bound = mean + (z_value * standard_error if direction == "upper" else -z_value * standard_error)
        return mean, float(np.clip(bound, 0.0, 1.0))

    coverage, coverage_lcb = mean_bound(clusters["coverage"], "lower")
    far, far_ucb = mean_bound(clusters["far"], "upper")
    alr, alr_ucb = mean_bound(clusters["alr"], "upper")
    return {
        "rows": len(frame),
        "clusters": int(len(clusters)),
        "authorized_count": int(np.asarray(authorized, dtype=bool).sum()),
        "coverage": coverage,
        "coverage_lcb95": coverage_lcb,
        "far": far,
        "far_ucb95": far_ucb,
        "alr": alr,
        "alr_ucb95": alr_ucb,
        "cau": float(clusters["cau"].mean()),
    }


def endpoint_valid(metrics: dict[str, float | int], coverage_floor: float) -> bool:
    return bool(
        np.isfinite(float(metrics["far"]))
        and np.isfinite(float(metrics["coverage_lcb95"]))
        and float(metrics["coverage_lcb95"]) >= coverage_floor
    )


def _fixed_threshold(scores: np.ndarray, coverage: float) -> float:
    return float(np.quantile(scores, 1.0 - coverage, method="higher"))


def _select_threshold(
    learner: str,
    scores: np.ndarray,
    calibration: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[float, dict[str, object]]:
    z_value = float(config["one_sided_confidence_z"])
    if learner != "selective_logistic":
        target = float(config["validation_coverage_target"])
        threshold = _fixed_threshold(scores, target)
        metrics = _cluster_metrics_with_bounds(
            calibration, scores >= threshold, z_value
        )
        return threshold, {
            "threshold_policy": "fixed_calibration_coverage",
            "target_coverage": target,
            "calibration_floor_satisfied": endpoint_valid(
                metrics, float(config["min_primary_mechanism_coverage"])
            ),
            **metrics,
        }

    candidates: list[dict[str, object]] = []
    for target in config["selective_coverage_candidates"]:
        threshold = _fixed_threshold(scores, float(target))
        layer_metrics: list[dict[str, float | int]] = []
        for _, frame in calibration.groupby("source_layer", sort=True):
            indices = frame.index.to_numpy()
            layer_scores = scores[indices]
            layer_metrics.append(
                _cluster_metrics_with_bounds(
                    frame, layer_scores >= threshold, z_value
                )
            )
        floor = min(float(row["coverage_lcb95"]) for row in layer_metrics)
        finite_far = [float(row["far"]) for row in layer_metrics if np.isfinite(row["far"])]
        candidates.append(
            {
                "threshold": threshold,
                "target_coverage": float(target),
                "coverage_lcb_floor": floor,
                "max_far": max(finite_far) if finite_far else np.nan,
                "mean_far": float(np.mean(finite_far)) if finite_far else np.nan,
                "mean_cau": float(np.mean([float(row["cau"]) for row in layer_metrics])),
                "valid": bool(
                    finite_far
                    and floor >= float(config["min_primary_mechanism_coverage"])
                ),
            }
        )
    valid = [row for row in candidates if bool(row["valid"])]
    selected = min(
        valid if valid else candidates,
        key=lambda row: (
            0 if bool(row["valid"]) else 1,
            float(row["max_far"]) if np.isfinite(row["max_far"]) else float("inf"),
            float(row["mean_far"]) if np.isfinite(row["mean_far"]) else float("inf"),
            -float(row["mean_cau"]),
            -float(row["target_coverage"]),
        ),
    )
    threshold = float(selected["threshold"])
    metrics = _cluster_metrics_with_bounds(calibration, scores >= threshold, z_value)
    return threshold, {
        "threshold_policy": "coverage_constrained_validation_selection",
        "target_coverage": float(selected["target_coverage"]),
        "calibration_floor_satisfied": bool(selected["valid"]),
        "candidate_count": len(candidates),
        "selection_max_far": selected["max_far"],
        "selection_coverage_lcb_floor": selected["coverage_lcb_floor"],
        **metrics,
    }


def _evaluation_frames(
    validation: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    controlled = validation["controlled"]
    provenance = validation["provenance"]
    calibration = pd.concat(
        [
            controlled[
                ~controlled["opportunity_slice"].isin(
                    config["controlled_test_holdouts"]
                )
            ],
            provenance[
                ~provenance["attack_type"].isin(
                    config["provenance_test_holdouts"]
                )
            ],
        ],
        ignore_index=True,
    )
    calibration.index = np.arange(len(calibration))
    frames = {
        "controlled_severe_stress": controlled[
            controlled["opportunity_slice"] == "severe_stress"
        ].copy(),
        "controlled_stress_only_period": controlled[
            controlled["opportunity_slice"] == "stress_only_period"
        ].copy(),
        "provenance_paraphrase": provenance[
            provenance["attack_type"] == "paraphrase"
        ].copy(),
        "provenance_multi_hop": provenance[
            provenance["attack_type"] == "multi_hop"
        ].copy(),
        "sequential_market_validation": validation["sequential_market"].copy(),
        "institutional_workflow_validation": validation[
            "institutional_workflow"
        ].copy(),
    }
    missing = set(config["primary_evaluations"]) - set(frames)
    if missing:
        raise ValueError(f"missing primary evaluation frames: {sorted(missing)}")
    if any(frame.empty for frame in frames.values()):
        empty = sorted(name for name, frame in frames.items() if frame.empty)
        raise ValueError(f"empty evaluation mechanisms: {empty}")
    return calibration, frames


def _write_learning_curve_figure(
    summary: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 3.35), constrained_layout=True
    )

    line_data = summary[summary["curriculum"] == "multi_generator"].copy()
    ax = axes[0]
    for learner in LEARNER_LABELS:
        frame = line_data[line_data["learner"] == learner].sort_values("budget")
        ax.plot(
            frame["budget"],
            frame["valid_mechanism_fraction_mean"],
            marker="o",
            color=LEARNER_COLORS[learner],
            label=LEARNER_LABELS[learner],
        )
    ax.set_xscale("log")
    ax.set_xticks([400, 1000, 5000, 20000], ["400", "1k", "5k", "20k"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Independent training clusters")
    ax.set_ylabel("Valid mechanism fraction")
    ax.set_title("Multi-generator learning curve")
    clean_axis(ax, "both")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
    )
    panel_label(ax, "a")

    final = summary[summary["budget"] == summary["budget"].max()].copy()
    pivot = final.pivot(
        index="curriculum",
        columns="learner",
        values="valid_mechanism_fraction_mean",
    ).reindex(
        index=["controlled_seen", "multi_generator", "full_audit_seen"],
        columns=list(LEARNER_LABELS),
    )
    ax = axes[1]
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(pivot.columns)),
        [LEARNER_LABELS[name].replace(" ", "\n") for name in pivot.columns],
    )
    ax.set_yticks(
        np.arange(len(pivot.index)),
        [name.replace("_", "\n") for name in pivot.index],
    )
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
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
    ax.set_title("Endpoint validity at 20k clusters")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Valid mechanism fraction")
    panel_label(ax, "b")

    qa = figure_text_qa(fig)
    hashes = save_figure(fig, output_dir / "learning_curve")
    plt.close(fig)
    return hashes, qa


def finalize_existing(
    config_path: Path,
    elapsed_seconds: float | None = None,
    resumed_from_checkpoints: bool = True,
) -> Path:
    """Finalize complete CSV checkpoints without refitting models."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["outputs"]["results_dir"]
    required = [
        "per_mechanism.csv",
        "endpoint_validity.csv",
        "threshold_selection.csv",
        "model_runtime.csv",
        "training_selection.csv",
        "learning_curve_summary.csv",
        "mechanism_summary.csv",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"cannot finalize incomplete training robustness outputs: {missing}"
        )
    summary = pd.read_csv(output_dir / "learning_curve_summary.csv")
    selection_frame = pd.read_csv(
        output_dir / "training_selection.csv",
        usecols=["training_cluster_id"],
    )
    runtime_frame = pd.read_csv(output_dir / "model_runtime.csv")
    learners = list(config["learners"])
    features = config["features"]["numeric"] + config["features"]["categorical"]
    coverage_floor = float(config["min_primary_mechanism_coverage"])

    figure_hashes, figure_qa = _write_learning_curve_figure(summary, output_dir)
    final_budget = int(max(config["budgets"]))
    final_rows = summary[summary["budget"] == final_budget]
    best = final_rows.sort_values(
        [
            "valid_mechanism_fraction_mean",
            "all_mechanisms_valid_seeds",
            "mechanism_coverage_lcb_floor_mean",
        ],
        ascending=[False, False, False],
    ).iloc[0]
    report_lines = [
        "# Prospective Training Robustness v0.3",
        "",
        "## Scope",
        "",
        "This is a prospectively versioned, validation-only secondary robustness study. "
        "It does not modify the registered v0.2 Logistic Regression primary, replace its "
        "structural N/A outcomes, evaluate any test split, or support deployment claims.",
        "",
        "## Design",
        "",
        f"Learners: {', '.join(LEARNER_LABELS[name] for name in learners)}. Training "
        f"budgets: {', '.join(str(value) for value in config['budgets'])} independent "
        f"clusters. A primary endpoint is defined only when all "
        f"{len(config['primary_evaluations'])} mechanisms have a one-sided 95% coverage "
        f"lower bound of at least {coverage_floor:.2f}. Thresholds use seen validation "
        "mechanisms only; evaluation mechanisms remain disjoint from threshold selection.",
        "",
        "## Result",
        "",
        f"At {final_budget:,} clusters, the highest valid-mechanism fraction is "
        f"{float(best['valid_mechanism_fraction_mean']):.3f} for "
        f"{best['curriculum']} with {LEARNER_LABELS[str(best['learner'])]}. "
        "The result diagnoses whether endpoint definition recovers with learner capacity "
        "and data budget; it is not a benchmark winner claim.",
        "",
        "## Learning-curve summary",
        "",
        summary.to_csv(index=False),
        "",
        "## Interpretation boundary",
        "",
        "The sequential and institutional generators are mechanistically distinct but "
        "remain repository-authored controlled generators. They improve generator "
        "sensitivity analysis but do not replace practitioner labels or independent "
        "external validation.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    docs_path = ROOT / config["outputs"]["docs_report"]
    docs_path.write_text("\n".join(report_lines), encoding="utf-8")

    output_names = required + [
        "learning_curve.pdf",
        "learning_curve.svg",
        "learning_curve.png",
        "report.md",
    ]
    manifest = {
        "project": config["project"],
        "version": config["version"],
        "mode": config["mode"],
        "evaluation_split": "validation",
        "confirmatory": False,
        "test_outcomes_evaluated": False,
        "community_hidden_outcomes_evaluated": False,
        "registered_primary_preserved": config["registered_primary_preserved"],
        "registered_primary_preserved_sha256": sha256(
            ROOT / config["registered_primary_preserved"]
        ),
        "budgets": [int(value) for value in config["budgets"]],
        "seeds": [int(value) for value in config["seeds"]],
        "learners": learners,
        "curricula": config["curricula"],
        "primary_evaluations": config["primary_evaluations"],
        "min_primary_mechanism_coverage": coverage_floor,
        "legal_features": features,
        "inputs": {
            relative: sha256(ROOT / relative)
            for relative in config["inputs"].values()
        },
        "config_sha256": sha256(config_path),
        "outputs": {name: sha256(output_dir / name) for name in output_names},
        "docs_report": {
            str(docs_path.relative_to(ROOT)): sha256(docs_path),
        },
        "figure_hashes": figure_hashes,
        "figure_text_qa": figure_qa,
        "selection_rows": len(selection_frame),
        "fit_count": len(runtime_frame),
        "elapsed_seconds": elapsed_seconds,
        "finalized_from_complete_csv_checkpoints": resumed_from_checkpoints,
        "claim_boundary": (
            "Prospective validation-only learner and sample-budget sensitivity. "
            "The registered v0.2 primary remains unchanged; test outcomes, community "
            "hidden outcomes, practitioner labels, and deployment evidence are absent."
        ),
    }
    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(manifest_path)
    return output_dir


def run(config_path: Path) -> Path:
    started = time.perf_counter()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("evaluation_split") != "validation":
        raise ValueError("training robustness must remain validation-only")
    if config.get("test_outcomes_evaluated") is not False:
        raise ValueError("test_outcomes_evaluated must remain false")
    feature_manifest = json.loads(
        (ROOT / "manifests" / "feature_access.json").read_text(encoding="utf-8")
    )
    features = config["features"]["numeric"] + config["features"]["categorical"]
    legal = set(feature_manifest["tasks"]["training_utility"]["legal"])
    forbidden = set(feature_manifest["global_forbidden"])
    if set(features) != legal or set(features) & forbidden:
        raise ValueError("v0.3 features must equal the frozen legal training manifest")

    train_sources: dict[str, pd.DataFrame] = {}
    validation_sources: dict[str, pd.DataFrame] = {}
    for layer, relative in config["inputs"].items():
        path = ROOT / relative
        train = _read_source(path, "train", features, layer)
        train_sources[layer] = _exclude_registered_holdouts(train, layer, config)
        validation_sources[layer] = _read_source(path, "validation", features, layer)
    calibration, evaluation_frames = _evaluation_frames(validation_sources, config)

    ordered_pools: dict[tuple[str, int], pd.DataFrame] = {}
    for seed in config["seeds"]:
        for source, frame in train_sources.items():
            ordered_pools[(source, int(seed))] = _ordered_unique_clusters(
                frame, int(seed), source
            )

    results: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    selection_rows: list[pd.DataFrame] = []
    learners = list(config["learners"])
    z_value = float(config["one_sided_confidence_z"])
    coverage_floor = float(config["min_primary_mechanism_coverage"])

    for curriculum, weights in config["curricula"].items():
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                seed = int(seed)
                sample, selection = _training_sample(
                    ordered_pools,
                    curriculum,
                    int(budget),
                    seed,
                    weights,
                )
                selection_rows.append(selection)
                train_x = _feature_frame(sample, config)
                train_y = sample["training_label"].astype(int)
                calibration_x = _feature_frame(calibration, config)
                fitted_logistic: Pipeline | None = None
                fitted_logistic_seconds = 0.0

                for learner in learners:
                    fit_started = time.perf_counter()
                    reused_model = False
                    if learner == "selective_logistic" and fitted_logistic is not None:
                        model = fitted_logistic
                        fit_seconds = 0.0
                        reused_model = True
                    else:
                        model = build_learner(learner, config, seed)
                        model.fit(train_x, train_y)
                        fit_seconds = time.perf_counter() - fit_started
                        if learner == "logistic_regression":
                            fitted_logistic = model
                            fitted_logistic_seconds = fit_seconds
                    calibration_scores = model.predict_proba(calibration_x)[:, 1]
                    threshold, threshold_record = _select_threshold(
                        learner, calibration_scores, calibration, config
                    )
                    threshold_rows.append(
                        {
                            "curriculum": curriculum,
                            "budget": int(budget),
                            "seed": seed,
                            "learner": learner,
                            "threshold": threshold,
                            **threshold_record,
                        }
                    )
                    mechanism_records: list[dict[str, object]] = []
                    predict_seconds = 0.0
                    for evaluation_name in config["primary_evaluations"]:
                        frame = evaluation_frames[evaluation_name]
                        predict_started = time.perf_counter()
                        scores = model.predict_proba(_feature_frame(frame, config))[:, 1]
                        predict_seconds += time.perf_counter() - predict_started
                        metrics = _cluster_metrics_with_bounds(
                            frame, scores >= threshold, z_value
                        )
                        valid = endpoint_valid(metrics, coverage_floor)
                        record = {
                            "curriculum": curriculum,
                            "budget": int(budget),
                            "seed": seed,
                            "learner": learner,
                            "evaluation": evaluation_name,
                            "threshold": threshold,
                            "endpoint_valid": valid,
                            **metrics,
                        }
                        results.append(record)
                        mechanism_records.append(record)
                    valid_count = sum(
                        bool(record["endpoint_valid"])
                        for record in mechanism_records
                    )
                    complete = valid_count == len(config["primary_evaluations"])
                    endpoint_rows.append(
                        {
                            "curriculum": curriculum,
                            "budget": int(budget),
                            "seed": seed,
                            "learner": learner,
                            "valid_mechanisms": valid_count,
                            "mechanism_count": len(config["primary_evaluations"]),
                            "valid_mechanism_fraction": valid_count
                            / len(config["primary_evaluations"]),
                            "all_mechanisms_valid": complete,
                            "primary_far": (
                                float(
                                    np.mean(
                                        [
                                            float(record["far"])
                                            for record in mechanism_records
                                        ]
                                    )
                                )
                                if complete
                                else np.nan
                            ),
                            "mechanism_coverage_lcb_floor": min(
                                float(record["coverage_lcb95"])
                                for record in mechanism_records
                            ),
                            "failed_mechanisms": ",".join(
                                sorted(
                                    str(record["evaluation"])
                                    for record in mechanism_records
                                    if not bool(record["endpoint_valid"])
                                )
                            ),
                        }
                    )
                    runtime_rows.append(
                        {
                            "curriculum": curriculum,
                            "budget": int(budget),
                            "seed": seed,
                            "learner": learner,
                            "fit_seconds": fit_seconds,
                            "predict_seconds": predict_seconds,
                            "model_reused": reused_model,
                            "shared_logistic_fit_seconds": (
                                fitted_logistic_seconds if reused_model else 0.0
                            ),
                        }
                    )

    output_dir = ROOT / config["outputs"]["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    endpoint_frame = pd.DataFrame(endpoint_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    runtime_frame = pd.DataFrame(runtime_rows)
    selection_frame = pd.concat(selection_rows, ignore_index=True)
    result_frame.to_csv(output_dir / "per_mechanism.csv", index=False)
    endpoint_frame.to_csv(output_dir / "endpoint_validity.csv", index=False)
    threshold_frame.to_csv(output_dir / "threshold_selection.csv", index=False)
    runtime_frame.to_csv(output_dir / "model_runtime.csv", index=False)
    selection_frame.to_csv(output_dir / "training_selection.csv", index=False)

    summary = (
        endpoint_frame.groupby(["curriculum", "budget", "learner"], sort=True)
        .agg(
            valid_mechanism_fraction_mean=("valid_mechanism_fraction", "mean"),
            valid_mechanism_fraction_std=("valid_mechanism_fraction", "std"),
            all_mechanisms_valid_seeds=("all_mechanisms_valid", "sum"),
            primary_far_mean=("primary_far", "mean"),
            primary_far_std=("primary_far", "std"),
            mechanism_coverage_lcb_floor_mean=(
                "mechanism_coverage_lcb_floor",
                "mean",
            ),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "learning_curve_summary.csv", index=False)
    mechanism_summary = (
        result_frame.groupby(
            ["curriculum", "budget", "learner", "evaluation"], sort=True
        )
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_lcb95_mean=("coverage_lcb95", "mean"),
            far_mean=("far", "mean"),
            far_ucb95_mean=("far_ucb95", "mean"),
            alr_mean=("alr", "mean"),
            cau_mean=("cau", "mean"),
            valid_seed_count=("endpoint_valid", "sum"),
        )
        .reset_index()
    )
    mechanism_summary.to_csv(output_dir / "mechanism_summary.csv", index=False)

    return finalize_existing(
        config_path,
        time.perf_counter() - started,
        resumed_from_checkpoints=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run prospective multi-learner training robustness."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "training_robustness_v03.yaml"),
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Rebuild figures, reports, and manifest from complete CSV checkpoints.",
    )
    parser.add_argument(
        "--observed-elapsed-seconds",
        type=float,
        default=None,
        help="Record the observed fit-run wall clock when finalizing checkpoints.",
    )
    args = parser.parse_args()
    if args.finalize_existing:
        finalize_existing(
            Path(args.config),
            elapsed_seconds=args.observed_elapsed_seconds,
            resumed_from_checkpoints=True,
        )
    else:
        run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
