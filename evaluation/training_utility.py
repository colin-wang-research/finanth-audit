from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_rel
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finauth_audit.evaluation.holm import holm_adjust
from finauth_audit.evaluation.seeds import derive_seed
from finauth_audit.generators.training_corpora import _authorization_label, _normalize


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_split(path: Path, split: str, required: list[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda name: name in set(required), chunksize=50_000):
        selected = chunk[chunk["split"] == split].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"split={split} is empty: {path}")
    return pd.concat(chunks, ignore_index=True)


def _prepare_eval(
    frame: pd.DataFrame, features: list[str], layer: str
) -> pd.DataFrame:
    normalized = _normalize(frame, features, layer)
    normalized["evaluation_cluster_id"] = (
        layer + ":" + normalized["event_cluster_id"].astype(str)
    )
    normalized["authorization_harm"] = 1 - _authorization_label(normalized)
    normalized["authority_laundering"] = (
        normalized["authority_laundering"].fillna(False).astype(bool)
    )
    return normalized


def _model(config: dict[str, Any], seed: int, c_value: float) -> Pipeline:
    numeric = list(config["features"]["numeric"])
    categorical = list(config["features"]["categorical"])
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=float(c_value),
        class_weight=str(config["class_weight"]),
        max_iter=int(config["max_iter"]),
        random_state=int(seed),
        solver="liblinear",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _feature_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    features = config["features"]["numeric"] + config["features"]["categorical"]
    result = frame[features].copy()
    for column in config["features"]["numeric"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    for column in config["features"]["categorical"]:
        result[column] = result[column].astype(str).fillna("unknown")
    return result


def _masked_frame(
    frame: pd.DataFrame, config: dict[str, Any], group: str
) -> pd.DataFrame:
    masked = frame.copy()
    groups = {
        "role": [
            "source_role",
            "claimed_role",
            "current_role_verified",
            "verified_current_role",
            "hop_depth",
            "transformation_type",
            "traceability",
            "lineage_attested",
        ],
        "cost": [
            "liquidity_cost_bps",
            "turnover_cost_bps",
            "fee_bps",
            "fee_rate",
        ],
        "belief": ["confidence", "uncertainty", "expected_edge_bps"],
    }
    categorical = set(config["features"]["categorical"])
    for column in groups[group]:
        masked[column] = "masked" if column in categorical else 0.0
    return masked


def _threshold(scores: np.ndarray, target_coverage: float) -> float:
    return float(np.quantile(scores, 1.0 - target_coverage, method="higher"))


def _cluster_metrics(frame: pd.DataFrame, authorized: np.ndarray) -> dict[str, float | int]:
    evaluated = frame[["evaluation_cluster_id", "authorization_harm", "authority_laundering", "full_utility"]].copy()
    evaluated["authorized"] = authorized.astype(bool)
    cluster_rows: list[dict[str, float]] = []
    for _, group in evaluated.groupby("evaluation_cluster_id"):
        auth = group[group["authorized"]]
        cluster_rows.append(
            {
                "coverage": float(group["authorized"].mean()),
                "far": float(auth["authorization_harm"].mean()) if len(auth) else np.nan,
                "alr": float(auth["authority_laundering"].mean()) if len(auth) else np.nan,
                "cau": float(auth["full_utility"].sum()),
            }
        )
    clusters = pd.DataFrame(cluster_rows)
    active = clusters[clusters["far"].notna()]
    return {
        "rows": len(frame),
        "clusters": int(frame["evaluation_cluster_id"].nunique()),
        "authorized_count": int(authorized.sum()),
        "coverage": float(clusters["coverage"].mean()),
        "far": float(active["far"].mean()) if len(active) else np.nan,
        "alr": float(active["alr"].mean()) if len(active) else np.nan,
        "cau": float(clusters["cau"].mean()),
    }


def _bootstrap_bounds(
    frame: pd.DataFrame,
    authorized: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    evaluated = frame[["evaluation_cluster_id", "authorization_harm", "authority_laundering"]].copy()
    evaluated["authorized"] = authorized.astype(bool)
    cluster_rows: list[dict[str, float]] = []
    for _, group in evaluated.groupby("evaluation_cluster_id"):
        auth = group[group["authorized"]]
        cluster_rows.append(
            {
                "coverage": float(group["authorized"].mean()),
                "far": float(auth["authorization_harm"].mean()) if len(auth) else np.nan,
                "alr": float(auth["authority_laundering"].mean()) if len(auth) else np.nan,
            }
        )
    clusters = pd.DataFrame(cluster_rows)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    coverage = np.mean(clusters["coverage"].to_numpy()[indices], axis=1)
    def sampled_nanmean(values: np.ndarray) -> np.ndarray:
        sampled = values[indices]
        valid = np.isfinite(sampled)
        counts = valid.sum(axis=1)
        sums = np.where(valid, sampled, 0.0).sum(axis=1)
        return np.divide(
            sums,
            counts,
            out=np.full(len(counts), np.nan, dtype=float),
            where=counts > 0,
        )

    far = sampled_nanmean(clusters["far"].to_numpy())
    alr = sampled_nanmean(clusters["alr"].to_numpy())
    return {
        "coverage_lcb95": float(np.nanquantile(coverage, 0.05)),
        "far_ucb95": float(np.nanquantile(far, 0.95)) if np.isfinite(far).any() else np.nan,
        "alr_ucb95": float(np.nanquantile(alr, 0.95)) if np.isfinite(alr).any() else np.nan,
    }


def _certification_volume(bounds: dict[str, float], config: dict[str, Any]) -> float:
    if not np.isfinite(bounds["far_ucb95"]) or not np.isfinite(bounds["alr_ucb95"]):
        return 0.0
    passed = 0
    total = 0
    grid = config["certification_grid"]
    for far_threshold in grid["far_thresholds"]:
        for alr_threshold in grid["alr_thresholds"]:
            for minimum_coverage in grid["minimum_coverage"]:
                total += 1
                passed += int(
                    bounds["far_ucb95"] <= float(far_threshold)
                    and bounds["alr_ucb95"] <= float(alr_threshold)
                    and bounds["coverage_lcb95"] >= float(minimum_coverage)
                )
    return passed / total


def _evaluate_scores(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    config: dict[str, Any],
    seed: int,
    namespace: str,
) -> dict[str, float | int]:
    authorized = scores >= threshold
    metrics = _cluster_metrics(frame, authorized)
    bounds = _bootstrap_bounds(
        frame,
        authorized,
        int(config["bootstrap_replicates"]),
        derive_seed(seed, f"training/{namespace}"),
    )
    return {
        **metrics,
        **bounds,
        "certification_volume": _certification_volume(bounds, config),
    }


def _validation_frames(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    features = config["features"]["numeric"] + config["features"]["categorical"]
    required = list(
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
    controlled = _prepare_eval(
        _read_split(ROOT / config["inputs"]["controlled"], "validation", required),
        features,
        "controlled",
    )
    provenance = _prepare_eval(
        _read_split(ROOT / config["inputs"]["provenance"], "validation", required),
        features,
        "provenance",
    )
    public = _prepare_eval(
        _read_split(ROOT / config["inputs"]["public"], "validation", required),
        features,
        "public",
    )
    controlled_holdout = controlled["opportunity_slice"].isin(
        config["controlled_test_holdouts"]
    )
    provenance_holdout = provenance["attack_type"].isin(
        config["provenance_test_holdouts"]
    )
    return {
        "calibration_seen": pd.concat(
            [controlled[~controlled_holdout], provenance[~provenance_holdout]],
            ignore_index=True,
        ),
        "validation_unseen": pd.concat(
            [controlled[controlled_holdout], provenance[provenance_holdout]],
            ignore_index=True,
        ),
        "validation_unseen_controlled": controlled[controlled_holdout].copy(),
        "validation_unseen_provenance": provenance[provenance_holdout].copy(),
        "validation_unseen_controlled_severe": controlled[
            controlled["opportunity_slice"] == "severe_stress"
        ].copy(),
        "validation_unseen_controlled_stress_only_period": controlled[
            controlled["opportunity_slice"] == "stress_only_period"
        ].copy(),
        "validation_unseen_provenance_paraphrase": provenance[
            provenance["attack_type"] == "paraphrase"
        ].copy(),
        "validation_unseen_provenance_multi_hop": provenance[
            provenance["attack_type"] == "multi_hop"
        ].copy(),
        "public_validation": public,
    }


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    corpora_manifest_path = ROOT / config["outputs"]["manifest"]
    corpora_manifest = json.loads(corpora_manifest_path.read_text(encoding="utf-8"))
    if corpora_manifest.get("test_outcomes_evaluated") is not False:
        raise RuntimeError("training corpora manifest does not preserve test sealing")
    frames = _validation_frames(config)
    results: list[dict[str, object]] = []
    ablations: list[dict[str, object]] = []
    negative_controls: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    models: dict[tuple[str, int], dict[str, object]] = {}
    output_dir = ROOT / config["outputs"]["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = list(config["variant_definitions"])
    for seed in config["seeds"]:
        for variant in variants:
            corpus = pd.read_csv(
                ROOT
                / config["outputs"]["corpora_dir"]
                / f"{variant.lower()}_seed_{int(seed)}.csv",
                low_memory=False,
            )
            train_x = _feature_frame(corpus, config)
            train_y = corpus["training_label"].astype(int)
            if train_y.nunique() < 2:
                raise ValueError(f"{variant} seed {seed} has one training class")
            calibration_x = _feature_frame(frames["calibration_seen"], config)
            calibration_y = frames["calibration_seen"]
            candidates: list[dict[str, object]] = []
            fitted: dict[float, Pipeline] = {}
            started = time.perf_counter()
            for c_value in config["c_grid"]:
                model = _model(config, int(seed), float(c_value))
                model.fit(train_x, train_y)
                scores = model.predict_proba(calibration_x)[:, 1]
                threshold = _threshold(scores, float(config["validation_coverage_target"]))
                metrics = _cluster_metrics(calibration_y, scores >= threshold)
                candidates.append(
                    {
                        "c": float(c_value),
                        "threshold": threshold,
                        **metrics,
                    }
                )
                fitted[float(c_value)] = model
            selected = min(
                candidates,
                key=lambda row: (
                    float(row["far"]) if np.isfinite(row["far"]) else float("inf"),
                    abs(float(row["coverage"]) - float(config["validation_coverage_target"])),
                    -float(row["cau"]),
                    float(row["c"]),
                ),
            )
            selected_model = fitted[float(selected["c"])]
            elapsed = time.perf_counter() - started
            model_record = {
                "variant": variant,
                "seed": int(seed),
                "selected_c": float(selected["c"]),
                "threshold": float(selected["threshold"]),
                "fit_selection_seconds": elapsed,
                "calibration_far": selected["far"],
                "calibration_coverage": selected["coverage"],
            }
            models[(variant, int(seed))] = model_record
            feature_names = selected_model.named_steps["preprocessor"].get_feature_names_out()
            coefficient_values = selected_model.named_steps["classifier"].coef_[0]
            for feature_name, coefficient in zip(feature_names, coefficient_values):
                coefficients.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "feature": str(feature_name),
                        "coefficient": float(coefficient),
                        "absolute_coefficient": float(abs(coefficient)),
                    }
                )
            for namespace in (
                "validation_unseen",
                "validation_unseen_controlled",
                "validation_unseen_provenance",
                "validation_unseen_controlled_severe",
                "validation_unseen_controlled_stress_only_period",
                "validation_unseen_provenance_paraphrase",
                "validation_unseen_provenance_multi_hop",
                "public_validation",
            ):
                frame = frames[namespace]
                scores = selected_model.predict_proba(_feature_frame(frame, config))[:, 1]
                metrics = _evaluate_scores(
                    frame,
                    scores,
                    float(selected["threshold"]),
                    config,
                    int(seed),
                    f"{variant}/{namespace}",
                )
                results.append({**model_record, "evaluation": namespace, **metrics})
            unseen_frame = frames["validation_unseen"]
            for group in ("role", "cost", "belief"):
                masked = _masked_frame(unseen_frame, config, group)
                masked_scores = selected_model.predict_proba(
                    _feature_frame(masked, config)
                )[:, 1]
                masked_metrics = _cluster_metrics(
                    unseen_frame, masked_scores >= float(selected["threshold"])
                )
                ablations.append(
                    {
                        **model_record,
                        "masked_group": group,
                        **masked_metrics,
                    }
                )
            if variant == "D7":
                neutral_train = _masked_frame(corpus, config, "role")
                neutral_calibration = _masked_frame(
                    frames["calibration_seen"], config, "role"
                )
                neutral_candidates: list[dict[str, object]] = []
                neutral_models: dict[float, Pipeline] = {}
                neutral_started = time.perf_counter()
                for c_value in config["c_grid"]:
                    neutral_model = _model(config, int(seed), float(c_value))
                    neutral_model.fit(_feature_frame(neutral_train, config), train_y)
                    neutral_scores = neutral_model.predict_proba(
                        _feature_frame(neutral_calibration, config)
                    )[:, 1]
                    neutral_threshold = _threshold(
                        neutral_scores, float(config["validation_coverage_target"])
                    )
                    neutral_metrics = _cluster_metrics(
                        frames["calibration_seen"],
                        neutral_scores >= neutral_threshold,
                    )
                    neutral_candidates.append(
                        {
                            "c": float(c_value),
                            "threshold": neutral_threshold,
                            **neutral_metrics,
                        }
                    )
                    neutral_models[float(c_value)] = neutral_model
                neutral_selected = min(
                    neutral_candidates,
                    key=lambda row: (
                        float(row["far"])
                        if np.isfinite(row["far"])
                        else float("inf"),
                        abs(
                            float(row["coverage"])
                            - float(config["validation_coverage_target"])
                        ),
                        -float(row["cau"]),
                        float(row["c"]),
                    ),
                )
                neutral_model = neutral_models[float(neutral_selected["c"])]
                neutral_record = {
                    "control": "D7_role_neutral",
                    "source_variant": "D7",
                    "seed": int(seed),
                    "selected_c": float(neutral_selected["c"]),
                    "threshold": float(neutral_selected["threshold"]),
                    "fit_selection_seconds": time.perf_counter() - neutral_started,
                    "calibration_far": neutral_selected["far"],
                    "calibration_coverage": neutral_selected["coverage"],
                }
                for namespace in (
                    "validation_unseen",
                    "validation_unseen_controlled",
                    "validation_unseen_provenance",
                    "validation_unseen_controlled_severe",
                    "validation_unseen_controlled_stress_only_period",
                    "validation_unseen_provenance_paraphrase",
                    "validation_unseen_provenance_multi_hop",
                    "public_validation",
                ):
                    frame = frames[namespace]
                    neutral_frame = _masked_frame(frame, config, "role")
                    neutral_scores = neutral_model.predict_proba(
                        _feature_frame(neutral_frame, config)
                    )[:, 1]
                    neutral_metrics = _evaluate_scores(
                        frame,
                        neutral_scores,
                        float(neutral_selected["threshold"]),
                        config,
                        int(seed),
                        f"D7_role_neutral/{namespace}",
                    )
                    negative_controls.append(
                        {**neutral_record, "evaluation": namespace, **neutral_metrics}
                    )

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_dir / "validation_results.csv", index=False)
    ablation_frame = pd.DataFrame(ablations)
    ablation_frame.to_csv(output_dir / "feature_ablation.csv", index=False)
    ablation_aggregate = (
        ablation_frame.groupby(["variant", "masked_group"])[["far", "coverage", "alr", "cau"]]
        .agg(["mean", "std"])
    )
    ablation_aggregate.columns = ["_".join(column) for column in ablation_aggregate.columns]
    ablation_aggregate = ablation_aggregate.reset_index()
    ablation_aggregate.to_csv(output_dir / "feature_ablation_aggregate.csv", index=False)
    negative_control_frame = pd.DataFrame(negative_controls)
    negative_control_frame.to_csv(output_dir / "negative_control_results.csv", index=False)
    negative_control_aggregate = (
        negative_control_frame.groupby(["control", "evaluation"])[
            ["far", "coverage", "alr", "cau", "certification_volume"]
        ]
        .agg(["mean", "std"])
    )
    negative_control_aggregate.columns = [
        "_".join(column) for column in negative_control_aggregate.columns
    ]
    negative_control_aggregate = negative_control_aggregate.reset_index()
    negative_control_aggregate.to_csv(
        output_dir / "negative_control_aggregate.csv", index=False
    )
    coefficient_frame = pd.DataFrame(coefficients)
    coefficient_frame.to_csv(output_dir / "coefficient_audit.csv", index=False)
    coefficient_summary = (
        coefficient_frame.groupby(["variant", "feature"])["absolute_coefficient"]
        .mean()
        .reset_index()
        .sort_values(["variant", "absolute_coefficient"], ascending=[True, False])
    )
    coefficient_summary["rank_within_variant"] = coefficient_summary.groupby("variant").cumcount() + 1
    coefficient_summary.to_csv(output_dir / "coefficient_summary.csv", index=False)
    aggregate = (
        result_frame.groupby(["variant", "evaluation"])[
            ["far", "coverage", "alr", "cau", "certification_volume"]
        ]
        .agg(["mean", "std"])
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(output_dir / "validation_aggregate.csv", index=False)

    primary_rows: list[dict[str, object]] = []
    primary_evaluations = list(config["primary_evaluations"])
    coverage_floor = float(config["min_primary_mechanism_coverage"])
    for (variant, seed), group in result_frame[
        result_frame["evaluation"].isin(primary_evaluations)
    ].groupby(["variant", "seed"]):
        complete = len(group) == len(primary_evaluations)
        mechanism_coverage_floor = float(group["coverage"].min()) if complete else 0.0
        mechanism_coverage_lcb_floor = (
            float(group["coverage_lcb95"].min()) if complete else 0.0
        )
        endpoint_valid = bool(
            complete
            and group["far"].notna().all()
            and mechanism_coverage_lcb_floor >= coverage_floor
        )
        primary_rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "primary_far": float(group["far"].mean()) if endpoint_valid else np.nan,
                "mechanism_coverage_floor": mechanism_coverage_floor,
                "mechanism_coverage_lcb_floor": mechanism_coverage_lcb_floor,
                "primary_endpoint_valid": endpoint_valid,
                "failed_mechanisms": ",".join(
                    sorted(
                        group.loc[
                            group["far"].isna()
                            | (group["coverage_lcb95"] < coverage_floor),
                            "evaluation",
                        ].astype(str)
                    )
                ),
            }
        )
    primary_frame = pd.DataFrame(primary_rows)
    primary_frame.to_csv(output_dir / "primary_endpoint_by_seed.csv", index=False)

    negative_primary_rows: list[dict[str, object]] = []
    for (control, seed), group in negative_control_frame[
        negative_control_frame["evaluation"].isin(primary_evaluations)
    ].groupby(["control", "seed"]):
        complete = len(group) == len(primary_evaluations)
        mechanism_coverage_floor = float(group["coverage"].min()) if complete else 0.0
        mechanism_coverage_lcb_floor = (
            float(group["coverage_lcb95"].min()) if complete else 0.0
        )
        endpoint_valid = bool(
            complete
            and group["far"].notna().all()
            and mechanism_coverage_lcb_floor >= coverage_floor
        )
        negative_primary_rows.append(
            {
                "control": control,
                "seed": int(seed),
                "primary_far": float(group["far"].mean()) if endpoint_valid else np.nan,
                "mechanism_coverage_floor": mechanism_coverage_floor,
                "mechanism_coverage_lcb_floor": mechanism_coverage_lcb_floor,
                "primary_endpoint_valid": endpoint_valid,
                "failed_mechanisms": ",".join(
                    sorted(
                        group.loc[
                            group["far"].isna()
                            | (group["coverage_lcb95"] < coverage_floor),
                            "evaluation",
                        ].astype(str)
                    )
                ),
            }
        )
    negative_primary_frame = pd.DataFrame(negative_primary_rows)
    negative_primary_frame.to_csv(
        output_dir / "negative_control_primary_by_seed.csv", index=False
    )

    reference_variant = str(config["holm_reference_variant"])
    prediction_diagnostic_variant = str(config["prediction_diagnostic_variant"])
    contrast_variants = list(config["holm_variants"])
    reference = primary_frame[
        (primary_frame["variant"] == reference_variant)
        & primary_frame["primary_endpoint_valid"].astype(bool)
    ].set_index("seed")["primary_far"]
    raw_p: dict[str, float] = {}
    contrast_rows: list[dict[str, object]] = []
    for variant in contrast_variants:
        current_rows = primary_frame[
            (primary_frame["variant"] == variant)
            & primary_frame["primary_endpoint_valid"].astype(bool)
        ]
        current = current_rows.set_index("seed")["primary_far"]
        aligned = pd.concat(
            [reference.rename("reference"), current.rename("variant")], axis=1
        ).dropna()
        if (
            len(aligned) != len(config["seeds"])
            or np.allclose(aligned["reference"] - aligned["variant"], 0.0)
        ):
            pvalue = 1.0
        else:
            pvalue = float(
                ttest_rel(
                    aligned["reference"], aligned["variant"], alternative="greater"
                ).pvalue
            )
            if not np.isfinite(pvalue):
                pvalue = 1.0
        raw_p[variant] = pvalue
        contrast_rows.append(
            {
                "variant": variant,
                "reference_variant": reference_variant,
                "mean_far_difference_vs_reference": float(
                    (aligned["variant"] - aligned["reference"]).mean()
                ) if len(aligned) else np.nan,
                "raw_p": pvalue,
                "valid_seed_pairs": len(aligned),
                "all_variant_seeds_valid": bool(len(current_rows) == len(config["seeds"])),
            }
        )
    adjusted = holm_adjust(raw_p, alpha=float(config["alpha"]))
    for row in contrast_rows:
        row.update(adjusted[str(row["variant"])])
        row["development_only"] = True
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame.to_csv(output_dir / "holm_validation_contrasts.csv", index=False)

    model_frame = pd.DataFrame(models.values())
    model_frame.to_csv(output_dir / "model_selection.csv", index=False)
    report = [
        "# Training Utility Validation Smoke",
        "",
        "This report is validation-only. No controlled, provenance, or public test outcome is evaluated.",
        "",
        "## Aggregate results",
        "",
        aggregate.to_csv(index=False),
        "",
        "## Holm-adjusted development contrasts",
        "",
        f"D0 is a separate prediction-only task-transfer diagnostic. The data-utility "
        f"Holm family compares {', '.join(contrast_variants)} with the ordinary-only "
        f"authorization reference {reference_variant}.",
        "",
        contrast_frame.to_csv(index=False),
        "",
        "## Primary endpoint validity by seed",
        "",
        primary_frame.to_csv(index=False),
        "",
        "## Feature-group masking",
        "",
        ablation_aggregate.to_csv(index=False),
        "",
        "## Role-neutral training negative control",
        "",
        negative_control_aggregate.to_csv(index=False),
        "",
        negative_primary_frame.to_csv(index=False),
        "",
        "## Top coefficient groups",
        "",
        coefficient_summary[coefficient_summary["rank_within_variant"] <= 5].to_csv(index=False),
        "",
        "Public validation is exploratory because the independent public power gate failed.",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    validity = (
        primary_frame.groupby("variant")["primary_endpoint_valid"]
        .sum()
        .astype(int)
        .to_dict()
    )
    public_aggregate = aggregate[aggregate["evaluation"] == "public_validation"]
    d7_public = public_aggregate[public_aggregate["variant"] == "D7"]
    d7_role_mask = ablation_aggregate[
        (ablation_aggregate["variant"] == "D7")
        & (ablation_aggregate["masked_group"] == "role")
    ]
    role_neutral_valid = int(negative_primary_frame["primary_endpoint_valid"].sum())
    role_neutral_unseen = negative_control_aggregate[
        negative_control_aggregate["evaluation"] == "validation_unseen"
    ]
    role_neutral_public = negative_control_aggregate[
        negative_control_aggregate["evaluation"] == "public_validation"
    ]
    docs_report_path = ROOT / config["outputs"]["docs_report"]
    docs_report = [
        "# Training Utility Report",
        "",
        "## Scope",
        "",
        "This is a validation-only tabular-learning diagnostic. It does not evaluate any "
        "controlled, provenance, or public test outcome and does not support an LLM-training "
        "or deployment claim.",
        "",
        "## Matched design",
        "",
        f"D0-D7 each use {int(config['training_clusters_per_variant'])} independent training "
        f"clusters for each of {len(config['seeds'])} frozen seeds. Logistic regression is "
        "the primary learner; C and the authorization threshold are selected on seen "
        "validation mechanisms only.",
        "",
        "## Primary finding",
        "",
        "A primary endpoint is defined only when all four frozen mechanisms have a 95% "
        f"coverage lower bound of at least {float(config['min_primary_mechanism_coverage']):.2f}. "
        f"Valid seed counts are: {validity}. {prediction_diagnostic_variant} is reported "
        "separately as a prediction-only task-transfer diagnostic. The registered "
        f"data-utility family compares {', '.join(contrast_variants)} with "
        f"{reference_variant}; no contrast is valid in this smoke because the reference "
        "or candidate coverage collapses on at least one mechanism. FAR remains N/A where "
        "coverage collapses; it is not converted to zero.",
        "",
        "## Shortcut and transfer diagnostics",
        "",
        (
            f"D7 public-validation FAR is {float(d7_public['far_mean'].iloc[0]):.3f} at mean "
            f"coverage {float(d7_public['coverage_mean'].iloc[0]):.3f}. This is exploratory "
            "because the independent public power gate failed."
            if len(d7_public)
            else "D7 public validation is unavailable."
        ),
        (
            f"Masking role features raises D7 validation-unseen FAR to "
            f"{float(d7_role_mask['far_mean'].iloc[0]):.3f} and ALR to "
            f"{float(d7_role_mask['alr_mean'].iloc[0]):.3f}, showing material role-feature "
            "dependence."
            if len(d7_role_mask)
            else "D7 role-mask diagnostics are unavailable."
        ),
        (
            f"The matched D7 role-neutral training control has "
            f"{role_neutral_valid}/{len(config['seeds'])} valid primary seed endpoints; "
            f"its validation-unseen FAR is "
            f"{float(role_neutral_unseen['far_mean'].iloc[0]):.3f} at coverage "
            f"{float(role_neutral_unseen['coverage_mean'].iloc[0]):.3f}. It is a negative "
            "control and is excluded from Holm contrasts."
            if len(role_neutral_unseen)
            else "The D7 role-neutral training control is unavailable."
        ),
        (
            f"On exploratory public validation, the role-neutral control has FAR "
            f"{float(role_neutral_public['far_mean'].iloc[0]):.3f} at coverage "
            f"{float(role_neutral_public['coverage_mean'].iloc[0]):.3f}, compared with "
            f"full D7 FAR {float(d7_public['far_mean'].iloc[0]):.3f} at coverage "
            f"{float(d7_public['coverage_mean'].iloc[0]):.3f}. This cross-domain "
            "asymmetry is descriptive; its cause is not identified by the current "
            "exploratory design, and no redundancy or causal explanation is claimed."
            if len(role_neutral_public) and len(d7_public)
            else "The public role-neutral comparison is unavailable."
        ),
        "",
        "## Boundary",
        "",
        "The smoke does not establish benchmark training utility. It establishes that the "
        "current fixed learner/data budget is vulnerable to mechanism-specific coverage "
        "collapse and role shortcuts. Negative transfer and invalid endpoints are retained.",
        "",
    ]
    docs_report_path.parent.mkdir(parents=True, exist_ok=True)
    docs_report_path.write_text("\n".join(docs_report), encoding="utf-8")
    outputs = [
        "validation_results.csv",
        "validation_aggregate.csv",
        "holm_validation_contrasts.csv",
        "model_selection.csv",
        "primary_endpoint_by_seed.csv",
        "feature_ablation.csv",
        "feature_ablation_aggregate.csv",
        "negative_control_results.csv",
        "negative_control_aggregate.csv",
        "negative_control_primary_by_seed.csv",
        "coefficient_audit.csv",
        "coefficient_summary.csv",
        "report.md",
    ]
    manifest = {
        "project": config["project"],
        "version": config["version"],
        "mode": config["mode"],
        "evaluation_split": "validation",
        "confirmatory": False,
        "test_outcomes_evaluated": False,
        "primary_learner": config["primary_learner"],
        "primary_endpoint": "unseen_mechanism_far_at_validation_fixed_20pct_coverage",
        "primary_evaluations": primary_evaluations,
        "min_primary_mechanism_coverage": coverage_floor,
        "prediction_diagnostic_variant": prediction_diagnostic_variant,
        "holm_reference_variant": reference_variant,
        "holm_family": [f"{variant}_vs_{reference_variant}" for variant in contrast_variants],
        "secondary_learners_status": "not_registered_in_v0.2.0_validation_smoke",
        "negative_controls": ["D7_role_neutral"],
        "negative_controls_in_holm_family": False,
        "seeds": [int(seed) for seed in config["seeds"]],
        "c_grid": [float(value) for value in config["c_grid"]],
        "training_clusters_per_variant": int(config["training_clusters_per_variant"]),
        "validation_frames": {
            name: {
                "rows": len(frame),
                "clusters": int(frame["evaluation_cluster_id"].nunique()),
            }
            for name, frame in frames.items()
        },
        "outputs": {name: sha256(output_dir / name) for name in outputs},
        "inputs": {
            str(corpora_manifest_path.relative_to(ROOT)): sha256(corpora_manifest_path),
            str(config_path.relative_to(ROOT)): sha256(config_path),
        },
        "docs_report": {
            str(docs_report_path.relative_to(ROOT)): sha256(docs_report_path)
        },
        "claim_boundary": (
            "Validation-only tabular training-utility development. Holm contrasts are "
            "development diagnostics. Public transfer remains exploratory."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D0-D7 training utility validation smoke.")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "training_utility_smoke.yaml")
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
