from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finauth_audit.evaluation.real_agent_v06_common import (
    V06Rule,
    apply_decision,
    load_hash_verified_split_context,
    materialize_outcomes,
    metric_row,
    registered_rules,
)
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
)


HIDDEN_SPLITS = {"community_hidden", "community-hidden", "hidden"}
RECALIBRATION_AMENDMENT = (
    ROOT
    / "manifests"
    / "preregistration"
    / "real_agent_v06_recalibration_infeasibility_amendment.json"
)
CONTEXT_ATTACHMENT_AMENDMENT = (
    ROOT
    / "manifests"
    / "preregistration"
    / "real_agent_v06_context_attachment_correction.json"
)
CONTEXT_ATTACHMENT_FIXTURE_AMENDMENT = (
    ROOT
    / "manifests"
    / "preregistration"
    / "real_agent_v06_context_attachment_test_fixture_correction.json"
)
CONTEXT_REGISTRY_AMENDMENT = (
    ROOT
    / "manifests"
    / "preregistration"
    / "real_agent_v06_context_registry_correction.json"
)
OUTCOME_IDENTITY_MERGE_AMENDMENT = (
    ROOT
    / "manifests"
    / "preregistration"
    / "real_agent_v06_outcome_identity_merge_correction.json"
)
OBJECTIVE_METRICS = (
    "material_harm_authorization_rate",
    "authority_violation_rate",
    "normalized_task_utility",
    "coverage",
)


def _normalise_split(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def assert_development_only(frame: pd.DataFrame, split_col: str = "split") -> None:
    """Reject any calibration frame that is not exclusively development data."""

    if split_col not in frame.columns:
        raise ValueError(f"development calibration requires a {split_col!r} column")
    splits = {_normalise_split(value) for value in frame[split_col].dropna().unique()}
    if not splits:
        raise ValueError("development calibration frame has no split values")
    if splits & {value.replace("-", "_") for value in HIDDEN_SPLITS}:
        raise ValueError("community-hidden outcomes must never be read for calibration")
    if splits != {"development"}:
        raise ValueError(
            "threshold calibration is development-only; observed splits: "
            f"{sorted(splits)}"
        )


def _utc_dates(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    required = {cluster_col, timestamp_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing chronological-fold fields: {missing}")
    timestamps = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{timestamp_col} contains invalid timestamps")
    dated = pd.DataFrame(
        {
            cluster_col: frame[cluster_col].astype(str),
            "utc_date": timestamps.dt.normalize(),
        }
    )
    dates_per_cluster = dated.groupby(cluster_col, sort=False)["utc_date"].nunique()
    invalid = dates_per_cluster[dates_per_cluster != 1]
    if not invalid.empty:
        raise ValueError(
            "each event cluster must map to exactly one UTC date; invalid clusters: "
            f"{invalid.index.tolist()}"
        )
    clusters_per_date = dated.groupby("utc_date", sort=False)[cluster_col].nunique()
    duplicate_dates = clusters_per_date[clusters_per_date != 1]
    if not duplicate_dates.empty:
        raise ValueError(
            "the v0.6 independent unit is one event cluster per UTC date; "
            f"invalid dates: {[value.isoformat() for value in duplicate_dates.index]}"
        )
    return (
        dated.drop_duplicates(cluster_col)
        .sort_values(["utc_date", cluster_col], kind="mergesort")
        .reset_index(drop=True)
    )


def chronological_fold_assignments(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    cluster_col: str = "event_cluster_id",
    timestamp_col: str = "decision_timestamp",
    split_col: str = "split",
) -> pd.DataFrame:
    """Assign whole UTC-date clusters to deterministic contiguous folds.

    Models and tasks observed on the same date always receive the same fold.
    The function deliberately rejects non-development rows before inspecting
    labels or outcomes.
    """

    assert_development_only(frame, split_col=split_col)
    if folds <= 1:
        raise ValueError("folds must be greater than one")
    clusters = _utc_dates(
        frame,
        cluster_col=cluster_col,
        timestamp_col=timestamp_col,
    )
    if len(clusters) < folds:
        raise ValueError(
            f"need at least {folds} UTC-date clusters, observed {len(clusters)}"
        )
    assignments: list[pd.DataFrame] = []
    for fold, indices in enumerate(np.array_split(np.arange(len(clusters)), folds)):
        current = clusters.iloc[indices].copy()
        current["fold"] = fold
        assignments.append(current)
    return pd.concat(assignments, ignore_index=True)


def assign_chronological_folds(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    cluster_col: str = "event_cluster_id",
    timestamp_col: str = "decision_timestamp",
    split_col: str = "split",
    fold_col: str = "calibration_fold",
) -> pd.DataFrame:
    assignments = chronological_fold_assignments(
        frame,
        folds=folds,
        cluster_col=cluster_col,
        timestamp_col=timestamp_col,
        split_col=split_col,
    ).rename(columns={"fold": fold_col})
    result = frame.copy()
    result[cluster_col] = result[cluster_col].astype(str)
    result = result.merge(
        assignments[[cluster_col, fold_col]],
        on=cluster_col,
        how="left",
        validate="many_to_one",
    )
    if result[fold_col].isna().any():
        raise RuntimeError("chronological fold assignment lost an event cluster")
    result[fold_col] = result[fold_col].astype(int)
    return result


def frozen_parameter_grid(
    config: Mapping[str, Any], rule_name: str
) -> tuple[tuple[str, ...], list[dict[str, float]]]:
    """Expand one preregistered grid in deterministic lexical parameter order."""

    try:
        raw_grid = config["development_recalibration"]["grids"][rule_name]
    except KeyError as exc:
        raise KeyError(f"no frozen recalibration grid for {rule_name!r}") from exc
    if not isinstance(raw_grid, Mapping) or not raw_grid:
        raise ValueError(f"frozen grid for {rule_name!r} is empty")
    parameter_names = tuple(sorted(str(name) for name in raw_grid))
    values: list[tuple[float, ...]] = []
    for name in parameter_names:
        current = tuple(float(value) for value in raw_grid[name])
        if not current:
            raise ValueError(f"frozen grid parameter {name!r} is empty")
        values.append(current)
    candidates = [
        dict(zip(parameter_names, combination, strict=True))
        for combination in itertools.product(*values)
    ]
    candidates.sort(key=lambda item: tuple(item[name] for name in parameter_names))
    return parameter_names, candidates


def _rule_by_name(config: Mapping[str, Any], rule_name: str) -> V06Rule:
    matches = [rule for rule in registered_rules(dict(config)) if rule.name == rule_name]
    if len(matches) != 1:
        raise KeyError(f"registered authorization rule not found: {rule_name!r}")
    return matches[0]


def apply_registered_rule(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    rule_name: str,
    thresholds: Mapping[str, float],
) -> np.ndarray:
    rule = _rule_by_name(config, rule_name)
    return np.asarray(rule.decide(frame, dict(thresholds)), dtype=object)


def decision_objective_metrics(
    frame: pd.DataFrame,
    decision: Iterable[object],
) -> dict[str, float | int | None]:
    """Project the common v0.6 metric semantics onto the frozen objective."""

    decision_array = np.asarray(list(decision), dtype=object)
    if len(decision_array) != len(frame):
        raise ValueError("decision length does not match frame")
    common = metric_row(apply_decision(frame, decision_array))
    return {
        "rows": int(common["rows"]),
        "authorized_count": int(common["authorized_count"]),
        **{metric: common[metric] for metric in OBJECTIVE_METRICS},
    }


def _strict_fold_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return None
    return float(numeric.mean())


def evaluate_calibration_candidate(
    folded_frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    rule_name: str,
    candidate_thresholds: Mapping[str, float],
    base_thresholds: Mapping[str, float],
    fold_col: str = "calibration_fold",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if fold_col not in folded_frame.columns:
        raise ValueError(f"missing fold column {fold_col!r}")
    thresholds = {str(key): float(value) for key, value in base_thresholds.items()}
    thresholds.update(
        {str(key): float(value) for key, value in candidate_thresholds.items()}
    )
    fold_rows: list[dict[str, object]] = []
    for fold, held_out in folded_frame.groupby(fold_col, sort=True):
        decision = apply_registered_rule(
            held_out,
            config=config,
            rule_name=rule_name,
            thresholds=thresholds,
        )
        metrics = decision_objective_metrics(held_out, decision)
        metrics.update({"rule": rule_name, "fold": int(fold)})
        fold_rows.append(metrics)
    folds = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    summary: dict[str, object] = {
        "rule": rule_name,
        "folds": int(len(folds)),
        "candidate_thresholds": dict(candidate_thresholds),
    }
    for metric in OBJECTIVE_METRICS:
        summary[f"mean_oof_{metric}"] = _strict_fold_mean(folds[metric])
    return folds, summary


def select_calibration_candidate(
    candidates: pd.DataFrame,
    *,
    parameter_names: Iterable[str],
    minimum_coverage: float,
) -> pd.Series:
    """Apply the preregistered lexicographic objective and coverage floor."""

    parameter_names = tuple(parameter_names)
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in [0, 1]")
    required = {
        "mean_oof_material_harm_authorization_rate",
        "mean_oof_authority_violation_rate",
        "mean_oof_normalized_task_utility",
        "mean_oof_coverage",
        *parameter_names,
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate summary is missing columns: {missing}")
    eligible = candidates.copy()
    objective_columns = [
        "mean_oof_material_harm_authorization_rate",
        "mean_oof_authority_violation_rate",
        "mean_oof_normalized_task_utility",
        "mean_oof_coverage",
    ]
    finite = np.isfinite(eligible[objective_columns].to_numpy(dtype=float)).all(axis=1)
    eligible = eligible[
        finite & eligible["mean_oof_coverage"].ge(float(minimum_coverage))
    ].copy()
    if eligible.empty:
        raise RuntimeError(
            "no frozen-grid candidate has complete fold metrics and satisfies "
            f"the {minimum_coverage:.3f} coverage floor"
        )
    sort_columns = [
        "mean_oof_material_harm_authorization_rate",
        "mean_oof_authority_violation_rate",
        "mean_oof_normalized_task_utility",
        "mean_oof_coverage",
        *parameter_names,
    ]
    ascending = [True, True, False, False, *([True] * len(parameter_names))]
    selected = eligible.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).iloc[0]
    return selected


def calibrate_rule(
    development_frame: pd.DataFrame,
    config: Mapping[str, Any],
    rule_name: str,
) -> dict[str, object]:
    """Select one rule's thresholds using only frozen-grid development folds."""

    calibration = config["development_recalibration"]
    folded = assign_chronological_folds(
        development_frame,
        folds=int(calibration["folds"]),
    )
    parameter_names, grid = frozen_parameter_grid(config, rule_name)
    summaries: list[dict[str, object]] = []
    fold_results: list[pd.DataFrame] = []
    for candidate_id, candidate in enumerate(grid):
        folds, summary = evaluate_calibration_candidate(
            folded,
            config=config,
            rule_name=rule_name,
            candidate_thresholds=candidate,
            base_thresholds=config["zero_shot_thresholds"],
        )
        summary.update({"candidate_id": candidate_id, **candidate})
        folds.insert(0, "candidate_id", candidate_id)
        for name, value in candidate.items():
            folds[name] = value
        summaries.append(summary)
        fold_results.append(folds)
    candidate_frame = pd.DataFrame(summaries)
    coverage_floor = float(calibration["minimum_execute_or_reduce_coverage"])
    objective_columns = [
        "mean_oof_material_harm_authorization_rate",
        "mean_oof_authority_violation_rate",
        "mean_oof_normalized_task_utility",
        "mean_oof_coverage",
    ]
    finite = np.isfinite(candidate_frame[objective_columns].to_numpy(dtype=float)).all(
        axis=1
    )
    eligible = finite & candidate_frame["mean_oof_coverage"].ge(coverage_floor)
    candidates_meeting_floor = int(eligible.sum())
    max_observed_coverage = float(
        pd.to_numeric(candidate_frame["mean_oof_coverage"], errors="coerce").max()
    )
    if candidates_meeting_floor == 0:
        return {
            "rule": rule_name,
            "calibration_valid": False,
            "calibration_status": "NO_ELIGIBLE_CANDIDATE_COVERAGE_FLOOR",
            "calibration_reason": "NO_ELIGIBLE_CANDIDATE_COVERAGE_FLOOR",
            "coverage_floor": coverage_floor,
            "candidates_meeting_floor": 0,
            "max_observed_coverage": max_observed_coverage,
            "parameter_names": parameter_names,
            "selected_candidate_id": None,
            "selected_thresholds": None,
            "selected_objectives": None,
            "candidate_summary": candidate_frame.sort_values("candidate_id").reset_index(
                drop=True
            ),
            "fold_metrics": pd.concat(fold_results, ignore_index=True),
        }
    selected = select_calibration_candidate(
        candidate_frame,
        parameter_names=parameter_names,
        minimum_coverage=coverage_floor,
    )
    selected_thresholds = {
        name: float(selected[name]) for name in parameter_names
    }
    return {
        "rule": rule_name,
        "calibration_valid": True,
        "calibration_status": "CALIBRATED",
        "calibration_reason": None,
        "coverage_floor": coverage_floor,
        "candidates_meeting_floor": candidates_meeting_floor,
        "max_observed_coverage": max_observed_coverage,
        "parameter_names": parameter_names,
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_thresholds": selected_thresholds,
        "selected_objectives": {
            metric: float(selected[f"mean_oof_{metric}"])
            for metric in OBJECTIVE_METRICS
        },
        "candidate_summary": candidate_frame.sort_values("candidate_id").reset_index(
            drop=True
        ),
        "fold_metrics": pd.concat(fold_results, ignore_index=True),
    }


def calibrate_all_rules(
    development_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    grids = config["development_recalibration"]["grids"]
    return {
        str(rule_name): calibrate_rule(development_frame, config, str(rule_name))
        for rule_name in grids
    }


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(current) for key, current in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(current) for current in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(current) for current in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _development_frame_hash(frame: pd.DataFrame) -> str:
    """Hash canonical values without making input row order scientifically relevant."""

    columns = sorted(str(column) for column in frame.columns)
    canonical = frame.copy()
    canonical.columns = canonical.columns.map(str)
    canonical = canonical[columns]
    def cell(value: object) -> object:
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, (list, tuple, dict, np.ndarray)):
            return _json_ready(value)
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
        return _json_ready(value)

    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = pd.to_datetime(canonical[column], utc=True).map(
                lambda value: value.isoformat() if pd.notna(value) else None
            )
        else:
            canonical[column] = canonical[column].map(cell)
    row_payloads = [
        _canonical_json_bytes(record)
        for record in canonical.to_dict(orient="records")
    ]
    digest = hashlib.sha256()
    for payload in sorted(row_payloads):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_calibration_manifest(
    development_frame: pd.DataFrame,
    config: Mapping[str, Any],
    calibration_results: Mapping[str, Mapping[str, object]],
    *,
    input_hashes: Mapping[str, str] | None = None,
    output_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the deterministic pre-paper-test calibration handoff manifest."""

    assert_development_only(development_frame)
    selected_thresholds: dict[str, dict[str, float]] = {}
    calibration_validity: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    for rule_name in config["development_recalibration"]["grids"]:
        name = str(rule_name)
        if name not in calibration_results:
            raise ValueError(f"missing calibration result for {name!r}")
        result = calibration_results[name]
        valid = bool(result.get("calibration_valid", True))
        thresholds_value = result.get("selected_thresholds")
        if valid:
            if not isinstance(thresholds_value, Mapping):
                raise TypeError(f"selected_thresholds for {name!r} must be a mapping")
            thresholds = {
                str(key): float(value) for key, value in thresholds_value.items()
            }
            selected_thresholds[name] = thresholds
        folds = result["fold_metrics"]
        if not isinstance(folds, pd.DataFrame):
            raise TypeError(f"fold_metrics for {name!r} must be a DataFrame")
        selected_id_value = result.get("selected_candidate_id")
        selected_id = None if selected_id_value is None else int(selected_id_value)
        selected_folds = (
            folds.iloc[0:0].copy()
            if selected_id is None
            else folds[folds["candidate_id"].eq(selected_id)]
            .sort_values("fold", kind="mergesort")
            .reset_index(drop=True)
        )
        calibration_validity[name] = {
            "calibration_valid": valid,
            "status": str(result.get("calibration_status", "CALIBRATED")),
            "reason": result.get("calibration_reason"),
            "selected_candidate": selected_id,
            "grid_candidates": int(result["candidate_summary"].shape[0]),
            "candidates_meeting_floor": int(result.get("candidates_meeting_floor", 0)),
            "coverage_floor": float(result.get("coverage_floor", 0.0)),
            "max_observed_coverage": float(result.get("max_observed_coverage", np.nan)),
        }
        diagnostics[name] = {
            "calibration_valid": valid,
            "calibration_status": str(result.get("calibration_status", "CALIBRATED")),
            "calibration_reason": result.get("calibration_reason"),
            "selected_candidate_id": selected_id,
            "grid_candidate_count": int(result["candidate_summary"].shape[0]),
            "parameter_names": list(result["parameter_names"]),
            "selected_objectives": (
                None
                if result.get("selected_objectives") is None
                else dict(result["selected_objectives"])
            ),
            "folds": selected_folds.to_dict(orient="records"),
        }
    frozen_calibration_surface = {
        "zero_shot_thresholds": config["zero_shot_thresholds"],
        "development_recalibration": config["development_recalibration"],
    }
    resolved_input_hashes = {
        "development_frame_sha256": _development_frame_hash(development_frame),
        "frozen_calibration_surface_sha256": _sha256_json(frozen_calibration_surface),
    }
    if input_hashes:
        resolved_input_hashes.update(
            {str(key): str(value) for key, value in input_hashes.items()}
        )
    resolved_output_hashes = {
        "selected_thresholds_by_rule_sha256": _sha256_json(selected_thresholds),
        "calibration_validity_by_rule_sha256": _sha256_json(calibration_validity),
        "per_rule_fold_diagnostics_sha256": _sha256_json(diagnostics),
    }
    if output_hashes:
        resolved_output_hashes.update(
            {str(key): str(value) for key, value in output_hashes.items()}
        )
    return _json_ready(
        {
            "project": "FinAuth-Audit",
            "version": str(config.get("version", "0.6.0")),
            "status": "FROZEN_BEFORE_PAPER_TEST",
            "calibration_split": "development",
            "fold_assignment": config["development_recalibration"].get(
                "fold_assignment", "chronological_blocked_folds"
            ),
            "selected_thresholds_by_rule": selected_thresholds,
            "calibration_validity_by_rule": calibration_validity,
            "paper_test_outcomes_read": False,
            "community_hidden_outcomes_read": False,
            "input_hashes": resolved_input_hashes,
            "output_hashes": resolved_output_hashes,
            "per_rule_fold_diagnostics": diagnostics,
        }
    )


def calibrate_and_build_manifest(
    development_frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str] | None = None,
    output_hashes: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    results = calibrate_all_rules(development_frame, config)
    manifest = build_calibration_manifest(
        development_frame,
        config,
        results,
        input_hashes=input_hashes,
        output_hashes=output_hashes,
    )
    return results, manifest


def write_calibration_manifest(path: Path, manifest: Mapping[str, object]) -> str:
    """Write a canonical manifest and return its file SHA-256."""

    payload = json.dumps(
        _json_ready(manifest),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_recalibration_amendment() -> dict[str, object]:
    amendment = json.loads(RECALIBRATION_AMENDMENT.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_BEFORE_DEVELOPMENT_OUTCOME_ACCESS":
        raise RuntimeError("v0.6 recalibration infeasibility amendment is not active")
    correction = json.loads(CONTEXT_ATTACHMENT_AMENDMENT.read_text(encoding="utf-8"))
    if correction.get("status") != "HASH_LOCKED_AFTER_DEVELOPMENT_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 context-attachment correction is not active")
    fixture = json.loads(
        CONTEXT_ATTACHMENT_FIXTURE_AMENDMENT.read_text(encoding="utf-8")
    )
    if fixture.get("status") != "HASH_LOCKED_AFTER_FIXTURE_ONLY_TEST_FAILURE":
        raise RuntimeError("v0.6 context-attachment fixture correction is not active")
    registry = json.loads(CONTEXT_REGISTRY_AMENDMENT.read_text(encoding="utf-8"))
    if registry.get("status") != "HASH_LOCKED_AFTER_CONTEXT_REGISTRY_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 context-registry correction is not active")
    outcome_merge = json.loads(
        OUTCOME_IDENTITY_MERGE_AMENDMENT.read_text(encoding="utf-8")
    )
    if outcome_merge.get("status") != "HASH_LOCKED_AFTER_OUTCOME_IDENTITY_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 outcome-identity merge correction is not active")
    outcome_superseded = set(outcome_merge.get("superseded_surface_paths", []))
    registry_superseded = set(registry.get("superseded_surface_paths", []))
    fixture_superseded = set(fixture.get("superseded_surface_paths", []))
    superseded = (
        set(correction.get("superseded_surface_paths", []))
        | fixture_superseded
        | registry_superseded
        | outcome_superseded
    )
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        path = resolve_root_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"v0.6 recalibration amendment changed: {relative}")
    for relative, expected in correction.get("surface_hashes", {}).items():
        if relative in fixture_superseded | registry_superseded | outcome_superseded:
            continue
        path = resolve_root_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"v0.6 context-attachment correction changed: {relative}")
    for relative, expected in fixture.get("surface_hashes", {}).items():
        if relative in registry_superseded | outcome_superseded:
            continue
        path = resolve_root_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"v0.6 context-attachment fixture correction changed: {relative}")
    for relative, expected in registry.get("surface_hashes", {}).items():
        if relative in outcome_superseded:
            continue
        path = resolve_root_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"v0.6 context-registry correction changed: {relative}")
    for relative, expected in outcome_merge.get("surface_hashes", {}).items():
        path = resolve_root_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"v0.6 outcome-identity merge correction changed: {relative}")
    return {
        "base": amendment,
        "correction": correction,
        "fixture": fixture,
        "registry": registry,
        "outcome_merge": outcome_merge,
    }


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported calibration input format: {path}")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _guard_allowed_calibration_path(path: Path, *, outcome: bool) -> None:
    lowered = str(path.resolve()).lower().replace("-", "_")
    if "community_hidden" in lowered or "/hidden/" in lowered:
        raise ValueError(f"hidden input is prohibited during calibration: {path}")
    if outcome and path.name != "development_outcomes.parquet":
        raise ValueError(
            "calibration may read only development_outcomes.parquet; "
            f"requested {path.name}"
        )
    if outcome and "paper_test" in lowered:
        raise ValueError(f"paper-test outcomes are prohibited during calibration: {path}")


def _write_diagnostics(
    output_dir: Path,
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    candidate_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    for rule_name, result in results.items():
        candidates = result["candidate_summary"]
        folds = result["fold_metrics"]
        if not isinstance(candidates, pd.DataFrame) or not isinstance(folds, pd.DataFrame):
            raise TypeError(f"calibration diagnostics for {rule_name!r} are not tabular")
        candidate = candidates.copy()
        if "rule" not in candidate.columns:
            candidate.insert(0, "rule", str(rule_name))
        if "candidate_thresholds" in candidate.columns:
            candidate["candidate_thresholds"] = candidate["candidate_thresholds"].map(
                lambda value: json.dumps(
                    _json_ready(value), sort_keys=True, separators=(",", ":")
                )
            )
        candidate_frames.append(candidate)
        fold = folds.copy()
        if "rule" not in fold.columns:
            fold.insert(0, "rule", str(rule_name))
        fold_frames.append(fold)
    candidates_path = output_dir / "candidate_diagnostics.csv"
    folds_path = output_dir / "fold_diagnostics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(candidate_frames, ignore_index=True).sort_values(
        ["rule", "candidate_id"], kind="mergesort"
    ).to_csv(candidates_path, index=False)
    pd.concat(fold_frames, ignore_index=True).sort_values(
        ["rule", "candidate_id", "fold"], kind="mergesort"
    ).to_csv(folds_path, index=False)
    return {
        _relative(candidates_path): sha256(candidates_path),
        _relative(folds_path): sha256(folds_path),
    }


def run(config_path: Path) -> Path:
    """Calibrate once on development outcomes and freeze the handoff manifest."""

    config_path = config_path.resolve()
    config = load_config(config_path)
    _verify_recalibration_amendment()
    manifest_path = resolve_root_path(config["freeze"]["development_calibration"])
    if manifest_path.exists():
        raise FileExistsError(f"development calibration is already frozen: {manifest_path}")
    registry_path = resolve_root_path(config["freeze"]["test_registry"])
    if registry_path.exists():
        raise RuntimeError("paper-test registry exists; development recalibration is closed")

    source = config["binance"]
    proposal_manifest_path = resolve_root_path(source["proposal_manifest"])
    proposal_manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    if proposal_manifest.get("outcome_fields_read") is not False:
        raise RuntimeError("proposal manifest crossed the pre-outcome boundary")
    if proposal_manifest.get("community_hidden_plaintext_in_repository") is not False:
        raise RuntimeError("proposal manifest does not guarantee hidden plaintext exclusion")
    proposal_path = resolve_root_path(proposal_manifest["proposal_file"])
    _guard_allowed_calibration_path(proposal_path, outcome=False)
    proposals = _read_table(proposal_path)
    if "split" not in proposals.columns:
        raise ValueError("proposal cache is missing split")
    if proposals["split"].astype(str).str.lower().eq("community_hidden").any():
        raise ValueError("community-hidden proposal plaintext entered the calibration cache")
    development_proposals = proposals[
        proposals["split"].astype(str).str.lower().eq("development")
    ].copy()
    if development_proposals.empty:
        raise ValueError("proposal cache contains no development proposals")

    development_proposals, context_audit = load_hash_verified_split_context(
        development_proposals, config, "development"
    )
    outcome_path = resolve_root_path(source["derived_dir"]) / "development_outcomes.parquet"
    _guard_allowed_calibration_path(outcome_path, outcome=True)
    outcomes = pd.read_parquet(outcome_path)
    if "split" in outcomes.columns:
        observed_splits = {
            str(value).strip().lower() for value in outcomes["split"].dropna().unique()
        }
        if observed_splits != {"development"}:
            raise ValueError(
                "development outcome file contains non-development rows: "
                f"{sorted(observed_splits)}"
            )
    materialized = materialize_outcomes(development_proposals, outcomes, config)
    assert_development_only(materialized)
    expected_clusters = int(source["development_clusters"])
    observed_clusters = int(materialized["event_cluster_id"].nunique())
    if observed_clusters != expected_clusters:
        raise ValueError(
            f"development calibration expected {expected_clusters} UTC dates, "
            f"observed {observed_clusters}"
        )

    results = calibrate_all_rules(materialized, config)
    diagnostics_dir = resolve_root_path(config["results_dir"]) / "development_calibration"
    diagnostic_hashes = _write_diagnostics(diagnostics_dir, results)
    input_hashes = {
        _relative(config_path): sha256(config_path),
        _relative(proposal_manifest_path): sha256(proposal_manifest_path),
        _relative(proposal_path): sha256(proposal_path),
        _relative(outcome_path): sha256(outcome_path),
        str(context_audit["context_file"]): str(context_audit["context_sha256"]),
        str(context_audit["split_registry_file"]): str(
            context_audit["split_registry_sha256"]
        ),
        str(context_audit["dataset_manifest"]): str(
            context_audit["dataset_manifest_sha256"]
        ),
        "registered_context_provenance_sha256": _sha256_json(context_audit),
    }
    manifest = build_calibration_manifest(
        materialized,
        config,
        results,
        input_hashes=input_hashes,
        output_hashes=diagnostic_hashes,
    )
    write_calibration_manifest(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze v0.6 development-only five-fold threshold recalibration."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "real_agent_v06.yaml"),
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


# Backwards-friendly descriptive alias used by downstream orchestration.
build_chronological_folds = assign_chronological_folds


if __name__ == "__main__":
    raise SystemExit(main())
