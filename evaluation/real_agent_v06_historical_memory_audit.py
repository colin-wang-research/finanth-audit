from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


AMENDMENT_PATH = ROOT / "manifests" / "preregistration" / "real_agent_v06_historical_memory_amendment.json"
CORRECTION_PATH = ROOT / "manifests" / "preregistration" / "real_agent_v06_historical_memory_merge_correction.json"


EXPLICIT_RETROSPECTIVE_PATTERNS = (
    r"\bas we now know\b",
    r"\bturned out\b",
    r"\bwent on to\b",
    # This deliberately errs toward false positives: the aggregate is a
    # contamination-signal upper bound, never evidence that memorization occurred.
    r"\b(?:subsequently|later|eventually)\s+(?:rose|fell|rallied|crashed|increased|decreased)\b",
    r"\bhas since\s+(?:risen|fallen|rallied|crashed|increased|decreased)\b",
)

UNSUPPORTED_EXTERNAL_SOURCE_PATTERNS = (
    r"\baccording to\s+(?:news|reports?|announcements?|regulatory filings?)\b",
    r"\b(?:news|media) reports?\b",
    r"\b(?:earnings release|regulatory filing|press release)\b",
)


def _decode_flags(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return str(value)
    if isinstance(parsed, list):
        return " ".join(str(item) for item in parsed)
    return str(parsed)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def annotate_text_signals(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "context_id",
        "event_cluster_id",
        "split",
        "model_id",
        "task_id",
        "action",
        "confidence",
        "uncertainty",
        "expected_edge_bps",
        "review_recommended",
        "raw_schema_valid",
        "rationale",
        "risk_flags",
        "decision_timestamp",
        "symbol",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"historical-memory audit is missing columns: {sorted(missing)}")
    annotated = frame.copy()
    text = (
        annotated["rationale"].fillna("").astype(str)
        + " "
        + annotated["risk_flags"].map(_decode_flags)
    )
    annotated["explicit_retrospective_signal"] = text.map(
        lambda value: _matches(value, EXPLICIT_RETROSPECTIVE_PATTERNS)
    )
    annotated["unsupported_external_source_signal"] = text.map(
        lambda value: _matches(value, UNSUPPORTED_EXTERNAL_SOURCE_PATTERNS)
    )
    annotated["decision_timestamp"] = pd.to_datetime(
        annotated["decision_timestamp"], utc=True, errors="raise"
    )
    annotated["calendar_quarter"] = annotated["decision_timestamp"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    annotated["active_proposal"] = annotated["action"].astype(str) != "abstain"
    annotated["absolute_expected_edge_bps"] = annotated["expected_edge_bps"].abs()
    return annotated


def temporal_summary(annotated: pd.DataFrame) -> pd.DataFrame:
    group = ["calendar_quarter", "model_id", "task_id"]
    summary = (
        annotated.groupby(group, observed=True)
        .agg(
            proposals=("context_id", "size"),
            independent_dates=("event_cluster_id", "nunique"),
            active_rate=("active_proposal", "mean"),
            review_recommended_rate=("review_recommended", "mean"),
            raw_schema_validity=("raw_schema_valid", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_uncertainty=("uncertainty", "mean"),
            mean_absolute_expected_edge_bps=("absolute_expected_edge_bps", "mean"),
            explicit_retrospective_signal_rate=("explicit_retrospective_signal", "mean"),
            unsupported_external_source_signal_rate=("unsupported_external_source_signal", "mean"),
        )
        .reset_index()
        .sort_values(group)
        .reset_index(drop=True)
    )
    return summary


def action_distribution(annotated: pd.DataFrame) -> pd.DataFrame:
    group = ["calendar_quarter", "model_id", "task_id", "action"]
    counts = (
        annotated.groupby(group, observed=True)
        .size()
        .rename("proposals")
        .reset_index()
    )
    totals = counts.groupby(group[:-1], observed=True)["proposals"].transform("sum")
    counts["action_rate"] = counts["proposals"] / totals
    return counts.sort_values(group).reset_index(drop=True)


def _jensen_shannon(left: pd.Series, right: pd.Series) -> float:
    labels = sorted(set(left.index).union(right.index))
    p = left.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    q = right.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def divergence(values: np.ndarray) -> float:
        positive = values > 0
        return float(np.sum(values[positive] * np.log2(values[positive] / midpoint[positive])))

    return float(0.5 * divergence(p) + 0.5 * divergence(q))


def temporal_drift(annotated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_id, task_id), current in annotated.groupby(
        ["model_id", "task_id"], observed=True
    ):
        quarters = sorted(current["calendar_quarter"].unique())
        if len(quarters) < 2:
            continue
        first = current[current["calendar_quarter"] == quarters[0]]
        last = current[current["calendar_quarter"] == quarters[-1]]
        first_actions = first["action"].value_counts(normalize=True)
        last_actions = last["action"].value_counts(normalize=True)
        rows.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "first_quarter": quarters[0],
                "last_quarter": quarters[-1],
                "first_proposals": len(first),
                "last_proposals": len(last),
                "active_rate_difference_last_minus_first": float(
                    last["active_proposal"].mean() - first["active_proposal"].mean()
                ),
                "mean_confidence_difference_last_minus_first": float(
                    last["confidence"].mean() - first["confidence"].mean()
                ),
                "mean_uncertainty_difference_last_minus_first": float(
                    last["uncertainty"].mean() - first["uncertainty"].mean()
                ),
                "action_distribution_jensen_shannon_bits": _jensen_shannon(
                    first_actions, last_actions
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["task_id", "model_id"]).reset_index(drop=True)


def lexical_summary(annotated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", annotated)]
    scopes.extend(
        (f"model={model_id}", current)
        for model_id, current in annotated.groupby("model_id", observed=True)
    )
    scopes.extend(
        (f"task={task_id}", current)
        for task_id, current in annotated.groupby("task_id", observed=True)
    )
    for scope, current in scopes:
        rows.append(
            {
                "scope": scope,
                "proposals": len(current),
                "independent_dates": int(current["event_cluster_id"].nunique()),
                "explicit_retrospective_signal_count": int(
                    current["explicit_retrospective_signal"].sum()
                ),
                "explicit_retrospective_signal_rate": float(
                    current["explicit_retrospective_signal"].mean()
                ),
                "unsupported_external_source_signal_count": int(
                    current["unsupported_external_source_signal"].sum()
                ),
                "unsupported_external_source_signal_rate": float(
                    current["unsupported_external_source_signal"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def merge_proposals_with_contexts(
    proposals: pd.DataFrame, contexts: pd.DataFrame
) -> pd.DataFrame:
    required_context = {"context_id", "decision_timestamp", "symbol"}
    missing = required_context.difference(contexts.columns)
    if missing:
        raise ValueError(f"context merge is missing fields: {sorted(missing)}")
    if contexts["context_id"].duplicated().any():
        raise ValueError("context merge contains duplicate context_id values")
    if "symbol" not in proposals.columns:
        raise ValueError("proposal cache is missing its registered symbol field")
    context_projection = contexts[list(required_context)].rename(
        columns={"symbol": "context_symbol"}
    )
    frame = proposals.merge(
        context_projection, on="context_id", how="left", validate="many_to_one"
    )
    if frame[["decision_timestamp", "context_symbol"]].isna().any().any():
        raise RuntimeError("proposal/context merge failed in historical-memory audit")
    symbol_mismatch = frame["symbol"].astype(str).ne(frame["context_symbol"].astype(str))
    if symbol_mismatch.any():
        raise RuntimeError("proposal symbol disagrees with the hash-verified context")
    return frame.drop(columns=["context_symbol"])


def _verify_amendment_surface() -> dict[str, Any]:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_BEFORE_ANY_OUTCOME_EVALUATION":
        raise RuntimeError("historical-memory amendment is not active")
    correction = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    if correction.get("status") != "HASH_LOCKED_AFTER_OUTPUT_BLIND_RUNTIME_FAILURE":
        raise RuntimeError("historical-memory merge correction is not active")
    superseded = set(correction.get("superseded_surface_paths", []))
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        path = resolve_root_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"historical-memory amendment surface changed: {relative}")
    for relative, expected in correction.get("surface_hashes", {}).items():
        path = resolve_root_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"historical-memory merge correction changed: {relative}")
    return {"base": amendment, "correction": correction}


def run(config_path: Path) -> Path:
    config = load_config(config_path.resolve())
    _verify_amendment_surface()
    registry = resolve_root_path(config["freeze"]["test_registry"])
    calibration = resolve_root_path(config["freeze"]["development_calibration"])
    structural_path = resolve_root_path(config["freeze"]["structural_audit"])
    output_dir = resolve_root_path(config["results_dir"]) / "historical_memory_audit"
    if registry.exists():
        raise RuntimeError("historical-memory audit must precede the paper-test registry")
    if calibration.exists():
        raise RuntimeError("historical-memory audit must precede development outcome access")
    if output_dir.exists():
        raise RuntimeError("historical-memory audit output already exists; rerun prohibited")
    if not structural_path.is_file():
        raise RuntimeError("outcome-blind structural audit must precede historical-memory audit")
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    if structural.get("passed") is not True or structural.get("outcome_files_read") is not False:
        raise RuntimeError("historical-memory audit requires a passing outcome-blind structural audit")

    proposal_manifest_path = resolve_root_path(config["binance"]["proposal_manifest"])
    proposal_manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    proposals = pd.read_csv(ROOT / proposal_manifest["proposal_file"])
    if set(proposals["split"].unique()) != {"development", "paper_test"}:
        raise RuntimeError("audit may read only persisted development and paper-test proposals")

    dataset_manifest_path = resolve_root_path(config["binance"]["dataset_manifest"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    context_dir = resolve_root_path(config["binance"]["derived_dir"])
    context_paths = [
        context_dir / "development_contexts.csv",
        context_dir / "paper_test_contexts.csv",
    ]
    declared_outputs = dataset_manifest.get("outputs", {})
    for path in context_paths:
        relative = str(path.relative_to(ROOT))
        expected = declared_outputs.get(relative)
        if not expected or sha256(path) != expected:
            raise RuntimeError(f"context timestamp surface changed before audit: {relative}")
    contexts = pd.concat(
        [
            pd.read_csv(context_paths[0]),
            pd.read_csv(context_paths[1]),
        ],
        ignore_index=True,
    )[["context_id", "decision_timestamp", "symbol"]]
    frame = merge_proposals_with_contexts(proposals, contexts)

    annotated = annotate_text_signals(frame)
    outputs = {
        "temporal_summary.csv": temporal_summary(annotated),
        "temporal_action_distribution.csv": action_distribution(annotated),
        "temporal_drift_summary.csv": temporal_drift(annotated),
        "lexical_summary.csv": lexical_summary(annotated),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="real-agent-v06-memory-audit-", dir=output_dir.parent
    ) as temporary:
        staged_dir = Path(temporary) / "completed"
        staged_dir.mkdir()
        hashes: dict[str, str] = {}
        for name, current in outputs.items():
            path = staged_dir / name
            current.to_csv(path, index=False)
            hashes[name] = sha256(path)

        manifest = {
            "project": "FinAuth-Audit",
            "version": "0.6.0",
            "status": "COMPLETED_BEFORE_ANY_OUTCOME_EVALUATION",
            "proposal_rows": len(annotated),
            "independent_dates": int(annotated["event_cluster_id"].nunique()),
            "splits_read": sorted(annotated["split"].unique().tolist()),
            "community_hidden_proposals_decrypted": False,
            "development_outcomes_read": False,
            "paper_test_outcomes_read": False,
            "community_hidden_outcomes_read": False,
            "rationale_text_persisted_in_outputs": False,
            "explicit_retrospective_patterns": list(EXPLICIT_RETROSPECTIVE_PATTERNS),
            "unsupported_external_source_patterns": list(
                UNSUPPORTED_EXTERNAL_SOURCE_PATTERNS
            ),
            "proposal_manifest_sha256": sha256(proposal_manifest_path),
            "dataset_manifest_sha256": sha256(dataset_manifest_path),
            "structural_audit_sha256": sha256(structural_path),
            "context_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in context_paths
            },
            "timestamp_integrity_assumption": (
                "Decision timestamps are trusted only as fields inside context files "
                "whose full-file hashes match the frozen dataset manifest. This audit "
                "does not make a separate provider-clock or exchange-clock claim."
            ),
            "lexical_false_positive_policy": (
                "Registered patterns intentionally favor conservative false positives. "
                "Their aggregate rate is a signal upper bound, not a memorization label."
            ),
            "interlock": {
                "structural_audit_required": True,
                "development_calibration_must_be_absent": True,
                "paper_test_registry_must_be_absent": True,
                "existing_audit_output_refused": True,
                "atomic_directory_publish": True,
            },
            "outputs": hashes,
            "interpretation": (
                "This outcome-blind audit detects only explicit retrospective wording, "
                "unsupported source references, and descriptive temporal proposal drift. "
                "Clean text does not establish absence of implicit memorization; models "
                "may use memorized patterns without explicit textual markers."
            ),
        }
        write_json(staged_dir / "manifest.json", manifest)
        os.rename(staged_dir, output_dir)
    manifest_path = output_dir / "manifest.json"
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the outcome-blind v0.6 historical-memory risk audit."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "real_agent_v06.yaml")
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
