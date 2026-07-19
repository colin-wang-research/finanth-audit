from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from finauth_audit.baselines.rules import phase1_rules
from finauth_audit.evaluation.feature_access_audit import audit_rules, load_manifest
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


def _check(condition: bool, name: str, detail: object) -> dict[str, object]:
    return {"check": name, "passed": bool(condition), "detail": detail}


def _verify_manifest_outputs(manifest: dict[str, Any]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for relative, expected in manifest.get("outputs", {}).items():
        path = resolve_root_path(relative)
        actual = sha256(path) if path.exists() else None
        checks.append(_check(actual == expected, f"hash:{relative}", f"expected={expected} actual={actual}"))
    return checks


def _load_binance(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    source = config["binance"]
    manifest_path = resolve_root_path(source["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = _verify_manifest_outputs(manifest)
    columns = [
        "row_id",
        "event_cluster_id",
        "symbol",
        "prior_family",
        "source_timestamp",
        "decision_timestamp",
        "action_timestamp",
        "outcome_timestamp",
        "depth_snapshot_age_seconds",
        "split",
    ]
    frames = []
    outcome_frames = []
    for split in ("development", "paper_test", "community_hidden"):
        path = resolve_root_path(source["derived_dir"]) / f"{split}.csv"
        frame = pd.read_csv(path, usecols=columns)
        frames.append(frame)
        outcome_path = resolve_root_path(source["derived_dir"]) / f"{split}_outcomes.parquet"
        outcome_frames.append(
            pd.read_parquet(
                outcome_path,
                columns=[
                    "row_id",
                    "event_cluster_id",
                    "outcome_depth_snapshot_timestamp",
                    "outcome_depth_snapshot_age_seconds",
                    "split",
                ],
            )
        )
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(outcome_frames, ignore_index=True),
        checks,
    )


def _load_databento(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    source = config["databento"]
    manifest_path = resolve_root_path(source["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = _verify_manifest_outputs(manifest)
    columns = [
        "row_id",
        "event_cluster_id",
        "prior_family",
        "source_timestamp",
        "decision_timestamp",
        "action_timestamp",
        "outcome_timestamp",
        "instrument_id",
        "split",
    ]
    frames = []
    for split in ("development", "paper_test", "community_hidden"):
        path = resolve_root_path(source["derived_dir"]) / f"{split}.parquet"
        frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True), checks


def _timestamp_checks(frame: pd.DataFrame, prefix: str) -> list[dict[str, object]]:
    parsed = frame.copy()
    for column in ("source_timestamp", "decision_timestamp", "action_timestamp", "outcome_timestamp"):
        parsed[column] = pd.to_datetime(parsed[column], utc=True, errors="coerce")
    valid = parsed[["source_timestamp", "decision_timestamp", "action_timestamp", "outcome_timestamp"]].notna().all(axis=1)
    ordered = (
        (parsed["source_timestamp"] < parsed["decision_timestamp"])
        & (parsed["decision_timestamp"] < parsed["action_timestamp"])
        & (parsed["action_timestamp"] < parsed["outcome_timestamp"])
    )
    return [
        _check(bool(valid.all()), f"{prefix}:timestamps_parse", f"valid={int(valid.sum())}/{len(parsed)}"),
        _check(bool(ordered.all()), f"{prefix}:timestamps_ordered", f"ordered={int(ordered.sum())}/{len(parsed)}"),
    ]


def _split_checks(
    frame: pd.DataFrame,
    prefix: str,
    development: int,
    paper_test: int,
) -> list[dict[str, object]]:
    cluster_splits = frame.groupby("event_cluster_id")["split"].nunique()
    counts = frame.groupby("split")["event_cluster_id"].nunique().astype(int).to_dict()
    return [
        _check(bool((cluster_splits == 1).all()), f"{prefix}:cluster_split_disjoint", cluster_splits.value_counts().to_dict()),
        _check(counts.get("development") == development, f"{prefix}:development_clusters", counts),
        _check(counts.get("paper_test") == paper_test, f"{prefix}:paper_test_clusters", counts),
        _check(counts.get("community_hidden", 0) > 0, f"{prefix}:hidden_clusters_retained", counts),
    ]


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    checks: list[dict[str, object]] = []
    feature_manifest_path = ROOT / "manifests" / "external_orderbook_v03_feature_access.json"
    access = audit_rules(
        phase1_rules(),
        "external_orderbook",
        load_manifest(feature_manifest_path),
    )
    checks.append(
        _check(
            bool((access["status"] == "VALID").all()),
            "feature_access",
            access[["rule", "status", "illegal_features"]].to_dict("records"),
        )
    )

    binance, binance_outcome_metadata, hash_checks = _load_binance(config)
    checks.extend(hash_checks)
    source = config["binance"]
    expected_per_cluster = len(source["symbols"]) * len(source["decision_hours_utc"]) * len(
        config["prior_families"]
    )
    cluster_sizes = binance.groupby("event_cluster_id").size()
    checks.extend(_timestamp_checks(binance, "binance"))
    checks.extend(
        _split_checks(
            binance,
            "binance",
            int(source["development_clusters"]),
            int(source["paper_test_clusters"]),
        )
    )
    checks.extend(
        [
            _check(binance["row_id"].is_unique, "binance:row_ids_unique", len(binance)),
            _check(
                bool((cluster_sizes == expected_per_cluster).all()),
                "binance:complete_date_clusters",
                cluster_sizes.value_counts().to_dict(),
            ),
            _check(
                set(binance["symbol"]) == set(source["symbols"]),
                "binance:symbol_coverage",
                sorted(set(binance["symbol"])),
            ),
            _check(
                set(binance["prior_family"]) == {item["name"] for item in config["prior_families"]},
                "binance:prior_family_coverage",
                sorted(set(binance["prior_family"])),
            ),
            _check(
                float(binance["depth_snapshot_age_seconds"].max())
                <= float(source["max_depth_snapshot_age_seconds"]),
                "binance:decision_depth_staleness",
                float(binance["depth_snapshot_age_seconds"].max()),
            ),
            _check(
                float(binance_outcome_metadata["outcome_depth_snapshot_age_seconds"].max())
                <= float(source["max_depth_snapshot_age_seconds"]),
                "binance:outcome_depth_staleness",
                float(binance_outcome_metadata["outcome_depth_snapshot_age_seconds"].max()),
            ),
            _check(
                set(binance["row_id"]) == set(binance_outcome_metadata["row_id"]),
                "binance:sealed_outcome_alignment",
                {
                    "features": int(binance["row_id"].nunique()),
                    "sealed_outcomes": int(binance_outcome_metadata["row_id"].nunique()),
                },
            ),
        ]
    )
    unique_decisions = binance[
        ["event_cluster_id", "symbol", "decision_timestamp", "outcome_timestamp"]
    ].drop_duplicates()
    buffer_ok = True
    minimum_buffer = None
    for _, group in unique_decisions.groupby(["event_cluster_id", "symbol"]):
        group = group.sort_values("decision_timestamp")
        decision = pd.to_datetime(group["decision_timestamp"], utc=True).to_numpy()
        outcome = pd.to_datetime(group["outcome_timestamp"], utc=True).to_numpy()
        if len(group) > 1:
            buffers = (decision[1:] - outcome[:-1]) / pd.Timedelta(seconds=1)
            current_min = float(buffers.min())
            minimum_buffer = current_min if minimum_buffer is None else min(minimum_buffer, current_min)
            buffer_ok = buffer_ok and bool((buffers >= int(source["next_decision_buffer_minutes"]) * 60).all())
    checks.append(_check(buffer_ok, "binance:nonoverlapping_outcomes", minimum_buffer))

    databento, hash_checks = _load_databento(config)
    checks.extend(hash_checks)
    source = config["databento"]
    expected_per_cluster = len(source["decision_times_local"]) * len(config["prior_families"])
    cluster_sizes = databento.groupby("event_cluster_id").size()
    checks.extend(_timestamp_checks(databento, "databento"))
    checks.extend(
        _split_checks(
            databento,
            "databento",
            int(source["development_clusters"]),
            int(source["paper_test_clusters"]),
        )
    )
    parsed_decision = pd.to_datetime(databento["decision_timestamp"], utc=True)
    parsed_action = pd.to_datetime(databento["action_timestamp"], utc=True)
    parsed_outcome = pd.to_datetime(databento["outcome_timestamp"], utc=True)
    action_delay = (parsed_action - parsed_decision).dt.total_seconds()
    holding = (parsed_outcome - parsed_action).dt.total_seconds()
    instrument_counts = databento.groupby("event_cluster_id")["instrument_id"].nunique()
    checks.extend(
        [
            _check(databento["row_id"].is_unique, "databento:row_ids_unique", len(databento)),
            _check(
                bool((cluster_sizes == expected_per_cluster).all()),
                "databento:complete_session_clusters",
                cluster_sizes.value_counts().to_dict(),
            ),
            _check(
                bool((instrument_counts == 1).all()),
                "databento:no_intraday_instrument_change",
                instrument_counts.value_counts().to_dict(),
            ),
            _check(
                bool(action_delay.between(1, 1 + int(source["max_quote_age_seconds"])).all()),
                "databento:entry_quote_staleness",
                {"min": float(action_delay.min()), "max": float(action_delay.max())},
            ),
            _check(
                bool(
                    holding.between(
                        int(source["holding_minutes"]) * 60,
                        int(source["holding_minutes"]) * 60 + int(source["max_quote_age_seconds"]),
                    ).all()
                ),
                "databento:exit_quote_staleness",
                {"min": float(holding.min()), "max": float(holding.max())},
            ),
        ]
    )

    results_dir = resolve_root_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    access_path = results_dir / "feature_access_audit.csv"
    access.to_csv(access_path, index=False)
    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "project": "FinAuth-Audit",
        "version": "0.3.0",
        "passed": passed,
        "checks_passed": sum(bool(item["passed"]) for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "outcome_columns_read": False,
        "sealed_execution_metadata_read": True,
        "outcome_metrics_computed": False,
        "outputs": {"feature_access_audit.csv": sha256(access_path)},
        "claim_boundary": (
            "Structural, temporal, split, hash, and feature-access audit only. "
            "The audit reads no outcome column and computes no rule result."
        ),
    }
    report_path = results_dir / "structural_audit.json"
    write_json(report_path, report)
    if not passed:
        failed = [item for item in checks if not item["passed"]]
        raise RuntimeError(f"external order-book structural audit failed: {failed}")
    print(report_path)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit v0.3 external order-book structure without outcomes.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "external_orderbook_v03.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
