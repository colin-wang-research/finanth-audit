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


CLONE_CONFIG_FIELDS = (
    "dataset",
    "schema",
    "symbol",
    "market_timezone",
    "regular_session_start",
    "regular_session_end",
    "decision_times_local",
    "decision_stride_minutes",
    "lookback_minutes",
    "action_delay_seconds",
    "holding_minutes",
    "max_quote_age_seconds",
    "one_contract",
    "benchmark_roundtrip_fee_bps",
    "exclude_intraday_instrument_changes",
    "independent_cluster",
)

FORBIDDEN_FEATURE_COLUMNS = {
    "outcome_timestamp",
    "entry_price",
    "exit_price",
    "full_utility",
    "reduced_utility",
    "realized_return",
    "harm_label",
    "tail_loss",
}


def _check(condition: bool, name: str, detail: object) -> dict[str, object]:
    return {"check": name, "passed": bool(condition), "detail": detail}


def _verify_manifest_outputs(manifest: dict[str, Any]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for relative, expected in manifest.get("outputs", {}).items():
        path = resolve_root_path(relative)
        actual = sha256(path) if path.exists() else None
        checks.append(
            _check(
                actual == expected,
                f"hash:{relative}",
                f"expected={expected} actual={actual}",
            )
        )
    return checks


def _load_all(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    source = config["databento"]
    manifest_path = resolve_root_path(source["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = _verify_manifest_outputs(manifest)
    features: list[pd.DataFrame] = []
    outcomes: list[pd.DataFrame] = []
    for split in ("development", "paper_test", "community_hidden"):
        features.append(
            pd.read_parquet(resolve_root_path(source["derived_dir"]) / f"{split}.parquet")
        )
        outcomes.append(
            pd.read_parquet(
                resolve_root_path(source["derived_dir"])
                / f"{split}_outcomes.parquet",
                columns=[
                    "row_id",
                    "event_cluster_id",
                    "outcome_timestamp",
                    "split",
                ],
            )
        )
    return (
        pd.concat(features, ignore_index=True),
        pd.concat(outcomes, ignore_index=True),
        checks,
    )


def _clone_checks(config: dict[str, Any]) -> list[dict[str, object]]:
    v03 = load_config(ROOT / "configs" / "external_orderbook_v03.yaml")
    checks = [
        _check(config["thresholds"] == v03["thresholds"], "clone:thresholds", True),
        _check(
            config["prior_families"] == v03["prior_families"],
            "clone:prior_families",
            True,
        ),
        _check(
            config["primary_endpoints"] == v03["primary_endpoints"],
            "clone:primary_endpoints",
            True,
        ),
    ]
    for field in CLONE_CONFIG_FIELDS:
        checks.append(
            _check(
                config["databento"][field] == v03["databento"][field],
                f"clone:databento.{field}",
                {
                    "v03": v03["databento"][field],
                    "v04": config["databento"][field],
                },
            )
        )
    return checks


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.get("version") != "0.4.0":
        raise RuntimeError("Databento powered audit requires v0.4.0 config")
    source = config["databento"]
    checks = _clone_checks(config)

    feature_manifest_path = resolve_root_path(config["freeze"]["feature_access"])
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

    features, sealed, hash_checks = _load_all(config)
    checks.extend(hash_checks)
    leaked = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(features.columns))
    checks.append(_check(not leaked, "features:no_postdecision_columns", leaked))
    checks.append(
        _check(
            "outcome_timestamp" in sealed.columns,
            "sealed:outcome_timestamp_present",
            list(sealed.columns),
        )
    )
    checks.append(
        _check(
            features["row_id"].is_unique,
            "databento_v04:row_ids_unique",
            len(features),
        )
    )
    checks.append(
        _check(
            set(features["row_id"]) == set(sealed["row_id"]),
            "databento_v04:sealed_alignment",
            {"features": len(features), "sealed": len(sealed)},
        )
    )

    merged = features.merge(
        sealed.drop(columns=["split"], errors="ignore"),
        on=["row_id", "event_cluster_id"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "source_timestamp",
        "decision_timestamp",
        "action_timestamp",
        "outcome_timestamp",
    ):
        merged[column] = pd.to_datetime(merged[column], utc=True, errors="coerce")
    valid = merged[
        [
            "source_timestamp",
            "decision_timestamp",
            "action_timestamp",
            "outcome_timestamp",
        ]
    ].notna().all(axis=1)
    ordered = (
        (merged["source_timestamp"] < merged["decision_timestamp"])
        & (merged["decision_timestamp"] < merged["action_timestamp"])
        & (merged["action_timestamp"] < merged["outcome_timestamp"])
    )
    checks.extend(
        [
            _check(bool(valid.all()), "databento_v04:timestamps_parse", int(valid.sum())),
            _check(bool(ordered.all()), "databento_v04:timestamps_ordered", int(ordered.sum())),
        ]
    )

    cluster_sizes = features.groupby("event_cluster_id").size()
    expected_per_cluster = len(source["decision_times_local"]) * len(
        config["prior_families"]
    )
    instrument_counts = features.groupby("event_cluster_id")["instrument_id"].nunique()
    cluster_splits = features.groupby("event_cluster_id")["split"].nunique()
    split_counts = (
        features.groupby("split")["event_cluster_id"].nunique().astype(int).to_dict()
    )
    cluster_dates = pd.to_datetime(
        features["event_cluster_id"].str.removeprefix("databento-")
    )
    start_date = pd.Timestamp(str(source["source_start_date"]))
    end_date = pd.Timestamp(str(source["source_end_date"]))
    prior_start = pd.Timestamp(str(source["prior_inspected_period_start"]))
    checks.extend(
        [
            _check(
                bool((cluster_sizes == expected_per_cluster).all()),
                "databento_v04:complete_session_clusters",
                cluster_sizes.value_counts().to_dict(),
            ),
            _check(
                bool((instrument_counts == 1).all()),
                "databento_v04:no_intraday_instrument_change",
                instrument_counts.value_counts().to_dict(),
            ),
            _check(
                bool((cluster_splits == 1).all()),
                "databento_v04:cluster_split_disjoint",
                cluster_splits.value_counts().to_dict(),
            ),
            _check(
                split_counts.get("development") == int(source["development_clusters"]),
                "databento_v04:development_clusters",
                split_counts,
            ),
            _check(
                split_counts.get("paper_test") == int(source["paper_test_clusters"]),
                "databento_v04:paper_test_clusters",
                split_counts,
            ),
            _check(
                split_counts.get("community_hidden", 0) > 0,
                "databento_v04:hidden_clusters_retained",
                split_counts,
            ),
            _check(
                bool((cluster_dates >= start_date).all()),
                "databento_v04:registered_start_enforced",
                cluster_dates.min().date().isoformat(),
            ),
            _check(
                bool((cluster_dates <= end_date).all()),
                "databento_v04:registered_end_enforced",
                cluster_dates.max().date().isoformat(),
            ),
            _check(
                bool((cluster_dates < prior_start).all()),
                "databento_v04:zero_calendar_overlap_with_v03",
                {
                    "maximum_v04": cluster_dates.max().date().isoformat(),
                    "v03_start": prior_start.date().isoformat(),
                },
            ),
        ]
    )

    split_dates = (
        features.assign(_cluster_date=cluster_dates)
        .groupby("split")["_cluster_date"]
        .agg(["min", "max"])
    )
    chronological = (
        split_dates.loc["development", "max"]
        < split_dates.loc["paper_test", "min"]
        <= split_dates.loc["paper_test", "max"]
        < split_dates.loc["community_hidden", "min"]
    )
    checks.append(
        _check(
            bool(chronological),
            "databento_v04:chronological_split_order",
            split_dates.astype(str).to_dict("index"),
        )
    )

    action_delay = (
        merged["action_timestamp"] - merged["decision_timestamp"]
    ).dt.total_seconds()
    holding = (
        merged["outcome_timestamp"] - merged["action_timestamp"]
    ).dt.total_seconds()
    checks.extend(
        [
            _check(
                bool(
                    action_delay.between(
                        int(source["action_delay_seconds"]),
                        int(source["action_delay_seconds"])
                        + int(source["max_quote_age_seconds"]),
                    ).all()
                ),
                "databento_v04:entry_quote_staleness",
                {"min": float(action_delay.min()), "max": float(action_delay.max())},
            ),
            _check(
                bool(
                    holding.between(
                        int(source["holding_minutes"]) * 60,
                        int(source["holding_minutes"]) * 60
                        + int(source["max_quote_age_seconds"]),
                    ).all()
                ),
                "databento_v04:exit_quote_staleness",
                {"min": float(holding.min()), "max": float(holding.max())},
            ),
        ]
    )

    source_manifest = json.loads(
        resolve_root_path(source["source_manifest"]).read_text(encoding="utf-8")
    )
    checks.extend(
        [
            _check(
                source_manifest.get("calendar_overlap_days") == 0,
                "source_manifest:calendar_overlap_zero",
                source_manifest.get("calendar_overlap_days"),
            ),
            _check(
                source_manifest.get("source_end_date") == str(source["source_end_date"]),
                "source_manifest:end_date",
                source_manifest.get("source_end_date"),
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
        "version": "0.4.0",
        "passed": passed,
        "checks_passed": sum(bool(item["passed"]) for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "outcome_columns_read": False,
        "sealed_execution_metadata_read": True,
        "outcome_metrics_computed": False,
        "outputs": {"feature_access_audit.csv": sha256(access_path)},
        "claim_boundary": (
            "Structural, temporal, non-overlap, split, hash, clone, and feature-access "
            "audit only. No price outcome, utility, harm, burden, or rule result is read."
        ),
    }
    report_path = resolve_root_path(config["freeze"]["structural_audit"])
    write_json(report_path, report)
    if not passed:
        failed = [item for item in checks if not item["passed"]]
        raise RuntimeError(f"Databento v0.4 structural audit failed: {failed}")
    print(report_path)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit v0.4 Databento structure without outcomes."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "databento_powered_v04.yaml")
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
