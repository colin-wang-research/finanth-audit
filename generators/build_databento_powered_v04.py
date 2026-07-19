from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from finauth_audit.generators.build_databento_bbo_v03 import (
    _build_session,
    _read_bbo,
)
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    assign_chronological_splits,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


def _resolve_upstream_record(upstream_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    # Databento manifests are rooted at the Databento project checkout.
    return upstream_path.parents[2] / candidate


def _source_files(config: dict[str, Any]) -> tuple[list[Path], dict[str, object]]:
    source = config["databento"]
    upstream_path = Path(source["upstream_manifest"])
    estimate_path = Path(source["estimate_manifest"])
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    expected_start = str(source["source_start_date"])
    expected_end = str(source["source_end_date"])
    if upstream.get("status") != "COMPLETE":
        raise RuntimeError("Databento v0.4 upstream manifest is not complete")
    for manifest_name, manifest in (("upstream", upstream), ("estimate", estimate)):
        if manifest.get("dataset") != source["dataset"]:
            raise RuntimeError(f"{manifest_name} dataset mismatch")
        if manifest.get("symbol") != source["symbol"]:
            raise RuntimeError(f"{manifest_name} symbol mismatch")
        if manifest.get("start_date_inclusive") != expected_start:
            raise RuntimeError(f"{manifest_name} start date mismatch")
        if manifest.get("end_date_inclusive") != expected_end:
            raise RuntimeError(f"{manifest_name} end date mismatch")
        if list(manifest.get("schemas", [])) != [source["schema"]]:
            raise RuntimeError(f"{manifest_name} schema mismatch")
    estimate_schema = estimate.get("preflight", {}).get("schemas", [{}])[0]
    if int(estimate_schema.get("estimated_records", -1)) != 87_534_222:
        raise RuntimeError("Databento v0.4 estimated record count changed")
    if abs(float(estimate_schema.get("estimated_cost_usd", -1.0)) - 116.230500787497) > 1e-9:
        raise RuntimeError("Databento v0.4 estimated cost changed")

    records: list[dict[str, object]] = []
    paths: list[Path] = []
    for record in upstream.get("chunks", []):
        if record.get("schema") != source["schema"]:
            continue
        if record.get("status") not in {"DOWNLOADED", "CACHE_HIT"}:
            raise RuntimeError(f"incomplete Databento chunk: {record}")
        if str(record.get("start_date_inclusive")) < expected_start:
            raise RuntimeError("Databento chunk starts before the registered period")
        if str(record.get("end_date_inclusive")) > expected_end:
            raise RuntimeError("Databento chunk ends after the registered period")
        path = _resolve_upstream_record(upstream_path, str(record["path"]))
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"Databento upstream hash mismatch: {path}")
        paths.append(path)
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "records": int(record["records"]),
                "start_date_inclusive": str(record["start_date_inclusive"]),
                "end_date_inclusive": str(record["end_date_inclusive"]),
                "sha256": actual,
                "status": "UPSTREAM_HASH_VERIFIED",
            }
        )
    if not paths:
        raise RuntimeError("no Databento v0.4 BBO files found")
    source_manifest = {
        "project": "FinAuth-Audit",
        "version": "0.4.0",
        "source": source["source_name"],
        "source_start_date": expected_start,
        "source_end_date": expected_end,
        "prior_inspected_period_start": str(source["prior_inspected_period_start"]),
        "calendar_overlap_days": 0,
        "upstream_manifest": str(upstream_path),
        "upstream_manifest_sha256": sha256(upstream_path),
        "estimate_manifest": str(estimate_path),
        "estimate_manifest_sha256": sha256(estimate_path),
        "files": records,
        "raw_or_row_level_redistribution": False,
        "entitlement_required": True,
        "outcome_metrics_computed": False,
        "claim_boundary": (
            "Licensed source provenance and hash verification only. Reproduction "
            "requires Databento entitlement; raw and row-level data are excluded."
        ),
    }
    return sorted(paths), source_manifest


def _build_session_v04(
    session: pd.DataFrame,
    session_date: str,
    source: dict[str, Any],
    timezone: ZoneInfo,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    start = pd.Timestamp(str(source["source_start_date"])).date()
    end = pd.Timestamp(str(source["source_end_date"])).date()
    prior_start = pd.Timestamp(str(source["prior_inspected_period_start"])).date()
    current = pd.Timestamp(session_date).date()
    if current >= prior_start:
        raise RuntimeError(f"session overlaps inspected v0.3 period: {session_date}")
    if current < start or current > end:
        raise RuntimeError(f"session outside registered v0.4 period: {session_date}")
    rows, outcomes, reason = _build_session(session, session_date, source, timezone)
    if reason is not None:
        return rows, outcomes, reason
    if len(rows) != len(outcomes):
        raise RuntimeError("v0.3 clone returned misaligned feature and outcome rows")
    for feature, outcome in zip(rows, outcomes, strict=True):
        outcome["outcome_timestamp"] = feature.pop("outcome_timestamp")
    return rows, outcomes, None


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.get("version") != "0.4.0":
        raise RuntimeError("Databento powered builder requires v0.4.0 config")
    source = config["databento"]
    timezone = ZoneInfo(str(source["market_timezone"]))
    source_paths, source_manifest = _source_files(config)
    source_manifest_path = resolve_root_path(source["source_manifest"])
    write_json(source_manifest_path, source_manifest)

    rows_by_session: dict[str, list[dict[str, object]]] = {}
    outcomes_by_session: dict[str, list[dict[str, object]]] = {}
    exclusions: dict[str, int] = {}
    session_start = datetime.strptime(
        str(source["regular_session_start"]), "%H:%M:%S"
    ).time()
    session_end = datetime.strptime(
        str(source["regular_session_end"]), "%H:%M:%S"
    ).time()
    for path in source_paths:
        frame = _read_bbo(path, timezone)
        frame = frame[
            (frame["_local_time"] >= session_start)
            & (frame["_local_time"] <= session_end)
        ]
        for local_date, session in frame.groupby("_local_date", sort=True):
            session_date = str(local_date)
            rows, outcomes, reason = _build_session_v04(
                session, session_date, source, timezone
            )
            if reason is not None:
                exclusions[reason] = exclusions.get(reason, 0) + 1
                continue
            expected = len(source["decision_times_local"]) * len(
                config["prior_families"]
            )
            if len(rows) != expected:
                exclusions["incomplete_session"] = (
                    exclusions.get("incomplete_session", 0) + 1
                )
                continue
            cluster = f"databento-{session_date}"
            if cluster in rows_by_session:
                exclusions["duplicate_session"] = (
                    exclusions.get("duplicate_session", 0) + 1
                )
                continue
            rows_by_session[cluster] = rows
            outcomes_by_session[cluster] = outcomes

    split_map = assign_chronological_splits(
        rows_by_session,
        development=int(source["development_clusters"]),
        paper_test=int(source["paper_test_clusters"]),
    )
    rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for cluster in sorted(rows_by_session):
        for row in rows_by_session[cluster]:
            row["split"] = split_map[cluster]
            rows.append(row)
        for row in outcomes_by_session[cluster]:
            row["split"] = split_map[cluster]
            outcome_rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["event_cluster_id", "decision_timestamp", "prior_family"]
    )
    outcomes = pd.DataFrame(outcome_rows).sort_values(
        ["event_cluster_id", "row_id"]
    )
    if "outcome_timestamp" in frame.columns:
        raise RuntimeError("post-decision outcome_timestamp leaked into feature rows")
    if "outcome_timestamp" not in outcomes.columns:
        raise RuntimeError("sealed outcomes do not contain outcome_timestamp")

    derived_dir = resolve_root_path(source["derived_dir"])
    derived_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for split in ("development", "paper_test", "community_hidden"):
        path = derived_dir / f"{split}.parquet"
        frame[frame["split"] == split].to_parquet(path, index=False)
        outputs[str(path.relative_to(ROOT))] = sha256(path)
        outcome_path = derived_dir / f"{split}_outcomes.parquet"
        outcomes[outcomes["split"] == split].to_parquet(outcome_path, index=False)
        outputs[str(outcome_path.relative_to(ROOT))] = sha256(outcome_path)
    split_registry = pd.DataFrame(
        [
            {"event_cluster_id": cluster, "split": split}
            for cluster, split in split_map.items()
        ]
    ).sort_values("event_cluster_id")
    split_path = derived_dir / "split_registry.csv"
    split_registry.to_csv(split_path, index=False)
    outputs[str(split_path.relative_to(ROOT))] = sha256(split_path)

    manifest_path = resolve_root_path(source["dataset_manifest"])
    cluster_dates = pd.to_datetime(
        frame["event_cluster_id"].str.removeprefix("databento-")
    )
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.4.0",
        "source": source["source_name"],
        "rows": len(frame),
        "event_clusters": int(frame["event_cluster_id"].nunique()),
        "minimum_session_date": cluster_dates.min().date().isoformat(),
        "maximum_session_date": cluster_dates.max().date().isoformat(),
        "calendar_overlap_with_v03": False,
        "rows_per_cluster": {
            "min": int(frame.groupby("event_cluster_id").size().min()),
            "max": int(frame.groupby("event_cluster_id").size().max()),
        },
        "splits": frame.groupby("split")["event_cluster_id"]
        .nunique()
        .astype(int)
        .to_dict(),
        "rows_by_split": frame["split"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "prior_families": frame["prior_family"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "source_roles": frame["source_role"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "exclusions": exclusions,
        "independent_cluster": source["independent_cluster"],
        "raw_or_row_level_redistribution": False,
        "entitlement_required": True,
        "outcome_metrics_computed": False,
        "outcome_inputs_materialized_separately": True,
        "post_decision_fields_absent_from_features": ["outcome_timestamp"],
        "inputs": {
            str(config_path.relative_to(ROOT)): sha256(config_path),
            str(source_manifest_path.relative_to(ROOT)): sha256(source_manifest_path),
        },
        "outputs": outputs,
        "claim_boundary": (
            "Licensed deterministic row construction and chronological split only. "
            "No rule metric, rank, effect size, or external power is computed."
        ),
    }
    write_json(manifest_path, manifest)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the preregistered powered Databento v0.4 layer."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "databento_powered_v04.yaml")
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
