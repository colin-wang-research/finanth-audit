from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    assign_chronological_splits,
    frozen_prior_parameters,
    load_config,
    resolve_root_path,
    sha256,
    stress_tag,
    write_json,
)


REQUIRED_BBO_COLUMNS = {
    "instrument_id",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
}


def _source_files(config: dict[str, Any]) -> tuple[list[Path], dict[str, object]]:
    source = config["databento"]
    upstream_path = Path(source["upstream_manifest"])
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    project_root = upstream_path.parents[2]
    records: list[dict[str, object]] = []
    paths: list[Path] = []
    for record in upstream.get("chunks", []):
        if record.get("schema") != source["schema"]:
            continue
        path = project_root / str(record["path"])
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
                "sha256": actual,
                "status": "UPSTREAM_HASH_VERIFIED",
            }
        )
    if not paths:
        raise RuntimeError("no Databento BBO files found in upstream manifest")
    source_manifest = {
        "project": "FinAuth-Audit",
        "version": "0.3.0",
        "source": source["source_name"],
        "upstream_manifest": str(upstream_path),
        "upstream_manifest_sha256": sha256(upstream_path),
        "files": records,
        "raw_or_row_level_redistribution": False,
        "entitlement_required": True,
        "outcome_metrics_computed": False,
        "claim_boundary": (
            "Licensed source provenance and hash verification only. Reproduction "
            "requires Databento entitlement; raw and row-level data are excluded from release."
        ),
    }
    return sorted(paths), source_manifest


def _read_bbo(path: Path, timezone: ZoneInfo) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[
            "instrument_id",
            "bid_px_00",
            "ask_px_00",
            "bid_sz_00",
            "ask_sz_00",
            "symbol",
        ],
    )
    if not REQUIRED_BBO_COLUMNS.issubset(frame.columns):
        raise ValueError(f"invalid Databento BBO schema: {path}")
    timestamps = pd.DatetimeIndex(frame.index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    else:
        timestamps = timestamps.tz_convert("UTC")
    frame = frame.copy()
    frame.index = timestamps
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    )
    frame = frame[
        (frame["bid_px_00"] > 0)
        & (frame["ask_px_00"] > frame["bid_px_00"])
        & (frame["bid_sz_00"] >= 1)
        & (frame["ask_sz_00"] >= 1)
    ]
    local = frame.index.tz_convert(timezone)
    frame["_local_date"] = local.date
    frame["_local_time"] = local.time
    return frame


def _first_at_or_after(frame: pd.DataFrame, timestamp: pd.Timestamp, max_age: int) -> tuple[pd.Timestamp, pd.Series] | None:
    position = frame.index.searchsorted(timestamp, side="left")
    if position >= len(frame):
        return None
    observed = frame.index[int(position)]
    age = float((observed - timestamp).total_seconds())
    if age < 0 or age > max_age:
        return None
    return observed, frame.iloc[int(position)]


def _last_at_or_before(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    max_age: int,
    *,
    strict: bool = False,
) -> tuple[pd.Timestamp, pd.Series] | None:
    side = "left" if strict else "right"
    position = frame.index.searchsorted(timestamp, side=side) - 1
    if position < 0:
        return None
    observed = frame.index[int(position)]
    age = float((timestamp - observed).total_seconds())
    if age < 0 or age > max_age:
        return None
    return observed, frame.iloc[int(position)]


def _build_session(
    session: pd.DataFrame,
    session_date: str,
    source: dict[str, Any],
    timezone: ZoneInfo,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    if session["instrument_id"].nunique() != 1:
        return [], [], "intraday_instrument_change"
    max_age = int(source["max_quote_age_seconds"])
    fee_bps = float(source["benchmark_roundtrip_fee_bps"])
    rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for local_time in source["decision_times_local"]:
        local_decision = pd.Timestamp(f"{session_date} {local_time}", tz=timezone)
        decision_timestamp = local_decision.tz_convert("UTC")
        start_timestamp = decision_timestamp - pd.Timedelta(
            minutes=int(source["lookback_minutes"])
        )
        short_timestamp = decision_timestamp - pd.Timedelta(minutes=5)
        action_target = decision_timestamp + pd.Timedelta(
            seconds=int(source["action_delay_seconds"])
        )
        current = _last_at_or_before(
            session, decision_timestamp, max_age, strict=True
        )
        start = _last_at_or_before(session, start_timestamp, max_age)
        short = _last_at_or_before(session, short_timestamp, max_age)
        entry = _first_at_or_after(session, action_target, max_age)
        if any(value is None for value in (current, start, short, entry)):
            return [], [], "missing_or_stale_required_quote"
        current_ts, current_row = current
        _, start_row = start
        _, short_row = short
        entry_ts, entry_row = entry
        outcome_target = entry_ts + pd.Timedelta(
            minutes=int(source["holding_minutes"])
        )
        exit_quote = _first_at_or_after(session, outcome_target, max_age)
        if exit_quote is None:
            return [], [], "missing_or_stale_required_quote"
        exit_ts, exit_row = exit_quote
        local_action_date = entry_ts.tz_convert(timezone).date()
        local_exit_date = exit_ts.tz_convert(timezone).date()
        expected_date = pd.Timestamp(session_date).date()
        if local_action_date != expected_date or local_exit_date != expected_date:
            return [], [], "action_or_exit_crosses_session_boundary"
        current_mid = (float(current_row["bid_px_00"]) + float(current_row["ask_px_00"])) / 2.0
        start_mid = (float(start_row["bid_px_00"]) + float(start_row["ask_px_00"])) / 2.0
        short_mid = (float(short_row["bid_px_00"]) + float(short_row["ask_px_00"])) / 2.0
        if min(current_mid, start_mid, short_mid) <= 0:
            return [], [], "nonpositive_mid"
        momentum_30 = (current_mid / start_mid - 1.0) * 10000.0
        momentum_5 = (current_mid / short_mid - 1.0) * 10000.0
        window = session.loc[start_timestamp:decision_timestamp].copy()
        window["_mid"] = (window["bid_px_00"] + window["ask_px_00"]) / 2.0
        minute_mid = window["_mid"].resample("1min").last().ffill(limit=1).dropna()
        if len(minute_mid) < 20:
            return [], [], "insufficient_feature_window"
        log_returns = np.diff(np.log(minute_mid.to_numpy(dtype=float)))
        volatility_bps = float(np.std(log_returns, ddof=1) * np.sqrt(30.0) * 10000.0)
        bid_size = float(current_row["bid_sz_00"])
        ask_size = float(current_row["ask_sz_00"])
        imbalance = float((bid_size - ask_size) / (bid_size + ask_size))
        spread_bps = float(
            (float(current_row["ask_px_00"]) - float(current_row["bid_px_00"]))
            / current_mid
            * 10000.0
        )
        thinness = float(np.clip(1.0 - min(bid_size, ask_size) / 10.0, 0.0, 1.0))
        volatility_proxy = float(np.clip(volatility_bps / 100.0, 0.0, 1.5))
        priors = frozen_prior_parameters(
            momentum_30_bps=momentum_30,
            momentum_5_bps=momentum_5,
            volatility_bps=volatility_bps,
            depth_imbalance=imbalance,
            thinness=thinness,
        )
        for prior in priors:
            if prior.direction > 0:
                entry_price = float(entry_row["ask_px_00"])
                exit_price = float(exit_row["bid_px_00"])
            else:
                entry_price = float(entry_row["bid_px_00"])
                exit_price = float(exit_row["ask_px_00"])
            cluster = f"databento-{session_date}"
            row_id = f"{cluster}-MES-{local_time.replace(':', '')}-{prior.name}"
            rows.append(
                {
                    "row_id": row_id,
                    "event_cluster_id": cluster,
                    "source": "databento_mes_bbo",
                    "symbol": "MES.v.0",
                    "prior_family": prior.name,
                    "source_timestamp": current_ts.isoformat(),
                    "decision_timestamp": decision_timestamp.isoformat(),
                    "action_timestamp": entry_ts.isoformat(),
                    "outcome_timestamp": exit_ts.isoformat(),
                    "timestamp": decision_timestamp.isoformat(),
                    "candidate_action": prior.direction,
                    "confidence": prior.confidence,
                    "uncertainty": prior.uncertainty,
                    "expected_edge_bps": prior.expected_edge_bps,
                    "liquidity_cost_bps": spread_bps,
                    "turnover_cost_bps": 0.0,
                    "fee_bps": fee_bps,
                    "volatility_proxy": volatility_proxy,
                    "liquidity_proxy": min(bid_size, ask_size),
                    "horizon": int(source["holding_minutes"]),
                    "source_role": prior.source_role,
                    "claimed_role": prior.source_role,
                    "current_role_verified": True,
                    "original_source_eligible": prior.original_source_eligible,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "depth_imbalance": imbalance,
                    "instrument_id": int(current_row["instrument_id"]),
                    "stress_tag": stress_tag(prior.name, spread_bps, volatility_proxy),
                    "outcome_used_by_rule": False,
                }
            )
            outcome_rows.append(
                {
                    "row_id": row_id,
                    "event_cluster_id": cluster,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                }
            )
    return rows, outcome_rows, None


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    source = config["databento"]
    timezone = ZoneInfo(str(source["market_timezone"]))
    source_paths, source_manifest = _source_files(config)
    source_manifest_path = resolve_root_path(source["source_manifest"])
    write_json(source_manifest_path, source_manifest)
    rows_by_session: dict[str, list[dict[str, object]]] = {}
    outcomes_by_session: dict[str, list[dict[str, object]]] = {}
    exclusions: dict[str, int] = {}
    session_start = datetime.strptime(str(source["regular_session_start"]), "%H:%M:%S").time()
    session_end = datetime.strptime(str(source["regular_session_end"]), "%H:%M:%S").time()
    for path in source_paths:
        frame = _read_bbo(path, timezone)
        frame = frame[
            (frame["_local_time"] >= session_start) & (frame["_local_time"] <= session_end)
        ]
        for local_date, session in frame.groupby("_local_date", sort=True):
            session_date = str(local_date)
            rows, outcomes, reason = _build_session(
                session, session_date, source, timezone
            )
            if reason is not None:
                exclusions[reason] = exclusions.get(reason, 0) + 1
                continue
            expected = len(source["decision_times_local"]) * len(config["prior_families"])
            if len(rows) != expected:
                exclusions["incomplete_session"] = exclusions.get("incomplete_session", 0) + 1
                continue
            cluster = f"databento-{session_date}"
            if cluster in rows_by_session:
                exclusions["duplicate_session"] = exclusions.get("duplicate_session", 0) + 1
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
    outcomes = pd.DataFrame(outcome_rows).sort_values(["event_cluster_id", "row_id"])
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
        [{"event_cluster_id": cluster, "split": split} for cluster, split in split_map.items()]
    ).sort_values("event_cluster_id")
    split_path = derived_dir / "split_registry.csv"
    split_registry.to_csv(split_path, index=False)
    outputs[str(split_path.relative_to(ROOT))] = sha256(split_path)
    manifest_path = resolve_root_path(source["dataset_manifest"])
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.3.0",
        "source": source["source_name"],
        "rows": len(frame),
        "event_clusters": int(frame["event_cluster_id"].nunique()),
        "rows_per_cluster": {
            "min": int(frame.groupby("event_cluster_id").size().min()),
            "max": int(frame.groupby("event_cluster_id").size().max()),
        },
        "splits": frame.groupby("split")["event_cluster_id"].nunique().astype(int).to_dict(),
        "rows_by_split": frame["split"].value_counts().sort_index().astype(int).to_dict(),
        "prior_families": frame["prior_family"].value_counts().sort_index().astype(int).to_dict(),
        "source_roles": frame["source_role"].value_counts().sort_index().astype(int).to_dict(),
        "exclusions": exclusions,
        "independent_cluster": source["independent_cluster"],
        "raw_or_row_level_redistribution": False,
        "entitlement_required": True,
        "outcome_metrics_computed": False,
        "outcome_inputs_materialized_separately": True,
        "inputs": {
            str(config_path.relative_to(ROOT)): sha256(config_path),
            str(source_manifest_path.relative_to(ROOT)): sha256(source_manifest_path),
        },
        "outputs": outputs,
        "claim_boundary": (
            "Licensed deterministic row construction and chronological split only. "
            "No rule metric, rank, external effect size, or external power is computed."
        ),
    }
    write_json(manifest_path, manifest)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the preregistered Databento BBO layer.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "external_orderbook_v03.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
