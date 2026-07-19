from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finauth_audit.generators.build_binance_depth_v03 import (
    _asof_row,
    _depth_path,
    _exact_kline,
    _load_depth,
    _load_klines,
    _month_path,
)
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    date_range,
    impact_bps,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)
from finauth_audit.generators.fetch_real_agent_v06 import (
    VERIFIED_STATUSES,
    assigned_symbol_for_date,
)


CONTEXT_AUDIT_FIELDS = {"source_timestamp", "action_timestamp", "volatility_proxy"}


def risk_role_for_cluster(cluster_id: str, task_config: dict[str, Any]) -> str:
    low_bit = hashlib.sha256(cluster_id.encode("utf-8")).digest()[-1] & 1
    key = "eligible_source_role" if low_bit == 0 else "ineligible_source_role"
    return str(task_config[key])


def _split_for_index(index: int, source: dict[str, Any]) -> str:
    development = int(source["development_clusters"])
    paper_test = int(source["paper_test_clusters"])
    if index < development:
        return "development"
    if index < development + paper_test:
        return "paper_test"
    return "community_hidden"


def _future_window(
    klines: pd.DataFrame,
    action: pd.Timestamp,
    outcome: pd.Timestamp,
    horizon_minutes: int,
) -> pd.DataFrame | None:
    expected = pd.date_range(action, outcome, freq="1min", tz="UTC")
    if len(expected) != horizon_minutes + 1 or not expected.isin(klines.index).all():
        return None
    frame = klines.loc[expected]
    required = {"open", "high", "low", "close"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        return None
    if (frame[list(required)] <= 0).any().any():
        return None
    return frame


def _realized_volatility_bps(window: pd.DataFrame, horizon_minutes: int) -> float:
    closes = window["close"].to_numpy(dtype=float)
    returns = np.diff(np.log(closes))
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(horizon_minutes) * 10000.0)


def _task_outcome(
    *,
    context_id: str,
    cluster_id: str,
    task_id: str,
    symbol: str,
    assigned_source_role: str,
    source_timestamp: pd.Timestamp,
    decision: pd.Timestamp,
    action: pd.Timestamp,
    outcome_timestamp: pd.Timestamp,
    decision_depth: pd.Series,
    action_depth: pd.Series,
    directional_depth: pd.Series,
    risk_depth: pd.Series,
    directional_window: pd.DataFrame,
    risk_window: pd.DataFrame,
) -> dict[str, object]:
    entry_price = float(directional_window.iloc[0]["open"])
    exit_price_30m = float(directional_window.iloc[-1]["open"])
    exit_price_60m = float(risk_window.iloc[-1]["open"])
    entry_depth = float(action_depth["bid_depth_1pct"] + action_depth["ask_depth_1pct"])
    directional_future_depth = float(
        directional_depth["bid_depth_1pct"] + directional_depth["ask_depth_1pct"]
    )
    risk_future_depth = float(risk_depth["bid_depth_1pct"] + risk_depth["ask_depth_1pct"])
    horizon_minutes = int((outcome_timestamp - action).total_seconds() // 60)
    task_window = directional_window if task_id == "directional_execution" else risk_window
    task_depth = directional_depth if task_id == "directional_execution" else risk_depth
    task_exit_price = exit_price_30m if task_id == "directional_execution" else exit_price_60m
    task_future_depth = (
        directional_future_depth if task_id == "directional_execution" else risk_future_depth
    )
    return {
        "context_id": context_id,
        "event_cluster_id": cluster_id,
        "task_id": task_id,
        "symbol": symbol,
        "assigned_source_role": assigned_source_role,
        "source_timestamp": source_timestamp.isoformat(),
        "decision_timestamp": decision.isoformat(),
        "action_timestamp": action.isoformat(),
        "outcome_timestamp": outcome_timestamp.isoformat(),
        "entry_price": entry_price,
        "exit_price": task_exit_price,
        "exit_price_30m": exit_price_30m,
        "exit_price_60m": exit_price_60m,
        "entry_bid_depth_1pct": float(action_depth["bid_depth_1pct"]),
        "entry_ask_depth_1pct": float(action_depth["ask_depth_1pct"]),
        "exit_bid_depth_1pct_30m": float(directional_depth["bid_depth_1pct"]),
        "exit_ask_depth_1pct_30m": float(directional_depth["ask_depth_1pct"]),
        "exit_bid_depth_1pct_60m": float(risk_depth["bid_depth_1pct"]),
        "exit_ask_depth_1pct_60m": float(risk_depth["ask_depth_1pct"]),
        "future_bid_depth_1pct": float(task_depth["bid_depth_1pct"]),
        "future_ask_depth_1pct": float(task_depth["ask_depth_1pct"]),
        "entry_depth": entry_depth,
        "future_depth": task_future_depth,
        "future_depth_deterioration_fraction": float(
            max(0.0, 1.0 - task_future_depth / entry_depth)
        ),
        "depth_deterioration_fraction_60m": float(
            max(0.0, 1.0 - risk_future_depth / entry_depth)
        ),
        "future_path": [float(value) for value in task_window["close"].to_numpy()],
        "future_min_price": float(task_window["low"].min()),
        "future_max_price": float(task_window["high"].max()),
        "future_min_price_30m": float(directional_window["low"].min()),
        "future_max_price_30m": float(directional_window["high"].max()),
        "future_abs_move_bps": float(
            abs(task_exit_price / entry_price - 1.0) * 10000.0
        ),
        "future_realized_volatility": _realized_volatility_bps(
            task_window, horizon_minutes
        ),
        "future_realized_volatility_bps_60m": _realized_volatility_bps(
            risk_window, 60
        ),
        "depth_snapshot_timestamp": decision_depth["_observed_timestamp"].isoformat(),
        "depth_snapshot_age_seconds": float(decision_depth["_age_seconds"]),
        "entry_depth_snapshot_timestamp": action_depth[
            "_observed_timestamp"
        ].isoformat(),
        "entry_depth_snapshot_age_seconds": float(action_depth["_age_seconds"]),
        "future_depth_snapshot_timestamp": task_depth[
            "_observed_timestamp"
        ].isoformat(),
        "future_depth_snapshot_age_seconds": float(task_depth["_age_seconds"]),
    }


def _build_date_contexts(
    *,
    source: dict[str, Any],
    tasks: dict[str, Any],
    current: date,
    symbol: str,
    klines: pd.DataFrame,
    depth: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    decision = pd.Timestamp(current, tz="UTC") + pd.Timedelta(
        hours=int(source["decision_hour_utc"])
    )
    lookback = int(source["lookback_minutes"])
    feature_start = decision - pd.Timedelta(minutes=lookback + 1)
    feature_end = decision - pd.Timedelta(minutes=1)
    short_start = decision - pd.Timedelta(minutes=6)
    action = decision + pd.Timedelta(minutes=int(source["action_delay_minutes"]))
    directional_outcome = action + pd.Timedelta(
        minutes=int(source["directional_holding_minutes"])
    )
    risk_outcome = action + pd.Timedelta(minutes=int(source["risk_horizon_minutes"]))

    required = [
        _exact_kline(klines, feature_start),
        _exact_kline(klines, feature_end),
        _exact_kline(klines, short_start),
    ]
    if any(value is None for value in required):
        return [], [], "missing_required_feature_kline"
    start_bar, end_bar, short_bar = required
    trailing = klines.loc[feature_start:feature_end, "close"]
    if len(trailing) != lookback + 1 or trailing.isna().any() or (trailing <= 0).any():
        return [], [], "insufficient_feature_window"

    directional_window = _future_window(
        klines,
        action,
        directional_outcome,
        int(source["directional_holding_minutes"]),
    )
    risk_window = _future_window(
        klines, action, risk_outcome, int(source["risk_horizon_minutes"])
    )
    if directional_window is None or risk_window is None:
        return [], [], "missing_required_outcome_kline"

    max_age = int(source["max_depth_snapshot_age_seconds"])
    decision_depth = _asof_row(
        depth, decision - pd.Timedelta(microseconds=1), max_age
    )
    action_depth = _asof_row(depth, action, max_age)
    directional_depth = _asof_row(depth, directional_outcome, max_age)
    risk_depth = _asof_row(depth, risk_outcome, max_age)
    if (
        decision_depth is None
        or action_depth is None
        or directional_depth is None
        or risk_depth is None
    ):
        return [], [], "missing_or_stale_depth"

    bid_depth = float(decision_depth["bid_depth_1pct"])
    ask_depth = float(decision_depth["ask_depth_1pct"])
    if min(
        bid_depth,
        ask_depth,
        float(action_depth["bid_depth_1pct"]),
        float(action_depth["ask_depth_1pct"]),
        float(directional_depth["bid_depth_1pct"]),
        float(directional_depth["ask_depth_1pct"]),
        float(risk_depth["bid_depth_1pct"]),
        float(risk_depth["ask_depth_1pct"]),
    ) <= 0:
        return [], [], "nonpositive_depth"

    start_close = float(start_bar["close"])
    end_close = float(end_bar["close"])
    short_close = float(short_bar["close"])
    if min(start_close, end_close, short_close) <= 0:
        return [], [], "nonpositive_feature_price"
    momentum_30 = (end_close / start_close - 1.0) * 10000.0
    momentum_5 = (end_close / short_close - 1.0) * 10000.0
    feature_returns = np.diff(np.log(trailing.to_numpy(dtype=float)))
    volatility_bps = float(
        np.std(feature_returns, ddof=1) * np.sqrt(lookback) * 10000.0
    )
    imbalance = float((bid_depth - ask_depth) / (bid_depth + ask_depth))
    notional = float(source["action_notional_usd"])
    slope = float(source["impact_slope_bps"])
    cap = float(source["impact_cap_bps"])
    cluster_id = f"binance-{current.isoformat()}"
    risk_role = risk_role_for_cluster(cluster_id, tasks["risk_limit_increase"])
    latest_source_timestamp = max(
        feature_end, pd.Timestamp(decision_depth["_observed_timestamp"])
    )

    common = {
        "symbol": symbol,
        "source_timestamp": latest_source_timestamp.isoformat(),
        "decision_timestamp": decision.isoformat(),
        "action_timestamp": action.isoformat(),
        "momentum_30_bps": float(momentum_30),
        "momentum_5_bps": float(momentum_5),
        "volatility_bps": volatility_bps,
        "volatility_proxy": float(np.clip(volatility_bps / 100.0, 0.0, 1.5)),
        "bid_depth_1pct": bid_depth,
        "ask_depth_1pct": ask_depth,
        "depth_imbalance": imbalance,
        "estimated_long_liquidity_cost_bps": float(
            2.0 * impact_bps(notional, ask_depth, slope, cap)
        ),
        "estimated_short_liquidity_cost_bps": float(
            2.0 * impact_bps(notional, bid_depth, slope, cap)
        ),
        "roundtrip_fee_bps": float(source["roundtrip_fee_bps"]),
        "directional_holding_minutes": int(source["directional_holding_minutes"]),
        "risk_horizon_minutes": int(source["risk_horizon_minutes"]),
        "risk_limit_increase_percent": float(
            tasks["risk_limit_increase"]["limit_increase_percent"]
        ),
    }
    task_specs = [
        (
            "directional_execution",
            str(tasks["directional_execution"]["source_role"]),
            directional_outcome,
        ),
        (
            "risk_limit_increase",
            risk_role,
            risk_outcome,
        ),
    ]
    contexts: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for task_id, assigned_role, outcome_timestamp in task_specs:
        context_id = f"real-agent-v06-{current.isoformat()}-{task_id}"
        context = {
            "context_id": context_id,
            "task_id": task_id,
            "assigned_source_role": assigned_role,
            **common,
        }
        contexts.append(context)
        outcomes.append(
            _task_outcome(
                context_id=context_id,
                cluster_id=cluster_id,
                task_id=task_id,
                symbol=symbol,
                assigned_source_role=assigned_role,
                source_timestamp=latest_source_timestamp,
                decision=decision,
                action=action,
                outcome_timestamp=outcome_timestamp,
                decision_depth=decision_depth,
                action_depth=action_depth,
                directional_depth=directional_depth,
                risk_depth=risk_depth,
                directional_window=directional_window,
                risk_window=risk_window,
            )
        )
    return contexts, outcomes, None


def _source_records(manifest: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in manifest.get("records", []):
        key = (str(record["kind"]), str(record["symbol"]), str(record["period"]))
        if key in records:
            raise RuntimeError(f"duplicate source-manifest record: {key}")
        records[key] = record
    return records


def _verified_archive(
    records: dict[tuple[str, str, str], dict[str, Any]],
    kind: str,
    symbol: str,
    period: str,
) -> Path | None:
    record = records.get((kind, symbol, period))
    if record is None:
        raise RuntimeError(f"source manifest is missing {(kind, symbol, period)}")
    if str(record.get("status")) == "MISSING_SOURCE":
        return None
    if str(record.get("status")) not in VERIFIED_STATUSES:
        raise RuntimeError(f"source archive is not checksum verified: {record}")
    path = resolve_root_path(str(record["path"]))
    if not path.exists() or sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"source archive hash mismatch: {path}")
    checksum_path = path.with_suffix(path.suffix + ".CHECKSUM")
    if not checksum_path.exists():
        raise RuntimeError(f"source checksum file is missing: {checksum_path}")
    checksum = checksum_path.read_text(encoding="utf-8").split()
    if not checksum or checksum[0].lower() != str(record["sha256"]).lower():
        raise RuntimeError(f"source checksum payload mismatch: {checksum_path}")
    return path


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if str(config.get("version")) != "0.6.0":
        raise RuntimeError("real-agent context builder requires frozen v0.6.0 config")
    source = config["binance"]
    start = date.fromisoformat(str(source["start_date"]))
    end = date.fromisoformat(str(source["end_date"]))
    prior_end = date.fromisoformat(str(source["prior_inspected_period_end"]))
    if start <= prior_end:
        raise RuntimeError("real-agent v0.6 source overlaps the inspected v0.5 period")
    symbols = [str(value) for value in source["symbols"]]

    design_freeze_path = resolve_root_path(config["freeze"]["design_freeze_manifest"])
    design_freeze = json.loads(design_freeze_path.read_text(encoding="utf-8"))
    expected_config_hash = design_freeze["surface_hashes"]["configs/real_agent_v06.yaml"]
    if sha256(config_path) != expected_config_hash:
        raise RuntimeError("real-agent v0.6 config differs from the frozen design")

    source_manifest_path = resolve_root_path(source["source_manifest"])
    if not source_manifest_path.exists():
        raise FileNotFoundError("fetch the frozen real-agent v0.6 source first")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("outcome_metrics_computed") is not False:
        raise RuntimeError("source acquisition crossed the pre-result boundary")
    if source_manifest.get("config_sha256") != sha256(config_path):
        raise RuntimeError("source manifest was not generated from the frozen config")
    records = _source_records(source_manifest)

    # Every monthly archive for every frozen symbol is part of the source contract.
    for symbol in symbols:
        month = pd.Timestamp(start, tz="UTC").to_period("M")
        final_month = pd.Timestamp(end, tz="UTC").to_period("M")
        while month <= final_month:
            _verified_archive(records, "klines_1m", symbol, str(month))
            month += 1

    feature_access_path = resolve_root_path(config["freeze"]["feature_access"])
    feature_access = json.loads(feature_access_path.read_text(encoding="utf-8"))
    prompt_fields = [str(value) for value in feature_access["allowed_prompt_fields"]]
    forbidden_prompt_fields = set(feature_access["global_forbidden"])

    raw_dir = resolve_root_path(source["raw_dir"])
    kline_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    contexts: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    exclusions: dict[str, int] = {}
    valid_dates: list[date] = []
    for current in date_range(start, end):
        symbol = assigned_symbol_for_date(current, symbols)
        timestamp = pd.Timestamp(current, tz="UTC")
        month_period = f"{timestamp.year:04d}-{timestamp.month:02d}"
        month_path = _verified_archive(records, "klines_1m", symbol, month_period)
        depth_path = _verified_archive(records, "bookDepth", symbol, current.isoformat())
        if month_path is None or depth_path is None:
            exclusions["missing_source_archive"] = exclusions.get("missing_source_archive", 0) + 1
            continue
        expected_month_path = _month_path(raw_dir, symbol, timestamp)
        expected_depth_path = _depth_path(raw_dir, symbol, current)
        if (
            month_path.resolve() != expected_month_path.resolve()
            or depth_path.resolve() != expected_depth_path.resolve()
        ):
            raise RuntimeError("source manifest path violates the frozen directory layout")
        month_key = (symbol, timestamp.year, timestamp.month)
        if month_key not in kline_cache:
            kline_cache[month_key] = _load_klines(month_path)
        depth = _load_depth(depth_path)
        date_contexts, date_outcomes, reason = _build_date_contexts(
            source=source,
            tasks=config["tasks"],
            current=current,
            symbol=symbol,
            klines=kline_cache[month_key],
            depth=depth,
        )
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        if len(date_contexts) != 2 or len(date_outcomes) != 2:
            exclusions["incomplete_task_pair"] = exclusions.get("incomplete_task_pair", 0) + 1
            continue
        valid_dates.append(current)
        contexts.extend(date_contexts)
        outcomes.extend(date_outcomes)
        if len(valid_dates) == int(source["total_clusters"]):
            break

    if len(valid_dates) != int(source["total_clusters"]):
        raise RuntimeError(
            f"insufficient structurally valid dates: {len(valid_dates)} != "
            f"{source['total_clusters']}"
        )

    split_by_cluster: dict[str, str] = {}
    for index, current in enumerate(valid_dates):
        split_by_cluster[f"binance-{current.isoformat()}"] = _split_for_index(index, source)
    for outcome in outcomes:
        outcome["split"] = split_by_cluster[str(outcome["event_cluster_id"])]

    context_frame = pd.DataFrame(contexts)
    context_fields = [*prompt_fields, *sorted(CONTEXT_AUDIT_FIELDS)]
    if set(context_frame.columns) != set(context_fields):
        missing = sorted(set(context_fields) - set(context_frame.columns))
        extra = sorted(set(context_frame.columns) - set(context_fields))
        raise RuntimeError(f"prompt context schema mismatch: missing={missing} extra={extra}")
    context_frame = context_frame[context_fields].sort_values("context_id").reset_index(drop=True)
    leaked = forbidden_prompt_fields & set(context_frame.columns)
    if leaked:
        raise RuntimeError(f"forbidden outcome fields in prompt contexts: {sorted(leaked)}")

    outcome_frame = pd.DataFrame(outcomes).sort_values(
        ["event_cluster_id", "task_id"]
    ).reset_index(drop=True)
    registry = outcome_frame[
        [
            "event_cluster_id",
            "context_id",
            "task_id",
            "split",
            "symbol",
            "assigned_source_role",
        ]
    ].copy()

    derived_dir = resolve_root_path(source["derived_dir"])
    derived_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    cluster_splits: dict[str, int] = {}
    row_splits: dict[str, int] = {}
    for split in ("development", "paper_test", "community_hidden"):
        context_ids = set(registry.loc[registry["split"] == split, "context_id"])
        split_contexts = context_frame[context_frame["context_id"].isin(context_ids)]
        split_outcomes = outcome_frame[outcome_frame["split"] == split]
        context_path = derived_dir / f"{split}_contexts.csv"
        outcome_path = derived_dir / f"{split}_outcomes.parquet"
        split_contexts.to_csv(context_path, index=False)
        split_outcomes.to_parquet(outcome_path, index=False)
        outputs[str(context_path.relative_to(ROOT))] = sha256(context_path)
        outputs[str(outcome_path.relative_to(ROOT))] = sha256(outcome_path)
        cluster_splits[split] = int(split_outcomes["event_cluster_id"].nunique())
        row_splits[split] = int(len(split_contexts))

    registry_path = derived_dir / "split_registry.csv"
    registry.to_csv(registry_path, index=False)
    outputs[str(registry_path.relative_to(ROOT))] = sha256(registry_path)

    manifest_path = resolve_root_path(source["dataset_manifest"])
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.6.0",
        "source": source["source_name"],
        "contexts": int(len(context_frame)),
        "sealed_outcome_rows": int(len(outcome_frame)),
        "event_clusters": int(registry["event_cluster_id"].nunique()),
        "tasks_per_cluster": 2,
        "splits": cluster_splits,
        "context_rows_by_split": row_splits,
        "symbols_by_cluster": (
            registry.drop_duplicates("event_cluster_id")["symbol"]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        ),
        "tasks": registry["task_id"].value_counts().sort_index().astype(int).to_dict(),
        "assigned_source_roles": (
            registry["assigned_source_role"].value_counts().sort_index().astype(int).to_dict()
        ),
        "minimum_date": min(valid_dates).isoformat(),
        "maximum_date": max(valid_dates).isoformat(),
        "calendar_overlap_with_v05": False,
        "symbol_assignment": source["symbol_assignment"],
        "independent_cluster": source["independent_cluster"],
        "split_assignment": source["split_assignment"],
        "prompt_fields": prompt_fields,
        "context_audit_fields": sorted(CONTEXT_AUDIT_FIELDS),
        "prompt_payload_requires_allowlist_projection": True,
        "outcome_inputs_materialized_separately": True,
        "model_proposals_generated": False,
        "outcome_metrics_computed": False,
        "community_hidden_outcomes_evaluated": False,
        "exclusions": exclusions,
        "inputs": {
            str(config_path.relative_to(ROOT)): sha256(config_path),
            str(design_freeze_path.relative_to(ROOT)): sha256(design_freeze_path),
            str(source_manifest_path.relative_to(ROOT)): sha256(source_manifest_path),
            str(feature_access_path.relative_to(ROOT)): sha256(feature_access_path),
        },
        "outputs": outputs,
        "claim_boundary": (
            "Deterministic point-in-time context construction and split-specific "
            "sealed future inputs only. Prompt contexts contain only the frozen "
            "allowlist. No model proposal, harm label, utility, rule metric, rank, "
            "or outcome summary is computed."
        ),
    }
    write_json(manifest_path, manifest)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build frozen real-agent v0.6 prompt contexts and sealed outcomes."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "real_agent_v06.yaml")
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
