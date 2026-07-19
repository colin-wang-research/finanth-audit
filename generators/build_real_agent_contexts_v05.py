from __future__ import annotations

import argparse
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


def _build_context(
    *,
    source: dict[str, Any],
    current: date,
    symbol: str,
    klines: pd.DataFrame,
    depth: pd.DataFrame,
) -> tuple[dict[str, object] | None, dict[str, object] | None, str | None]:
    decision = pd.Timestamp(current, tz="UTC") + pd.Timedelta(
        hours=int(source["decision_hour_utc"])
    )
    feature_start = decision - pd.Timedelta(minutes=31)
    feature_end = decision - pd.Timedelta(minutes=1)
    short_start = decision - pd.Timedelta(minutes=6)
    action = decision + pd.Timedelta(minutes=int(source["action_delay_minutes"]))
    outcome = action + pd.Timedelta(minutes=int(source["holding_minutes"]))
    required = [
        _exact_kline(klines, feature_start),
        _exact_kline(klines, feature_end),
        _exact_kline(klines, short_start),
        _exact_kline(klines, action),
        _exact_kline(klines, outcome),
    ]
    if any(value is None for value in required):
        return None, None, "missing_required_kline"
    start_bar, end_bar, short_bar, entry_bar, exit_bar = required
    trailing = klines.loc[feature_start:feature_end, "close"]
    if len(trailing) < 30 or (trailing <= 0).any():
        return None, None, "insufficient_feature_window"
    max_age = int(source["max_depth_snapshot_age_seconds"])
    decision_depth = _asof_row(depth, decision, max_age)
    outcome_depth = _asof_row(depth, outcome, max_age)
    if decision_depth is None or outcome_depth is None:
        return None, None, "missing_or_stale_depth"
    bid_depth = float(decision_depth["bid_depth_1pct"])
    ask_depth = float(decision_depth["ask_depth_1pct"])
    exit_bid_depth = float(outcome_depth["bid_depth_1pct"])
    exit_ask_depth = float(outcome_depth["ask_depth_1pct"])
    entry_price = float(entry_bar["open"])
    exit_price = float(exit_bar["open"])
    prices = [
        float(start_bar["close"]),
        float(end_bar["close"]),
        float(short_bar["close"]),
        entry_price,
        exit_price,
    ]
    if min(prices + [bid_depth, ask_depth, exit_bid_depth, exit_ask_depth]) <= 0:
        return None, None, "nonpositive_market_value"
    momentum_30 = (prices[1] / prices[0] - 1.0) * 10000.0
    momentum_5 = (prices[1] / prices[2] - 1.0) * 10000.0
    log_returns = np.diff(np.log(trailing.to_numpy(dtype=float)))
    volatility_bps = float(np.std(log_returns, ddof=1) * np.sqrt(30.0) * 10000.0)
    imbalance = float((bid_depth - ask_depth) / (bid_depth + ask_depth))
    notional = float(source["action_notional_usd"])
    slope = float(source["impact_slope_bps"])
    cap = float(source["impact_cap_bps"])
    long_cost = 2.0 * impact_bps(notional, ask_depth, slope, cap)
    short_cost = 2.0 * impact_bps(notional, bid_depth, slope, cap)
    context_id = f"real-agent-{current.isoformat()}"
    cluster_id = f"binance-{current.isoformat()}"
    context = {
        "context_id": context_id,
        "event_cluster_id": cluster_id,
        "source": "binance_public_depth",
        "symbol": symbol,
        "source_timestamp": feature_end.isoformat(),
        "decision_timestamp": decision.isoformat(),
        "action_timestamp": action.isoformat(),
        "momentum_30_bps": momentum_30,
        "momentum_5_bps": momentum_5,
        "volatility_bps": volatility_bps,
        "volatility_proxy": float(np.clip(volatility_bps / 100.0, 0.0, 1.5)),
        "bid_depth_1pct": bid_depth,
        "ask_depth_1pct": ask_depth,
        "depth_imbalance": imbalance,
        "depth_snapshot_timestamp": decision_depth["_observed_timestamp"].isoformat(),
        "depth_snapshot_age_seconds": float(decision_depth["_age_seconds"]),
        "estimated_long_liquidity_cost_bps": long_cost,
        "estimated_short_liquidity_cost_bps": short_cost,
        "roundtrip_fee_bps": float(source["roundtrip_fee_bps"]),
        "holding_minutes": int(source["holding_minutes"]),
    }
    sealed = {
        "context_id": context_id,
        "event_cluster_id": cluster_id,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_bid_depth_1pct": bid_depth,
        "entry_ask_depth_1pct": ask_depth,
        "exit_bid_depth_1pct": exit_bid_depth,
        "exit_ask_depth_1pct": exit_ask_depth,
        "outcome_timestamp": outcome.isoformat(),
        "outcome_depth_snapshot_timestamp": outcome_depth[
            "_observed_timestamp"
        ].isoformat(),
        "outcome_depth_snapshot_age_seconds": float(outcome_depth["_age_seconds"]),
    }
    return context, sealed, None


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if str(config.get("version")) != "0.5.0":
        raise RuntimeError("real-agent context builder requires v0.5.0")
    source = config["binance"]
    if str(source["start_date"]) <= str(source["prior_inspected_period_end"]):
        raise RuntimeError("real-agent source overlaps the inspected v0.3 period")
    source_manifest_path = resolve_root_path(source["source_manifest"])
    if not source_manifest_path.exists():
        raise FileNotFoundError("fetch the registered Binance source before building contexts")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("outcome_metrics_computed") is not False:
        raise RuntimeError("source acquisition crossed the pre-result boundary")
    raw_dir = resolve_root_path(source["raw_dir"])
    start = date.fromisoformat(str(source["start_date"]))
    end = date.fromisoformat(str(source["end_date"]))
    symbols = [str(value) for value in source["symbols"]]
    kline_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    contexts: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    exclusions: dict[str, int] = {}
    for current in date_range(start, end):
        symbol = symbols[len(contexts) % len(symbols)]
        timestamp = pd.Timestamp(current, tz="UTC")
        month_key = (symbol, timestamp.year, timestamp.month)
        month_path = _month_path(raw_dir, symbol, timestamp)
        depth_path = _depth_path(raw_dir, symbol, current)
        if not month_path.exists() or not depth_path.exists():
            exclusions["missing_source_archive"] = exclusions.get("missing_source_archive", 0) + 1
            continue
        if month_key not in kline_cache:
            kline_cache[month_key] = _load_klines(month_path)
        depth = _load_depth(depth_path)
        context, sealed, reason = _build_context(
            source=source,
            current=current,
            symbol=symbol,
            klines=kline_cache[month_key],
            depth=depth,
        )
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        contexts.append(context or {})
        outcomes.append(sealed or {})
        if len(contexts) == int(source["total_clusters"]):
            break
    if len(contexts) != int(source["total_clusters"]):
        raise RuntimeError(
            f"insufficient structurally valid dates: {len(contexts)} != {source['total_clusters']}"
        )
    development = int(source["development_clusters"])
    paper_test = int(source["paper_test_clusters"])
    for index, (context, sealed) in enumerate(zip(contexts, outcomes, strict=True)):
        if index < development:
            split = "development"
        elif index < development + paper_test:
            split = "paper_test"
        else:
            split = "community_hidden"
        context["split"] = split
        sealed["split"] = split
    context_frame = pd.DataFrame(contexts).sort_values("event_cluster_id")
    outcome_frame = pd.DataFrame(outcomes).sort_values("event_cluster_id")
    feature_access_path = resolve_root_path(config["freeze"]["feature_access"])
    feature_access = json.loads(feature_access_path.read_text(encoding="utf-8"))
    forbidden = set(feature_access["global_forbidden"]) & set(context_frame.columns)
    if forbidden:
        raise RuntimeError(f"forbidden outcome fields in contexts: {sorted(forbidden)}")
    if "outcome_timestamp" in context_frame.columns:
        raise RuntimeError("outcome_timestamp leaked into prompt contexts")
    derived_dir = resolve_root_path(source["derived_dir"])
    derived_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for split in ("development", "paper_test", "community_hidden"):
        context_path = derived_dir / f"{split}_contexts.csv"
        outcome_path = derived_dir / f"{split}_outcomes.parquet"
        context_frame[context_frame["split"] == split].to_csv(context_path, index=False)
        outcome_frame[outcome_frame["split"] == split].to_parquet(outcome_path, index=False)
        outputs[str(context_path.relative_to(ROOT))] = sha256(context_path)
        outputs[str(outcome_path.relative_to(ROOT))] = sha256(outcome_path)
    registry = context_frame[["event_cluster_id", "context_id", "split", "symbol"]]
    registry_path = derived_dir / "split_registry.csv"
    registry.to_csv(registry_path, index=False)
    outputs[str(registry_path.relative_to(ROOT))] = sha256(registry_path)
    manifest_path = resolve_root_path(source["dataset_manifest"])
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.5.0",
        "source": source["source_name"],
        "contexts": len(context_frame),
        "event_clusters": int(context_frame["event_cluster_id"].nunique()),
        "splits": context_frame["split"].value_counts().sort_index().astype(int).to_dict(),
        "symbols": context_frame["symbol"].value_counts().sort_index().astype(int).to_dict(),
        "minimum_date": context_frame["event_cluster_id"].min(),
        "maximum_date": context_frame["event_cluster_id"].max(),
        "calendar_overlap_with_v03": False,
        "independent_cluster": source["independent_cluster"],
        "split_assignment": source["split_assignment"],
        "exclusions": exclusions,
        "outcome_metrics_computed": False,
        "outcome_inputs_materialized_separately": True,
        "community_hidden_outcomes_evaluated": False,
        "inputs": {
            str(config_path.relative_to(ROOT)): sha256(config_path),
            str(source_manifest_path.relative_to(ROOT)): sha256(source_manifest_path),
            str(feature_access_path.relative_to(ROOT)): sha256(feature_access_path),
        },
        "outputs": outputs,
        "claim_boundary": (
            "Point-in-time context construction and sealed outcome inputs only; "
            "no model proposal, rule metric, harm label, or outcome summary is computed."
        ),
    }
    write_json(manifest_path, manifest)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real-agent v0.5 contexts and sealed outcomes.")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "real_agent_v05.yaml")
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
