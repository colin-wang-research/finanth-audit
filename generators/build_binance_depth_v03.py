from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    assign_chronological_splits,
    date_range,
    frozen_prior_parameters,
    impact_bps,
    load_config,
    resolve_root_path,
    sha256,
    stress_tag,
    write_json,
)


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _load_klines(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != KLINE_COLUMNS:
        frame = pd.read_csv(path, names=KLINE_COLUMNS, header=None)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "open", "close"])
    return frame.set_index("open_time").sort_index()


def _load_depth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "percentage", "notional"}
    if not required.issubset(frame.columns):
        raise ValueError(f"invalid Binance bookDepth schema: {path}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["percentage"] = pd.to_numeric(frame["percentage"], errors="coerce")
    frame["notional"] = pd.to_numeric(frame["notional"], errors="coerce")
    frame = frame[frame["percentage"].isin([-1, 1])].dropna(
        subset=["timestamp", "notional"]
    )
    pivot = frame.pivot_table(
        index="timestamp", columns="percentage", values="notional", aggfunc="last"
    ).rename(columns={-1: "bid_depth_1pct", 1: "ask_depth_1pct"})
    return pivot.dropna().sort_index()


def _asof_row(frame: pd.DataFrame, timestamp: pd.Timestamp, max_age_seconds: int) -> pd.Series | None:
    position = frame.index.searchsorted(timestamp, side="right") - 1
    if position < 0:
        return None
    row = frame.iloc[int(position)].copy()
    observed = frame.index[int(position)]
    age = float((timestamp - observed).total_seconds())
    if age < 0 or age > max_age_seconds:
        return None
    row["_observed_timestamp"] = observed
    row["_age_seconds"] = age
    return row


def _exact_kline(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    if timestamp not in frame.index:
        return None
    row = frame.loc[timestamp]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


def _month_path(raw_dir: Path, symbol: str, timestamp: pd.Timestamp) -> Path:
    return raw_dir / "klines_1m" / symbol / f"{symbol}-1m-{timestamp.year:04d}-{timestamp.month:02d}.zip"


def _depth_path(raw_dir: Path, symbol: str, current: date) -> Path:
    return raw_dir / "bookDepth" / symbol / f"{symbol}-bookDepth-{current.isoformat()}.zip"


def _build_symbol_date(
    *,
    source: dict[str, Any],
    current: date,
    symbol: str,
    klines: pd.DataFrame,
    depth: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    action_notional = float(source["action_notional_usd"])
    slope = float(source["impact_slope_bps"])
    cap = float(source["impact_cap_bps"])
    fee_bps = float(source["roundtrip_fee_bps"])
    max_depth_age = int(source["max_depth_snapshot_age_seconds"])
    base_date = pd.Timestamp(current, tz="UTC")
    for hour in source["decision_hours_utc"]:
        decision_timestamp = base_date + pd.Timedelta(hours=int(hour))
        feature_start = decision_timestamp - pd.Timedelta(minutes=31)
        feature_end = decision_timestamp - pd.Timedelta(minutes=1)
        if feature_start.date() != current or feature_end.date() != current:
            return [], [], "feature_window_crosses_date_boundary"
        short_start = decision_timestamp - pd.Timedelta(minutes=6)
        action_timestamp = decision_timestamp + pd.Timedelta(
            minutes=int(source["action_delay_minutes"])
        )
        outcome_timestamp = action_timestamp + pd.Timedelta(
            minutes=int(source["holding_minutes"])
        )
        start_bar = _exact_kline(klines, feature_start)
        end_bar = _exact_kline(klines, feature_end)
        short_bar = _exact_kline(klines, short_start)
        entry_bar = _exact_kline(klines, action_timestamp)
        exit_bar = _exact_kline(klines, outcome_timestamp)
        if any(value is None for value in (start_bar, end_bar, short_bar, entry_bar, exit_bar)):
            return [], [], "missing_required_kline"
        trailing = klines.loc[feature_start:feature_end, "close"]
        if len(trailing) < 30 or (trailing <= 0).any():
            return [], [], "insufficient_feature_window"
        decision_depth = _asof_row(depth, decision_timestamp, max_depth_age)
        outcome_depth = _asof_row(depth, outcome_timestamp, max_depth_age)
        if decision_depth is None or outcome_depth is None:
            return [], [], "missing_or_stale_depth"
        bid_depth = float(decision_depth["bid_depth_1pct"])
        ask_depth = float(decision_depth["ask_depth_1pct"])
        exit_bid_depth = float(outcome_depth["bid_depth_1pct"])
        exit_ask_depth = float(outcome_depth["ask_depth_1pct"])
        if min(bid_depth, ask_depth, exit_bid_depth, exit_ask_depth) <= 0:
            return [], [], "nonpositive_depth"
        start_close = float(start_bar["close"])
        end_close = float(end_bar["close"])
        short_close = float(short_bar["close"])
        entry_open = float(entry_bar["open"])
        exit_open = float(exit_bar["open"])
        if min(start_close, end_close, short_close, entry_open, exit_open) <= 0:
            return [], [], "nonpositive_price"
        momentum_30 = (end_close / start_close - 1.0) * 10000.0
        momentum_5 = (end_close / short_close - 1.0) * 10000.0
        log_returns = np.diff(np.log(trailing.to_numpy(dtype=float)))
        volatility_bps = float(np.std(log_returns, ddof=1) * np.sqrt(30.0) * 10000.0)
        imbalance = float((bid_depth - ask_depth) / (bid_depth + ask_depth))
        bid_impact = impact_bps(action_notional, bid_depth, slope, cap)
        ask_impact = impact_bps(action_notional, ask_depth, slope, cap)
        thinness = min(1.0, max(bid_impact, ask_impact) / cap)
        volatility_proxy = float(np.clip(volatility_bps / 100.0, 0.0, 1.5))
        priors = frozen_prior_parameters(
            momentum_30_bps=momentum_30,
            momentum_5_bps=momentum_5,
            volatility_bps=volatility_bps,
            depth_imbalance=imbalance,
            thinness=thinness,
        )
        for prior in priors:
            entry_side_depth = ask_depth if prior.direction > 0 else bid_depth
            exit_side_depth = exit_bid_depth if prior.direction > 0 else exit_ask_depth
            entry_impact = impact_bps(action_notional, entry_side_depth, slope, cap)
            deployable_liquidity_cost = 2.0 * entry_impact
            cluster = f"binance-{current.isoformat()}"
            row_id = (
                f"{cluster}-{symbol}-{decision_timestamp.strftime('%H%M')}-{prior.name}"
            )
            rows.append(
                {
                    "row_id": row_id,
                    "event_cluster_id": cluster,
                    "source": "binance_public_depth",
                    "symbol": symbol,
                    "prior_family": prior.name,
                    "source_timestamp": feature_end.isoformat(),
                    "decision_timestamp": decision_timestamp.isoformat(),
                    "action_timestamp": action_timestamp.isoformat(),
                    "outcome_timestamp": outcome_timestamp.isoformat(),
                    "timestamp": decision_timestamp.isoformat(),
                    "candidate_action": prior.direction,
                    "confidence": prior.confidence,
                    "uncertainty": prior.uncertainty,
                    "expected_edge_bps": prior.expected_edge_bps,
                    "liquidity_cost_bps": deployable_liquidity_cost,
                    "turnover_cost_bps": 0.0,
                    "fee_bps": fee_bps,
                    "volatility_proxy": volatility_proxy,
                    "liquidity_proxy": entry_side_depth,
                    "horizon": int(source["holding_minutes"]),
                    "source_role": prior.source_role,
                    "claimed_role": prior.source_role,
                    "current_role_verified": True,
                    "original_source_eligible": prior.original_source_eligible,
                    "bid_depth_1pct": bid_depth,
                    "ask_depth_1pct": ask_depth,
                    "depth_imbalance": imbalance,
                    "depth_snapshot_timestamp": decision_depth["_observed_timestamp"].isoformat(),
                    "depth_snapshot_age_seconds": decision_depth["_age_seconds"],
                    "stress_tag": stress_tag(
                        prior.name, deployable_liquidity_cost, volatility_proxy
                    ),
                    "outcome_used_by_rule": False,
                }
            )
            outcome_rows.append(
                {
                    "row_id": row_id,
                    "event_cluster_id": cluster,
                    "entry_price": entry_open,
                    "exit_price": exit_open,
                    "entry_side_depth_1pct": entry_side_depth,
                    "exit_side_depth_1pct": exit_side_depth,
                    "outcome_depth_snapshot_timestamp": outcome_depth[
                        "_observed_timestamp"
                    ].isoformat(),
                    "outcome_depth_snapshot_age_seconds": outcome_depth[
                        "_age_seconds"
                    ],
                }
            )
    return rows, outcome_rows, None


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    source = config["binance"]
    source_manifest_path = resolve_root_path(source["source_manifest"])
    if not source_manifest_path.exists():
        raise FileNotFoundError("run fetch_binance_depth_v03 before building the dataset")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("outcome_metrics_computed") is not False:
        raise RuntimeError("source manifest violates the pre-result boundary")
    raw_dir = resolve_root_path(source["raw_dir"])
    start = date.fromisoformat(str(source["start_date"]))
    end = date.fromisoformat(str(source["end_date"]))
    cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    rows_by_date: dict[str, list[dict[str, object]]] = {}
    outcomes_by_date: dict[str, list[dict[str, object]]] = {}
    exclusions: dict[str, int] = {}
    for current in date_range(start, end):
        date_rows: list[dict[str, object]] = []
        date_outcomes: list[dict[str, object]] = []
        reason: str | None = None
        for symbol in source["symbols"]:
            timestamp = pd.Timestamp(current, tz="UTC")
            month_key = (str(symbol), timestamp.year, timestamp.month)
            month_path = _month_path(raw_dir, str(symbol), timestamp)
            depth_path = _depth_path(raw_dir, str(symbol), current)
            if not month_path.exists() or not depth_path.exists():
                reason = "missing_source_archive"
                break
            if month_key not in cache:
                cache[month_key] = _load_klines(month_path)
            depth = _load_depth(depth_path)
            symbol_rows, symbol_outcomes, reason = _build_symbol_date(
                source=source,
                current=current,
                symbol=str(symbol),
                klines=cache[month_key],
                depth=depth,
            )
            if reason is not None:
                break
            date_rows.extend(symbol_rows)
            date_outcomes.extend(symbol_outcomes)
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        expected = len(source["symbols"]) * len(source["decision_hours_utc"]) * len(
            config["prior_families"]
        )
        if len(date_rows) != expected:
            exclusions["incomplete_date"] = exclusions.get("incomplete_date", 0) + 1
            continue
        cluster = f"binance-{current.isoformat()}"
        rows_by_date[cluster] = date_rows
        outcomes_by_date[cluster] = date_outcomes
    split_map = assign_chronological_splits(
        rows_by_date,
        development=int(source["development_clusters"]),
        paper_test=int(source["paper_test_clusters"]),
    )
    all_rows: list[dict[str, object]] = []
    all_outcomes: list[dict[str, object]] = []
    for cluster in sorted(rows_by_date):
        for row in rows_by_date[cluster]:
            row["split"] = split_map[cluster]
            all_rows.append(row)
        for row in outcomes_by_date[cluster]:
            row["split"] = split_map[cluster]
            all_outcomes.append(row)
    frame = pd.DataFrame(all_rows).sort_values(
        ["event_cluster_id", "symbol", "decision_timestamp", "prior_family"]
    )
    outcomes = pd.DataFrame(all_outcomes).sort_values(["event_cluster_id", "row_id"])
    derived_dir = resolve_root_path(source["derived_dir"])
    derived_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for split in ("development", "paper_test", "community_hidden"):
        path = derived_dir / f"{split}.csv"
        frame[frame["split"] == split].to_csv(path, index=False)
        outputs[str(path.relative_to(ROOT))] = sha256(path)
        outcome_path = derived_dir / f"{split}_outcomes.parquet"
        outcomes[outcomes["split"] == split].to_parquet(outcome_path, index=False)
        outputs[str(outcome_path.relative_to(ROOT))] = sha256(outcome_path)
    split_registry = (
        pd.DataFrame(
            [{"event_cluster_id": cluster, "split": split} for cluster, split in split_map.items()]
        )
        .sort_values("event_cluster_id")
        .reset_index(drop=True)
    )
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
        "symbols": frame["symbol"].value_counts().sort_index().astype(int).to_dict(),
        "prior_families": frame["prior_family"].value_counts().sort_index().astype(int).to_dict(),
        "source_roles": frame["source_role"].value_counts().sort_index().astype(int).to_dict(),
        "stress_tags": frame["stress_tag"].value_counts().sort_index().astype(int).to_dict(),
        "date_min": split_registry["event_cluster_id"].min(),
        "date_max": split_registry["event_cluster_id"].max(),
        "exclusions": exclusions,
        "independent_cluster": source["independent_cluster"],
        "outcome_metrics_computed": False,
        "outcome_inputs_materialized_separately": True,
        "inputs": {
            str(config_path.relative_to(ROOT)): sha256(config_path),
            str(source_manifest_path.relative_to(ROOT)): sha256(source_manifest_path),
        },
        "outputs": outputs,
        "claim_boundary": (
            "Deterministic row construction and chronological cluster split only. "
            "No rule metric, rank, external effect size, or external power is computed."
        ),
    }
    write_json(manifest_path, manifest)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the preregistered Binance order-book layer.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "external_orderbook_v03.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
