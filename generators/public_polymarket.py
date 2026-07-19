from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(str(value))
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def _timestamp(value: object) -> float | None:
    if value is None or not str(value).strip():
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else float(parsed.timestamp())


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _fee_rate(market: dict[str, Any]) -> tuple[float | None, str]:
    if not bool(market.get("feesEnabled")):
        return 0.0, "market_metadata_fees_disabled"
    schedule = market.get("feeSchedule")
    if isinstance(schedule, dict) and schedule.get("rate") is not None:
        try:
            return float(schedule["rate"]), "market_metadata_fee_schedule"
        except (TypeError, ValueError):
            return None, "invalid_fee_schedule"
    return None, "missing_fee_schedule"


def _scheduled_end_timestamp(market: dict[str, Any]) -> float | None:
    scheduled = _timestamp(market.get("endDate"))
    if scheduled is not None:
        return scheduled
    events = market.get("events") or []
    if events and isinstance(events[0], dict):
        return _timestamp(events[0].get("endDate"))
    return None


def _stress_tag(
    prior_probability: float,
    uncertainty: float,
    history_points: int,
    time_to_resolution_hours: float,
    fee_rate: float,
) -> str:
    if fee_rate > 0 and 0.40 <= prior_probability <= 0.60:
        return "fee_peak_near_50"
    if time_to_resolution_hours <= 6.0:
        return "short_time_to_resolution"
    if uncertainty >= 0.10:
        return "high_historical_price_uncertainty"
    if history_points <= 5:
        return "thin_historical_price_record"
    if prior_probability <= 0.10 or prior_probability >= 0.90:
        return "extreme_probability"
    return "ordinary"


def build_row(
    market: dict[str, Any],
    history: list[dict[str, object]],
    config: dict[str, Any],
) -> tuple[dict[str, object] | None, str | None]:
    created = _timestamp(market.get("createdAt") or market.get("startDate"))
    closed = _timestamp(market.get("closedTime") or market.get("umaEndDate"))
    scheduled_end = _scheduled_end_timestamp(market)
    if created is None or closed is None or scheduled_end is None:
        return None, "missing_market_timestamp"
    if scheduled_end <= created:
        return None, "invalid_scheduled_lifecycle"
    decision_timestamp = created + float(config["decision_lifecycle_fraction"]) * (
        scheduled_end - created
    )
    if not created < decision_timestamp < closed - 60:
        return None, "fixed_decision_after_close"
    prices = []
    for value in _parse_json_list(market.get("outcomePrices")):
        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            return None, "invalid_resolution_price"
    if len(prices) < 2:
        return None, "missing_resolution_price"
    if prices[0] >= 0.99 and prices[1] <= 0.01:
        resolved_yes = 1
    elif prices[1] >= 0.99 and prices[0] <= 0.01:
        resolved_yes = 0
    else:
        return None, "ambiguous_resolution"
    fee_rate, fee_source = _fee_rate(market)
    if fee_rate is None:
        return None, fee_source

    points: list[tuple[int, float]] = []
    for item in history:
        try:
            timestamp = int(item["t"])
            probability = float(item["p"])
        except (KeyError, TypeError, ValueError):
            continue
        if created <= timestamp < closed - 60 and 0.005 <= probability <= 0.995:
            points.append((timestamp, probability))
    points = sorted(set(points))
    minimum = int(config["min_history_points"])
    if len(points) < minimum:
        return None, "insufficient_history"
    predecision = [(timestamp, value) for timestamp, value in points if timestamp < decision_timestamp]
    postdecision = [(timestamp, value) for timestamp, value in points if timestamp > decision_timestamp]
    if len(predecision) < minimum:
        return None, "insufficient_predecision_history"
    if not postdecision:
        return None, "missing_postdecision_execution"
    source_timestamp, prior_probability = predecision[-1]
    action_timestamp, execution_probability = postdecision[0]
    if not (source_timestamp < decision_timestamp < action_timestamp < closed):
        return None, "invalid_temporal_order"

    trailing = np.asarray([value for _, value in predecision], dtype=float)
    changes = np.diff(trailing)
    uncertainty = float(np.std(changes, ddof=1)) if len(changes) >= 2 else 0.0
    estimated_latency_cost_bps = (
        float(np.median(np.abs(changes))) * 10000.0 if len(changes) else 0.0
    )
    confidence = float(max(prior_probability, 1.0 - prior_probability))
    direction = 1 if prior_probability >= 0.5 else -1
    expected_edge_bps = float(abs(prior_probability - 0.5) * 20000.0)
    entry_price = execution_probability if direction > 0 else 1.0 - execution_probability
    payoff = float(resolved_yes if direction > 0 else 1 - resolved_yes)
    fee_per_share = float(fee_rate * entry_price * (1.0 - entry_price))
    gross_utility = payoff - entry_price
    full_utility = gross_utility - fee_per_share
    reduced_utility = 0.5 * gross_utility - 0.5 * fee_per_share
    estimated_fee_bps = float(fee_rate * prior_probability * (1.0 - prior_probability) * 10000.0)
    time_to_resolution_hours = (closed - decision_timestamp) / 3600.0
    stress_tag = _stress_tag(
        prior_probability,
        uncertainty,
        len(predecision),
        time_to_resolution_hours,
        fee_rate,
    )
    events = market.get("events") or []
    event_title = ""
    if events and isinstance(events[0], dict):
        event_title = str(events[0].get("title") or "")
    event_cluster_id = str(
        market.get("event_cluster_id")
        or f"polymarket-market-{market.get('id')}"
    )
    market_id = str(market.get("id"))
    return {
        "row_id": f"pit-{market_id}",
        "event_cluster_id": event_cluster_id,
        "source": "polymarket",
        "market_id": market_id,
        "condition_id": str(market.get("conditionId") or ""),
        "question": str(market.get("question") or ""),
        "event_title": event_title,
        "series_key": str(market.get("sampling_series_key") or "unclassified"),
        "sampling_month": str(market.get("sampling_month") or "unknown"),
        "market_created_timestamp": _iso(created),
        "source_timestamp": _iso(source_timestamp),
        "decision_timestamp": _iso(decision_timestamp),
        "action_timestamp": _iso(action_timestamp),
        "outcome_timestamp": _iso(closed),
        "timestamp": _iso(decision_timestamp),
        "split": "unassigned",
        "candidate_action": direction,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "expected_edge_bps": expected_edge_bps,
        "liquidity_cost_bps": estimated_latency_cost_bps,
        "turnover_cost_bps": 0.0,
        "fee_bps": estimated_fee_bps,
        "volatility_proxy": uncertainty,
        "liquidity_proxy": float(len(predecision)),
        "horizon": max(1, int(round(time_to_resolution_hours))),
        "source_role": "edge_proposer",
        "claimed_role": "edge_proposer",
        "current_role_verified": True,
        "original_source_eligible": True,
        "full_utility": full_utility,
        "reduced_utility": reduced_utility,
        "realized_return": gross_utility,
        "harm_label": full_utility < 0.0,
        "tail_loss": min(full_utility, 0.0),
        "resolved_outcome_yes": resolved_yes,
        "prior_probability": prior_probability,
        "execution_probability": execution_probability,
        "actual_fee_per_share": fee_per_share,
        "realized_latency_cost_bps": abs(execution_probability - prior_probability) * 10000.0,
        "time_to_resolution_hours": time_to_resolution_hours,
        "price_history_points_predecision": len(predecision),
        "historical_probability_source": str(
            next(
                (
                    item.get("source")
                    for item in history
                    if int(item.get("t", -1)) == source_timestamp
                ),
                "clob_prices_history",
            )
        ),
        "decision_policy": (
            f"fixed_lifecycle_fraction_{float(config['decision_lifecycle_fraction']):.2f}"
        ),
        "scheduled_end_timestamp": _iso(scheduled_end),
        "fee_rate": fee_rate,
        "fee_schedule_source": fee_source,
        "stress_tag": stress_tag,
        "historical_probability_observed": True,
        "historical_spread_observed": False,
        "historical_depth_observed": False,
        "resolution_text_used_by_rule": False,
        "outcome_used_by_rule": False,
        "future_decoy": 0.0,
    }, None


def assign_splits(frame: pd.DataFrame, fractions: dict[str, float]) -> pd.DataFrame:
    ordered = frame.sort_values(["outcome_timestamp", "event_cluster_id"]).reset_index(drop=True)
    clusters = ordered[["event_cluster_id", "outcome_timestamp"]].drop_duplicates()
    clusters = clusters.sort_values(["outcome_timestamp", "event_cluster_id"]).reset_index(drop=True)
    n = len(clusters)
    train_end = int(round(float(fractions["train"]) * n))
    validation_end = train_end + int(round(float(fractions["validation"]) * n))
    split_map: dict[str, str] = {}
    for index, cluster in enumerate(clusters["event_cluster_id"]):
        split_map[str(cluster)] = (
            "train" if index < train_end else "validation" if index < validation_end else "test"
        )
    ordered["split"] = ordered["event_cluster_id"].map(split_map)
    return ordered


def evenly_spaced_rows(
    rows: list[dict[str, object]], target: int
) -> list[dict[str, object]]:
    if len(rows) <= target:
        return rows
    indices = {
        int(round(position * (len(rows) - 1) / (target - 1)))
        for position in range(target)
    }
    return [rows[index] for index in sorted(indices)]


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = payload["polymarket"]
    metadata_path = ROOT / config["metadata_path"]
    history_path = ROOT / config["history_path"]
    if not metadata_path.exists() or not history_path.exists():
        raise FileNotFoundError("run fetch_polymarket_point_in_time before building the dataset")
    markets = json.loads(metadata_path.read_text(encoding="utf-8"))
    with gzip.open(history_path, "rt", encoding="utf-8") as handle:
        histories = json.load(handle)

    rows: list[dict[str, object]] = []
    exclusions: dict[str, int] = {}
    for market in markets:
        tokens = [str(value) for value in _parse_json_list(market.get("clobTokenIds"))]
        history = histories.get(tokens[0], []) if tokens else []
        row, reason = build_row(market, history, config)
        if row is None:
            exclusions[str(reason)] = exclusions.get(str(reason), 0) + 1
            continue
        rows.append(row)
    rows = sorted(rows, key=lambda item: (str(item["outcome_timestamp"]), str(item["event_cluster_id"])))
    target = int(config["target_event_clusters"])
    if len(rows) < target:
        raise RuntimeError(f"only {len(rows)} valid event clusters; target={target}; exclusions={exclusions}")
    selected_rows = evenly_spaced_rows(rows, target)
    frame = assign_splits(pd.DataFrame(selected_rows), payload["split_fractions"])
    output_path = ROOT / config["dataset_path"]
    manifest_path = ROOT / config["dataset_manifest"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    series_counts = frame["series_key"].value_counts()
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0",
        "source": "Polymarket public point-in-time CLOB history and Data API trades",
        "rows": len(frame),
        "event_clusters": int(frame["event_cluster_id"].nunique()),
        "rows_per_cluster_max": int(frame.groupby("event_cluster_id").size().max()),
        "splits": frame["split"].value_counts().sort_index().astype(int).to_dict(),
        "clusters_by_split": frame.groupby("split")["event_cluster_id"].nunique().astype(int).to_dict(),
        "stress_tags": frame["stress_tag"].value_counts().sort_index().astype(int).to_dict(),
        "history_sources": frame["historical_probability_source"].value_counts().sort_index().astype(int).to_dict(),
        "decision_policies": frame["decision_policy"].value_counts().sort_index().astype(int).to_dict(),
        "distinct_series": int(frame["series_key"].nunique()),
        "top_series": {
            str(key): int(value) for key, value in series_counts.head(10).items()
        },
        "max_series_share": float(series_counts.max() / len(frame)),
        "outcome_time_min": str(frame["outcome_timestamp"].min()),
        "outcome_time_max": str(frame["outcome_timestamp"].max()),
        "excluded_candidates": exclusions,
        "historical_fields": ["prior_probability", "source_timestamp", "action_timestamp"],
        "historical_fields_unavailable": ["spread", "depth", "order_book"],
        "outcome_only_fields": [
            "resolved_outcome_yes",
            "execution_probability",
            "actual_fee_per_share",
            "realized_latency_cost_bps",
            "full_utility",
            "reduced_utility",
            "harm_label",
        ],
        "outputs": {
            str(output_path.relative_to(ROOT)): sha256(output_path),
        },
        "inputs": {
            str(metadata_path.relative_to(ROOT)): sha256(metadata_path),
            str(history_path.relative_to(ROOT)): sha256(history_path),
            str(config_path.relative_to(ROOT)): sha256(config_path),
        },
        "claim_boundary": (
            "Point-in-time public replay with observed historical probabilities and fee-adjusted "
            "resolution utility. Historical spread and depth are unavailable and omitted. This is "
            "offline replay, not deployment evidence."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_path)
    print(manifest_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Polymarket point-in-time public replay.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "public_audit.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
