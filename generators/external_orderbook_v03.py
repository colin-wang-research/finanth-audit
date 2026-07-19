from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PriorParameters:
    name: str
    direction: int
    confidence: float
    uncertainty: float
    expected_edge_bps: float
    source_role: str
    original_source_eligible: bool


PRIOR_ROLES: dict[str, tuple[str, bool]] = {
    "momentum_30m": ("edge_proposer", True),
    "reversal_30m": ("edge_proposer", True),
    "depth_imbalance": ("learned_prior", True),
    "volatility_breakout": ("policy_proposer", True),
    "spread_blind_high_conf": ("edge_proposer", True),
    "liquidity_critic": ("risk_critic", False),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.resolve().read_text(encoding="utf-8"))


def resolve_root_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def month_range(start: date, end: date) -> Iterable[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


def assign_chronological_splits(
    clusters: Iterable[str], development: int, paper_test: int
) -> dict[str, str]:
    ordered = sorted(set(str(value) for value in clusters))
    if len(ordered) < development + paper_test:
        raise ValueError(
            f"insufficient clusters: {len(ordered)} < {development + paper_test}"
        )
    result: dict[str, str] = {}
    for index, cluster in enumerate(ordered):
        if index < development:
            split = "development"
        elif index < development + paper_test:
            split = "paper_test"
        else:
            split = "community_hidden"
        result[cluster] = split
    return result


def impact_bps(
    action_notional: float,
    side_depth_notional: float,
    slope_bps: float = 50.0,
    cap_bps: float = 100.0,
) -> float:
    if not math.isfinite(side_depth_notional) or side_depth_notional <= 0:
        return float(cap_bps)
    return float(min(cap_bps, slope_bps * action_notional / side_depth_notional))


def direction(value: float, fallback: int = 1) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 1 if fallback >= 0 else -1


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def frozen_prior_parameters(
    *,
    momentum_30_bps: float,
    momentum_5_bps: float,
    volatility_bps: float,
    depth_imbalance: float,
    thinness: float,
) -> list[PriorParameters]:
    """Return the preregistered v0.3 weak-prior parameterization.

    The formulas are deliberately simple and are frozen before any external
    outcome metric is computed. They create heterogeneous weak priors rather
    than predictors optimized for either external source.
    """

    mom_strength = min(abs(momentum_30_bps) / 80.0, 1.0)
    short_strength = min(abs(momentum_5_bps) / 50.0, 1.0)
    volatility_norm = min(max(volatility_bps, 0.0) / 80.0, 1.5)
    imbalance_strength = min(abs(depth_imbalance), 1.0)
    thinness = _clip(thinness, 0.0, 1.0)

    raw = [
        (
            "momentum_30m",
            direction(momentum_30_bps),
            0.55 + 0.40 * mom_strength,
            0.15 + 0.35 * volatility_norm,
            max(0.5, min(100.0, abs(momentum_30_bps) * 0.35)),
        ),
        (
            "reversal_30m",
            -direction(momentum_30_bps),
            0.53 + 0.32 * mom_strength,
            0.25 + 0.40 * volatility_norm,
            max(0.5, min(100.0, abs(momentum_30_bps) * 0.25)),
        ),
        (
            "depth_imbalance",
            direction(depth_imbalance, fallback=direction(momentum_30_bps)),
            0.55 + 0.40 * imbalance_strength,
            0.15 + 0.65 * (1.0 - imbalance_strength),
            max(0.5, 35.0 * imbalance_strength),
        ),
        (
            "volatility_breakout",
            direction(momentum_5_bps, fallback=direction(momentum_30_bps)),
            0.55 + 0.35 * min(short_strength + 0.20 * volatility_norm, 1.0),
            0.20 + 0.45 * volatility_norm,
            max(0.5, min(100.0, abs(momentum_5_bps) * 0.30 + volatility_bps * 0.08)),
        ),
        (
            "spread_blind_high_conf",
            direction(momentum_30_bps),
            0.94,
            0.08,
            max(8.0, min(100.0, abs(momentum_30_bps) * 0.50)),
        ),
        (
            "liquidity_critic",
            direction(momentum_30_bps, fallback=direction(depth_imbalance)),
            0.85 + 0.10 * thinness,
            0.20 + 0.20 * thinness,
            max(5.0, min(100.0, abs(momentum_30_bps) * 0.20 + 20.0 * thinness)),
        ),
    ]
    priors: list[PriorParameters] = []
    for name, action, confidence, uncertainty, edge in raw:
        role, eligible = PRIOR_ROLES[name]
        priors.append(
            PriorParameters(
                name=name,
                direction=int(action),
                confidence=_clip(confidence, 0.50, 0.99),
                uncertainty=_clip(uncertainty, 0.0, 1.50),
                expected_edge_bps=float(edge),
                source_role=role,
                original_source_eligible=eligible,
            )
        )
    return priors


def stress_tag(
    prior_name: str,
    liquidity_cost_bps: float,
    volatility_proxy: float,
) -> str:
    if prior_name == "liquidity_critic":
        return "role_ineligible_source"
    if prior_name == "spread_blind_high_conf":
        return "cost_blind_high_confidence"
    if liquidity_cost_bps > 15.0:
        return "high_execution_friction"
    if volatility_proxy > 0.80:
        return "high_volatility"
    return "ordinary"
