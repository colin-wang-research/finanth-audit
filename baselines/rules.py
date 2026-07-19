from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


DecisionFunction = Callable[[pd.DataFrame, dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class AuthorizationRule:
    name: str
    features_used: tuple[str, ...]
    decide_fn: DecisionFunction
    classification: str = "deployable"

    def decide(self, frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
        result = self.decide_fn(frame, thresholds)
        allowed = {"execute", "reduce", "review", "abstain"}
        if len(result) != len(frame) or set(np.unique(result)).difference(allowed):
            raise ValueError(f"{self.name} returned invalid decisions")
        return result


ELIGIBLE_ROLES = {"edge_proposer", "learned_prior", "policy_proposer"}


def _constant(value: str) -> DecisionFunction:
    return lambda frame, thresholds: np.full(len(frame), value, dtype=object)


def _direct(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    return np.full(len(frame), "execute", dtype=object)


def _confidence(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    return np.where(frame["confidence"].to_numpy() >= thresholds["confidence"], "execute", "abstain")


def _uncertainty(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    return np.where(frame["uncertainty"].to_numpy() <= thresholds["uncertainty"], "execute", "abstain")


def _risk(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    allowed = (total_cost <= thresholds["risk_cost_bps"]) & (
        frame["volatility_proxy"] <= thresholds["risk_volatility"]
    )
    return np.where(allowed, "execute", "abstain")


def _cost_aware(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    allowed = frame["expected_edge_bps"] > total_cost + thresholds["cost_margin_bps"]
    return np.where(allowed, "execute", "abstain")


def _hard_role(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    allowed = frame["source_role"].isin(ELIGIBLE_ROLES)
    return np.where(allowed, "execute", "abstain")


def _lifecycle(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    role_ok = frame["source_role"].isin(ELIGIBLE_ROLES)
    high_review_risk = (frame["uncertainty"] >= thresholds["lifecycle_review_uncertainty"]) | (
        frame["volatility_proxy"] >= thresholds["lifecycle_review_volatility"]
    )
    edge_after_cost = frame["expected_edge_bps"] - total_cost
    decision = np.full(len(frame), "abstain", dtype=object)
    decision[~role_ok.to_numpy()] = "review"
    decision[(role_ok & high_review_risk).to_numpy()] = "review"
    reduce_mask = role_ok & ~high_review_risk & (edge_after_cost > 0)
    decision[reduce_mask.to_numpy()] = "reduce"
    execute_mask = role_ok & ~high_review_risk & (
        edge_after_cost > thresholds["lifecycle_execute_margin_bps"]
    )
    decision[execute_mask.to_numpy()] = "execute"
    return decision


def phase1_rules() -> list[AuthorizationRule]:
    return [
        AuthorizationRule("No Action", (), _constant("abstain")),
        AuthorizationRule("Direct Prior", ("candidate_action",), _direct),
        AuthorizationRule("Confidence Gate", ("confidence",), _confidence),
        AuthorizationRule("Uncertainty Gate", ("uncertainty",), _uncertainty),
        AuthorizationRule(
            "Risk Filter",
            ("liquidity_cost_bps", "turnover_cost_bps", "fee_bps", "volatility_proxy"),
            _risk,
        ),
        AuthorizationRule(
            "Cost-Aware Gate",
            ("expected_edge_bps", "liquidity_cost_bps", "turnover_cost_bps", "fee_bps"),
            _cost_aware,
        ),
        AuthorizationRule("Hard Role Gate", ("source_role",), _hard_role),
        AuthorizationRule(
            "Lifecycle Checklist",
            (
                "source_role",
                "uncertainty",
                "volatility_proxy",
                "expected_edge_bps",
                "liquidity_cost_bps",
                "turnover_cost_bps",
                "fee_bps",
            ),
            _lifecycle,
        ),
    ]
