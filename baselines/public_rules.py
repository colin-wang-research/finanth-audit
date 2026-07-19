from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from finauth_audit.baselines.rules import AuthorizationRule, ELIGIBLE_ROLES


def _fixed(
    name: str,
    features: tuple[str, ...],
    function: Callable[[pd.DataFrame], np.ndarray],
    *,
    classification: str = "deployable",
) -> AuthorizationRule:
    return AuthorizationRule(
        name=name,
        features_used=features,
        decide_fn=lambda frame, thresholds: function(frame),
        classification=classification,
    )


def _margin(frame: pd.DataFrame) -> pd.Series:
    return frame["expected_edge_bps"] - (
        frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    )


def public_rule_registry(config: dict[str, Any]) -> list[AuthorizationRule]:
    registry = config["rule_registry"]
    rules: list[AuthorizationRule] = [
        _fixed(
            "No Action",
            (),
            lambda frame: np.full(len(frame), "abstain", dtype=object),
            classification="diagnostic",
        ),
        _fixed(
            "Direct Prior",
            ("candidate_action",),
            lambda frame: np.full(len(frame), "execute", dtype=object),
        ),
    ]
    for threshold in registry["confidence_thresholds"]:
        value = float(threshold)
        rules.append(
            _fixed(
                f"Confidence Gate [{value:.2f}]",
                ("confidence",),
                lambda frame, value=value: np.where(
                    frame["confidence"].to_numpy() >= value, "execute", "abstain"
                ),
            )
        )
    for threshold in registry["uncertainty_thresholds"]:
        value = float(threshold)
        rules.append(
            _fixed(
                f"Uncertainty Gate [{value:.2f}]",
                ("uncertainty",),
                lambda frame, value=value: np.where(
                    frame["uncertainty"].to_numpy() <= value, "execute", "abstain"
                ),
            )
        )
    for threshold in registry["edge_thresholds_bps"]:
        value = float(threshold)
        rules.append(
            _fixed(
                f"Edge Gate [{value:.0f} bps]",
                ("expected_edge_bps",),
                lambda frame, value=value: np.where(
                    frame["expected_edge_bps"].to_numpy() >= value, "execute", "abstain"
                ),
            )
        )
    cost_features = (
        "expected_edge_bps",
        "liquidity_cost_bps",
        "turnover_cost_bps",
        "fee_bps",
    )
    for threshold in registry["cost_margin_thresholds_bps"]:
        value = float(threshold)
        rules.append(
            _fixed(
                f"Cost-Aware Gate [{value:.0f} bps]",
                cost_features,
                lambda frame, value=value: np.where(_margin(frame) >= value, "execute", "abstain"),
            )
        )
    for threshold in registry["fee_thresholds_bps"]:
        value = float(threshold)
        rules.append(
            _fixed(
                f"Fee Gate [{value:.0f} bps]",
                ("fee_bps",),
                lambda frame, value=value: np.where(
                    frame["fee_bps"].to_numpy() <= value, "execute", "abstain"
                ),
            )
        )
    for profile in registry["risk_profiles"]:
        cost = float(profile["cost_bps"])
        volatility = float(profile["volatility"])
        rules.append(
            _fixed(
                f"Risk Filter [cost={cost:.0f}, vol={volatility:.2f}]",
                (
                    "liquidity_cost_bps",
                    "turnover_cost_bps",
                    "fee_bps",
                    "volatility_proxy",
                ),
                lambda frame, cost=cost, volatility=volatility: np.where(
                    (
                        frame["liquidity_cost_bps"]
                        + frame["turnover_cost_bps"]
                        + frame["fee_bps"]
                    )
                    <= cost,
                    np.where(frame["volatility_proxy"] <= volatility, "execute", "abstain"),
                    "abstain",
                ),
            )
        )
    for threshold in registry["horizon_thresholds_hours"]:
        value = int(threshold)
        rules.append(
            _fixed(
                f"Horizon Gate [{value} h]",
                ("horizon",),
                lambda frame, value=value: np.where(
                    frame["horizon"].to_numpy() >= value, "execute", "abstain"
                ),
            )
        )
    rules.append(
        _fixed(
            "Hard Role Gate",
            ("source_role",),
            lambda frame: np.where(frame["source_role"].isin(ELIGIBLE_ROLES), "execute", "abstain"),
        )
    )
    lifecycle_features = (
        "source_role",
        "uncertainty",
        "volatility_proxy",
        "expected_edge_bps",
        "liquidity_cost_bps",
        "turnover_cost_bps",
        "fee_bps",
    )
    for profile in registry["lifecycle_profiles"]:
        name = str(profile["name"])
        execute_margin = float(profile["execute_margin_bps"])
        review_uncertainty = float(profile["review_uncertainty"])
        review_volatility = float(profile["review_volatility"])

        def lifecycle(
            frame: pd.DataFrame,
            execute_margin: float = execute_margin,
            review_uncertainty: float = review_uncertainty,
            review_volatility: float = review_volatility,
        ) -> np.ndarray:
            role_ok = frame["source_role"].isin(ELIGIBLE_ROLES)
            review = (frame["uncertainty"] >= review_uncertainty) | (
                frame["volatility_proxy"] >= review_volatility
            )
            decision = np.full(len(frame), "abstain", dtype=object)
            decision[(~role_ok | review).to_numpy()] = "review"
            reduce = role_ok & ~review & (_margin(frame) > 0)
            execute = role_ok & ~review & (_margin(frame) >= execute_margin)
            decision[reduce.to_numpy()] = "reduce"
            decision[execute.to_numpy()] = "execute"
            return decision

        rules.append(_fixed(f"Lifecycle Checklist [{name}]", lifecycle_features, lifecycle))

    epv_margin = float(registry["epv_margin_bps"])
    epv_uncertainty = float(registry["epv_uncertainty_max"])

    def epv_adapter(frame: pd.DataFrame) -> np.ndarray:
        role_ok = frame["source_role"].isin(ELIGIBLE_ROLES)
        conservative_margin = _margin(frame) - 1000.0 * frame["uncertainty"]
        decision = np.full(len(frame), "abstain", dtype=object)
        reduce = role_ok & (frame["uncertainty"] <= epv_uncertainty) & (conservative_margin > 0)
        execute = reduce & (conservative_margin >= epv_margin)
        decision[reduce.to_numpy()] = "reduce"
        decision[execute.to_numpy()] = "execute"
        return decision

    rules.append(_fixed("EPV Adapter", lifecycle_features, epv_adapter))
    return rules
