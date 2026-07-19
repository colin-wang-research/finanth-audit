from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from finauth_audit.baselines.rules import AuthorizationRule, ELIGIBLE_ROLES


def _constant(value: str):
    return lambda frame, thresholds: np.full(len(frame), value, dtype=object)


def _no_role(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    return np.full(len(frame), "execute", dtype=object)


def _shared(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    return np.where(frame["confidence"] >= thresholds["shared_confidence"], "execute", "abstain")


def _soft(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    claimed_ok = frame["claimed_role"].isin(ELIGIBLE_ROLES)
    score = frame["confidence"] - np.where(
        claimed_ok & frame["current_role_verified"], 0.0, thresholds["soft_penalty"]
    )
    return np.where(score >= thresholds["soft_threshold"], "execute", "abstain")


def _hard_role(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    role_ok = frame["current_role_verified"] & frame["verified_current_role"].isin(ELIGIBLE_ROLES)
    return np.where(role_ok, "execute", "review")


def _lineage_roles(value: str) -> list[str]:
    return [part for part in str(value).split("|") if part]


def _provenance_hard(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    decisions = np.full(len(frame), "review", dtype=object)
    current_ok = frame["current_role_verified"] & frame["verified_current_role"].isin(ELIGIBLE_ROLES)
    for position, (_, row) in enumerate(frame.iterrows()):
        if not bool(current_ok.iloc[position]):
            decisions[position] = "abstain"
            continue
        if not bool(row["lineage_attested"]):
            decisions[position] = "review"
            continue
        roles = _lineage_roles(row["lineage_role_chain"])
        decisions[position] = "execute" if roles and all(role in ELIGIBLE_ROLES for role in roles) else "abstain"
    return decisions


def _lifecycle(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    decisions = np.full(len(frame), "abstain", dtype=object)
    role_ok = frame["current_role_verified"] & frame["verified_current_role"].isin(ELIGIBLE_ROLES)
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    margin = frame["expected_edge_bps"] - total_cost
    decisions[(~role_ok).to_numpy()] = "review"
    decisions[(role_ok & (frame["uncertainty"] > 0.58)).to_numpy()] = "review"
    decisions[(role_ok & (frame["uncertainty"] <= 0.58) & (margin > 0)).to_numpy()] = "reduce"
    decisions[
        (
            role_ok
            & (frame["uncertainty"] <= 0.58)
            & (margin > thresholds["lifecycle_execute_margin_bps"])
        ).to_numpy()
    ] = "execute"
    return decisions


def _epv_adapter(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    """Frozen equal-information EPV comparator; no provenance or outcome access."""
    decisions = np.full(len(frame), "abstain", dtype=object)
    role_ok = frame["current_role_verified"] & frame["verified_current_role"].isin(ELIGIBLE_ROLES)
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    conservative_margin = frame["expected_edge_bps"] - total_cost - 10.0 * frame["uncertainty"]
    reduce = role_ok & (conservative_margin > 0) & (
        frame["uncertainty"] <= thresholds["epv_uncertainty_max"]
    )
    execute = reduce & (conservative_margin > thresholds["epv_margin_bps"])
    decisions[reduce.to_numpy()] = "reduce"
    decisions[execute.to_numpy()] = "execute"
    return decisions


@dataclass
class ProvenanceLearnedRule:
    execute_threshold: float
    review_threshold: float
    random_state: int
    name: str = "Provenance Learned Gate"
    classification: str = "deployable"
    features_used: tuple[str, ...] = (
        "confidence",
        "uncertainty",
        "expected_edge_bps",
        "liquidity_cost_bps",
        "turnover_cost_bps",
        "fee_bps",
        "hop_depth",
        "current_role_verified",
        "lineage_attested",
        "claimed_role",
        "verified_current_role",
        "transformation_type",
        "traceability",
    )
    _model: Pipeline | None = None

    def fit(self, frame: pd.DataFrame) -> "ProvenanceLearnedRule":
        numeric = [
            "confidence",
            "uncertainty",
            "expected_edge_bps",
            "liquidity_cost_bps",
            "turnover_cost_bps",
            "fee_bps",
            "hop_depth",
        ]
        categorical = [
            "current_role_verified",
            "lineage_attested",
            "claimed_role",
            "verified_current_role",
            "transformation_type",
            "traceability",
        ]
        target = (
            frame["current_source_eligible"]
            & ~frame["authority_laundering"]
            & ~frame["direct_leakage"]
        ).astype(int)
        preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        self._model = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self._model.fit(frame[list(self.features_used)], target)
        return self

    def decide(self, frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit ProvenanceLearnedRule before decide")
        probability = self._model.predict_proba(frame[list(self.features_used)])[:, 1]
        decisions = np.full(len(frame), "abstain", dtype=object)
        decisions[probability >= self.review_threshold] = "review"
        decisions[probability >= self.execute_threshold] = "execute"
        return decisions

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "execute_threshold": self.execute_threshold,
            "review_threshold": self.review_threshold,
            "random_state": self.random_state,
            "features_used": list(self.features_used),
            "target": "current source eligible and no direct/indirect authority laundering",
            "claim_boundary": "Training labels are used only in the training split; decision-time inputs follow the legal feature manifest.",
        }


def provenance_rules(
    thresholds: dict[str, float], train: pd.DataFrame, model_seed: int
) -> list[object]:
    learned = ProvenanceLearnedRule(
        execute_threshold=float(thresholds["provenance_learned_execute"]),
        review_threshold=float(thresholds["provenance_learned_review"]),
        random_state=model_seed,
    ).fit(train)
    return [
        AuthorizationRule("No Action", (), _constant("abstain")),
        AuthorizationRule("No Role Gate", ("candidate_action",), _no_role),
        AuthorizationRule("Shared Threshold", ("confidence",), _shared),
        AuthorizationRule(
            "Soft Penalty",
            ("confidence", "claimed_role", "current_role_verified"),
            _soft,
        ),
        AuthorizationRule(
            "Hard Role Gate",
            ("current_role_verified", "verified_current_role"),
            _hard_role,
        ),
        AuthorizationRule(
            "Provenance Hard Gate",
            (
                "current_role_verified",
                "verified_current_role",
                "lineage_attested",
                "lineage_role_chain",
            ),
            _provenance_hard,
        ),
        learned,
        AuthorizationRule(
            "Lifecycle Checklist",
            (
                "current_role_verified",
                "verified_current_role",
                "uncertainty",
                "expected_edge_bps",
                "liquidity_cost_bps",
                "turnover_cost_bps",
                "fee_bps",
            ),
            _lifecycle,
        ),
        AuthorizationRule(
            "EPV Adapter",
            (
                "current_role_verified",
                "verified_current_role",
                "uncertainty",
                "expected_edge_bps",
                "liquidity_cost_bps",
                "turnover_cost_bps",
                "fee_bps",
            ),
            _epv_adapter,
        ),
    ]
