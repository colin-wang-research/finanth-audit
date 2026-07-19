from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    impact_bps,
    resolve_root_path,
    sha256,
)


AUTHORIZED_DECISIONS = {"execute", "reduce"}


def attach_registered_context_fields(
    proposals: pd.DataFrame,
    contexts: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    required_proposal = {
        "context_id",
        "event_cluster_id",
        "task_id",
        "symbol",
        "split",
        "assigned_source_role",
    }
    required_context = {
        "context_id",
        "task_id",
        "symbol",
        "assigned_source_role",
        "decision_timestamp",
        "volatility_bps",
    }
    required_registry = {
        "context_id",
        "event_cluster_id",
        "task_id",
        "symbol",
        "split",
        "assigned_source_role",
    }
    missing_proposal = required_proposal.difference(proposals.columns)
    missing_context = required_context.difference(contexts.columns)
    missing_registry = required_registry.difference(registry.columns)
    if missing_proposal:
        raise ValueError(
            f"registered context attachment is missing proposal fields: {sorted(missing_proposal)}"
        )
    if missing_context:
        raise ValueError(
            f"registered context attachment is missing context fields: {sorted(missing_context)}"
        )
    if missing_registry:
        raise ValueError(
            "registered context attachment is missing registry fields: "
            f"{sorted(missing_registry)}"
        )
    if contexts["context_id"].duplicated().any():
        raise ValueError("registered context attachment has duplicate context_id values")
    if registry["context_id"].duplicated().any():
        raise ValueError("registered split registry has duplicate context_id values")
    for field in ("decision_timestamp", "volatility_bps"):
        if field in proposals.columns:
            raise ValueError(f"proposal cache unexpectedly already contains {field!r}")

    context_projection = contexts[sorted(required_context)].rename(
        columns={
            "task_id": "context_task_id",
            "symbol": "context_symbol",
            "assigned_source_role": "context_assigned_source_role",
        }
    )
    registry_projection = registry[sorted(required_registry)].rename(
        columns={
            "event_cluster_id": "registry_event_cluster_id",
            "task_id": "registry_task_id",
            "symbol": "registry_symbol",
            "split": "registry_split",
            "assigned_source_role": "registry_assigned_source_role",
        }
    )
    projection = context_projection.merge(
        registry_projection,
        on="context_id",
        how="inner",
        validate="one_to_one",
    )
    if len(projection) != len(context_projection) or len(projection) != len(
        registry_projection
    ):
        raise RuntimeError("registered context and split registry are not one-to-one")
    for context_field, registry_field in (
        ("context_task_id", "registry_task_id"),
        ("context_symbol", "registry_symbol"),
        ("context_assigned_source_role", "registry_assigned_source_role"),
    ):
        mismatch = projection[context_field].astype(str).ne(
            projection[registry_field].astype(str)
        )
        if mismatch.any():
            raise RuntimeError(
                f"registered context {context_field.removeprefix('context_')} "
                "disagrees with the split registry"
            )
    registry_splits = projection["registry_split"].astype(str).str.lower()
    if not registry_splits.eq(split).all():
        raise RuntimeError("registered split registry contains an unexpected split")

    frame = proposals.merge(
        projection, on="context_id", how="left", validate="many_to_one"
    )
    attached = [
        "context_task_id",
        "context_symbol",
        "context_assigned_source_role",
        "registry_event_cluster_id",
        "registry_task_id",
        "registry_symbol",
        "registry_split",
        "registry_assigned_source_role",
        "decision_timestamp",
        "volatility_bps",
    ]
    if frame[attached].isna().any().any():
        raise RuntimeError("registered context attachment lost a proposal row")
    checks = {
        "event_cluster_id": "registry_event_cluster_id",
        "task_id": "registry_task_id",
        "symbol": "registry_symbol",
        "split": "registry_split",
        "assigned_source_role": "registry_assigned_source_role",
    }
    for proposal_field, context_field in checks.items():
        mismatch = frame[proposal_field].astype(str).ne(frame[context_field].astype(str))
        if mismatch.any():
            raise RuntimeError(
                f"proposal {proposal_field} disagrees with the hash-verified context"
            )
    timestamps = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce")
    volatility = pd.to_numeric(frame["volatility_bps"], errors="coerce")
    if timestamps.isna().any() or volatility.isna().any() or volatility.lt(0).any():
        raise RuntimeError("attached registered context fields are invalid")
    frame["decision_timestamp"] = timestamps
    frame["volatility_bps"] = volatility.astype(float)
    drop_columns = {
        *checks.values(),
        "context_task_id",
        "context_symbol",
        "context_assigned_source_role",
    }
    return frame.drop(columns=sorted(drop_columns))


def load_hash_verified_split_context(
    proposals: pd.DataFrame, config: dict[str, Any], split: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    if split not in {"development", "paper_test"}:
        raise ValueError(f"registered context attachment prohibits split {split!r}")
    source = config["binance"]
    dataset_manifest_path = resolve_root_path(source["dataset_manifest"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("outcome_metrics_computed") is not False:
        raise RuntimeError("dataset manifest crossed the pre-result boundary")
    context_path = resolve_root_path(source["derived_dir"]) / f"{split}_contexts.csv"
    registry_path = resolve_root_path(source["derived_dir"]) / "split_registry.csv"
    try:
        relative = str(context_path.relative_to(ROOT))
    except ValueError:
        relative = str(context_path.resolve())
    expected = dataset_manifest.get("outputs", {}).get(relative)
    if not expected or sha256(context_path) != expected:
        raise RuntimeError(f"registered context hash mismatch: {relative}")
    try:
        registry_relative = str(registry_path.relative_to(ROOT))
    except ValueError:
        registry_relative = str(registry_path.resolve())
    registry_expected = dataset_manifest.get("outputs", {}).get(registry_relative)
    if not registry_expected or sha256(registry_path) != registry_expected:
        raise RuntimeError(f"registered split-registry hash mismatch: {registry_relative}")
    contexts = pd.read_csv(
        context_path,
        usecols=[
            "context_id",
            "task_id",
            "symbol",
            "assigned_source_role",
            "decision_timestamp",
            "volatility_bps",
        ],
    )
    registry = pd.read_csv(
        registry_path,
        usecols=[
            "context_id",
            "event_cluster_id",
            "task_id",
            "symbol",
            "split",
            "assigned_source_role",
        ],
    )
    registry = registry[registry["split"].astype(str).str.lower().eq(split)].copy()
    attached = attach_registered_context_fields(
        proposals,
        contexts,
        registry,
        split=split,
    )
    return attached, {
        "context_file": relative,
        "context_sha256": str(expected),
        "split_registry_file": registry_relative,
        "split_registry_sha256": str(registry_expected),
        "dataset_manifest": (
            str(dataset_manifest_path.relative_to(ROOT))
            if dataset_manifest_path.is_relative_to(ROOT)
            else str(dataset_manifest_path.resolve())
        ),
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "volatility_provenance": (
            "sqrt(30)-scaled standard deviation of one-minute log returns from "
            "decision-31min through decision-1min; no outcome-window bar"
        ),
        "decision_timestamp_provenance": (
            "registered calendar date plus frozen decision_hour_utc"
        ),
    }


@dataclass(frozen=True)
class V06Rule:
    name: str

    def decide(self, frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
        if self.name == "Direct Prior":
            decision = np.full(len(frame), "execute", dtype=object)
        elif self.name == "Confidence Gate":
            decision = np.where(
                frame["confidence"].to_numpy(dtype=float) >= thresholds["confidence"],
                "execute",
                "abstain",
            )
        elif self.name == "Uncertainty Gate":
            decision = np.where(
                frame["uncertainty"].to_numpy(dtype=float) <= thresholds["uncertainty"],
                "execute",
                "abstain",
            )
        elif self.name == "Risk Filter":
            total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
            allowed = (total_cost <= thresholds["risk_cost_bps"]) & (
                frame["volatility_proxy"] <= thresholds["risk_volatility"]
            )
            decision = np.where(allowed, "execute", "abstain")
        elif self.name == "Cost-Aware Gate":
            total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
            allowed = frame["expected_edge_bps"] > total_cost + thresholds["cost_margin_bps"]
            decision = np.where(allowed, "execute", "abstain")
        elif self.name == "Hard Role Gate":
            decision = np.where(frame["original_source_eligible"].astype(bool), "execute", "abstain")
        elif self.name == "Lifecycle Checklist":
            total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
            role_ok = frame["original_source_eligible"].astype(bool)
            high_review_risk = (
                frame["uncertainty"] >= thresholds["lifecycle_review_uncertainty"]
            ) | (frame["volatility_proxy"] >= thresholds["lifecycle_review_volatility"])
            edge_after_cost = frame["expected_edge_bps"] - total_cost
            decision = np.full(len(frame), "abstain", dtype=object)
            decision[(~role_ok).to_numpy()] = "review"
            decision[(role_ok & high_review_risk).to_numpy()] = "review"
            reduce_mask = role_ok & ~high_review_risk & (edge_after_cost > 0)
            decision[reduce_mask.to_numpy()] = "reduce"
            execute_mask = role_ok & ~high_review_risk & (
                edge_after_cost > thresholds["lifecycle_execute_margin_bps"]
            )
            decision[execute_mask.to_numpy()] = "execute"
        else:
            raise KeyError(f"unknown v0.6 rule: {self.name}")
        decision = np.asarray(decision, dtype=object)
        decision[frame["candidate_action"].eq(0).to_numpy()] = "abstain"
        return decision


def registered_rules(config: dict[str, Any]) -> list[V06Rule]:
    return [V06Rule(str(name)) for name in config["evaluation"]["zero_shot_rules"]]


def _directional_outcomes(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    mask = frame["task_id"].eq("directional_execution")
    if not mask.any():
        return
    current = frame.loc[mask]
    direction = current["candidate_action"].to_numpy(dtype=float)
    entry = current["entry_price"].to_numpy(dtype=float)
    exit_price = current["exit_price_30m"].to_numpy(dtype=float)
    gross = direction * (exit_price / entry - 1.0) * 10000.0
    source = config["binance"]
    notional = float(source["action_notional_usd"])
    slope = float(source["impact_slope_bps"])
    cap = float(source["impact_cap_bps"])
    entry_depth = np.where(
        direction > 0,
        current["entry_ask_depth_1pct"],
        current["entry_bid_depth_1pct"],
    ).astype(float)
    exit_depth = np.where(
        direction > 0,
        current["exit_bid_depth_1pct_30m"],
        current["exit_ask_depth_1pct_30m"],
    ).astype(float)
    entry_impact = np.asarray(
        [impact_bps(notional, value, slope, cap) for value in entry_depth], dtype=float
    )
    exit_impact = np.asarray(
        [impact_bps(notional, value, slope, cap) for value in exit_depth], dtype=float
    )
    reduced_entry = np.asarray(
        [impact_bps(notional * 0.5, value, slope, cap) for value in entry_depth], dtype=float
    )
    reduced_exit = np.asarray(
        [impact_bps(notional * 0.5, value, slope, cap) for value in exit_depth], dtype=float
    )
    fee = current["fee_bps"].to_numpy(dtype=float)
    active = direction != 0
    full = np.where(active, gross - fee - entry_impact - exit_impact, 0.0)
    reduced = np.where(active, 0.5 * (gross - fee - reduced_entry - reduced_exit), 0.0)
    future_min = current["future_min_price_30m"].to_numpy(dtype=float)
    future_max = current["future_max_price_30m"].to_numpy(dtype=float)
    long_adverse = (future_min / entry - 1.0) * 10000.0
    short_adverse = -(future_max / entry - 1.0) * 10000.0
    adverse = np.where(direction > 0, long_adverse, short_adverse)
    adverse = np.where(active, adverse, 0.0)
    volatility_bps = current["volatility_bps"].to_numpy(dtype=float)
    task = config["tasks"]["directional_execution"]
    material_threshold = np.maximum(
        float(task["material_loss_floor_bps"]),
        float(task["material_loss_volatility_multiplier"]) * volatility_bps,
    )
    tail_threshold = np.maximum(
        float(task["tail_loss_floor_bps"]),
        float(task["tail_loss_volatility_multiplier"]) * volatility_bps,
    )
    scale = np.maximum(volatility_bps, float(task["material_loss_floor_bps"]))
    frame.loc[mask, "gross_return_bps"] = gross
    frame.loc[mask, "full_task_utility"] = full
    frame.loc[mask, "reduced_task_utility"] = reduced
    frame.loc[mask, "normalized_full_task_utility"] = np.clip(full / scale, -5.0, 5.0)
    frame.loc[mask, "normalized_reduced_task_utility"] = np.clip(reduced / scale, -5.0, 5.0)
    frame.loc[mask, "economic_loss_full"] = active & (full < 0.0)
    frame.loc[mask, "economic_loss_reduced"] = active & (reduced < 0.0)
    frame.loc[mask, "material_harm_full"] = active & (full < -material_threshold)
    frame.loc[mask, "material_harm_reduced"] = active & (reduced < -material_threshold)
    frame.loc[mask, "tail_harm_full"] = active & (adverse < -tail_threshold)
    frame.loc[mask, "tail_harm_reduced"] = active & (0.5 * adverse < -tail_threshold)
    frame.loc[mask, "risk_event"] = False
    frame.loc[mask, "material_risk_event"] = False
    frame.loc[mask, "candidate_benign_opportunity"] = active & (full > 0.0)


def _risk_limit_outcomes(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    mask = frame["task_id"].eq("risk_limit_increase")
    if not mask.any():
        return
    current = frame.loc[mask]
    active = current["candidate_action"].ne(0).to_numpy()
    entry = current["entry_price"].to_numpy(dtype=float)
    exit_price = current["exit_price_60m"].to_numpy(dtype=float)
    abs_move = np.abs(exit_price / entry - 1.0) * 10000.0
    realized_vol = current["future_realized_volatility_bps_60m"].to_numpy(dtype=float)
    deterioration = current["depth_deterioration_fraction_60m"].to_numpy(dtype=float)
    pre_vol = current["volatility_bps"].to_numpy(dtype=float)
    task = config["tasks"]["risk_limit_increase"]
    shock_move = np.maximum(
        float(task["shock_abs_move_floor_bps"]),
        float(task["shock_abs_move_volatility_multiplier"]) * pre_vol,
    )
    shock_vol = np.maximum(
        float(task["shock_realized_vol_floor_bps"]),
        float(task["shock_realized_volatility_multiplier"]) * pre_vol,
    )
    material_move = np.maximum(
        float(task["material_abs_move_floor_bps"]),
        float(task["material_abs_move_volatility_multiplier"]) * pre_vol,
    )
    risk_event = active & (
        (abs_move >= shock_move)
        | (realized_vol >= shock_vol)
        | (deterioration >= float(task["shock_depth_deterioration_fraction"]))
    )
    material_event = active & (
        (abs_move >= material_move)
        | (deterioration >= float(task["material_depth_deterioration_fraction"]))
    )
    benign_value = float(task["normalized_benign_capacity_utility"])
    full_utility = np.where(material_event, -2.0, np.where(risk_event, -1.0, benign_value))
    full_utility = np.where(active, full_utility, 0.0)
    reduced_utility = 0.5 * full_utility
    role_eligible = current["original_source_eligible"].astype(bool).to_numpy()
    frame.loc[mask, "gross_return_bps"] = np.nan
    frame.loc[mask, "full_task_utility"] = full_utility
    frame.loc[mask, "reduced_task_utility"] = reduced_utility
    frame.loc[mask, "normalized_full_task_utility"] = full_utility
    frame.loc[mask, "normalized_reduced_task_utility"] = reduced_utility
    frame.loc[mask, "economic_loss_full"] = False
    frame.loc[mask, "economic_loss_reduced"] = False
    frame.loc[mask, "material_harm_full"] = material_event
    frame.loc[mask, "material_harm_reduced"] = material_event
    frame.loc[mask, "tail_harm_full"] = False
    frame.loc[mask, "tail_harm_reduced"] = False
    frame.loc[mask, "risk_event"] = risk_event
    frame.loc[mask, "material_risk_event"] = material_event
    frame.loc[mask, "candidate_benign_opportunity"] = active & ~risk_event & role_eligible
    frame.loc[mask, "future_abs_move_bps_60m"] = abs_move


def materialize_outcomes(
    proposals: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    join_keys = ["context_id", "event_cluster_id", "task_id"]
    identity_fields = [
        "symbol",
        "assigned_source_role",
        "split",
        "decision_timestamp",
    ]
    required_proposal_identity = set(join_keys + identity_fields)
    required_outcome_identity = set(join_keys + identity_fields)
    missing_proposal = required_proposal_identity.difference(proposals.columns)
    missing_outcome = required_outcome_identity.difference(outcomes.columns)
    if missing_proposal:
        raise ValueError(
            f"v0.6 proposals are missing registered identity: {sorted(missing_proposal)}"
        )
    if missing_outcome:
        raise ValueError(
            f"v0.6 outcomes are missing sealed identity: {sorted(missing_outcome)}"
        )
    if outcomes.duplicated(join_keys).any():
        raise ValueError("v0.6 sealed outcomes contain duplicate row identity")

    proposal_keys = proposals[join_keys].drop_duplicates()
    outcome_keys = outcomes[join_keys].drop_duplicates()
    key_alignment = proposal_keys.merge(
        outcome_keys,
        on=join_keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not key_alignment["_merge"].eq("both").all():
        raise RuntimeError("v0.6 proposal and outcome row identities are not identical")

    outcome_identity = outcomes[join_keys + identity_fields].rename(
        columns={field: f"outcome_{field}" for field in identity_fields}
    )
    identity = proposals[join_keys + identity_fields].merge(
        outcome_identity,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    outcome_columns = [f"outcome_{field}" for field in identity_fields]
    if identity[outcome_columns].isna().any().any():
        raise RuntimeError("v0.6 sealed outcome identity is incomplete")
    for field in ("symbol", "assigned_source_role"):
        mismatch = identity[field].astype(str).ne(
            identity[f"outcome_{field}"].astype(str)
        )
        if mismatch.any():
            raise RuntimeError(f"v0.6 outcome {field} disagrees with registered identity")
    proposal_split = identity["split"].astype(str).str.lower().str.replace("-", "_")
    outcome_split = (
        identity["outcome_split"].astype(str).str.lower().str.replace("-", "_")
    )
    if proposal_split.ne(outcome_split).any():
        raise RuntimeError("v0.6 outcome split disagrees with registered identity")
    proposal_timestamp = pd.to_datetime(
        identity["decision_timestamp"], utc=True, errors="coerce"
    )
    outcome_timestamp = pd.to_datetime(
        identity["outcome_decision_timestamp"], utc=True, errors="coerce"
    )
    if (
        proposal_timestamp.isna().any()
        or outcome_timestamp.isna().any()
        or proposal_timestamp.ne(outcome_timestamp).any()
    ):
        raise RuntimeError(
            "v0.6 outcome decision_timestamp disagrees with registered identity"
        )

    outcome_payload = outcomes.drop(columns=identity_fields)
    frame = proposals.merge(
        outcome_payload,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    required = [
        "entry_price",
        "exit_price_30m",
        "exit_price_60m",
        "future_min_price_30m",
        "future_max_price_30m",
        "future_realized_volatility_bps_60m",
        "entry_bid_depth_1pct",
        "entry_ask_depth_1pct",
        "exit_bid_depth_1pct_30m",
        "exit_ask_depth_1pct_30m",
        "depth_deterioration_fraction_60m",
    ]
    if frame[required].isna().any().any():
        missing = frame.loc[frame[required].isna().any(axis=1), ["context_id", "task_id"]]
        raise RuntimeError(f"v0.6 sealed outcome inputs are incomplete: {missing.head().to_dict('records')}")
    initialized = {
        "gross_return_bps": np.nan,
        "full_task_utility": 0.0,
        "reduced_task_utility": 0.0,
        "normalized_full_task_utility": 0.0,
        "normalized_reduced_task_utility": 0.0,
        "economic_loss_full": False,
        "economic_loss_reduced": False,
        "material_harm_full": False,
        "material_harm_reduced": False,
        "tail_harm_full": False,
        "tail_harm_reduced": False,
        "risk_event": False,
        "material_risk_event": False,
        "candidate_benign_opportunity": False,
    }
    for column, value in initialized.items():
        frame[column] = value
    _directional_outcomes(frame, config)
    _risk_limit_outcomes(frame, config)
    frame["row_id"] = (
        frame["model_id"].astype(str)
        + ":"
        + frame["task_id"].astype(str)
        + ":"
        + frame["context_id"].astype(str)
    )
    return frame


def apply_decision(frame: pd.DataFrame, decision: np.ndarray) -> pd.DataFrame:
    ruled = frame.copy()
    ruled["decision"] = np.asarray(decision, dtype=object)
    ruled["authorized"] = ruled["decision"].isin(AUTHORIZED_DECISIONS)
    execute = ruled["decision"].eq("execute")
    reduce = ruled["decision"].eq("reduce")
    ruled["selected_task_utility"] = np.select(
        [execute, reduce],
        [ruled["full_task_utility"], ruled["reduced_task_utility"]],
        default=0.0,
    )
    ruled["selected_normalized_task_utility"] = np.select(
        [execute, reduce],
        [ruled["normalized_full_task_utility"], ruled["normalized_reduced_task_utility"]],
        default=0.0,
    )
    ruled["selected_economic_loss"] = np.select(
        [execute, reduce],
        [ruled["economic_loss_full"], ruled["economic_loss_reduced"]],
        default=False,
    ).astype(bool)
    ruled["selected_material_harm"] = np.select(
        [execute, reduce],
        [ruled["material_harm_full"], ruled["material_harm_reduced"]],
        default=False,
    ).astype(bool)
    ruled["selected_tail_harm"] = np.select(
        [execute, reduce],
        [ruled["tail_harm_full"], ruled["tail_harm_reduced"]],
        default=False,
    ).astype(bool)
    ruled["selected_risk_event"] = ruled["authorized"] & ruled["risk_event"].astype(bool)
    ruled["selected_material_risk_event"] = ruled["authorized"] & ruled[
        "material_risk_event"
    ].astype(bool)
    ruled["authority_violation"] = (
        ruled["authorized"]
        & ruled["task_id"].eq("risk_limit_increase")
        & ~ruled["original_source_eligible"].astype(bool)
    )
    ruled["missed_benign"] = ruled["candidate_benign_opportunity"].astype(bool) & ~ruled[
        "authorized"
    ]
    return ruled


def _rate(numerator: pd.Series, denominator: pd.Series) -> float | None:
    count = int(denominator.sum())
    if count == 0:
        return None
    return float(numerator[denominator].mean())


def metric_row(frame: pd.DataFrame) -> dict[str, object]:
    authorized = frame["authorized"].astype(bool)
    directional_auth = authorized & frame["task_id"].eq("directional_execution")
    risk_auth = authorized & frame["task_id"].eq("risk_limit_increase")
    benign = frame["candidate_benign_opportunity"].astype(bool)
    return {
        "rows": len(frame),
        "clusters": int(frame["event_cluster_id"].nunique()),
        "coverage": float(authorized.mean()),
        "execute_rate": float(frame["decision"].eq("execute").mean()),
        "reduce_rate": float(frame["decision"].eq("reduce").mean()),
        "review_rate": float(frame["decision"].eq("review").mean()),
        "abstain_rate": float(frame["decision"].eq("abstain").mean()),
        "economic_loss_authorization_rate": _rate(
            frame["selected_economic_loss"].astype(bool), directional_auth
        ),
        "material_harm_authorization_rate": _rate(
            frame["selected_material_harm"].astype(bool), authorized
        ),
        "tail_harm_authorization_rate": _rate(
            frame["selected_tail_harm"].astype(bool), directional_auth
        ),
        "risk_event_authorization_rate": _rate(
            frame["selected_risk_event"].astype(bool), risk_auth
        ),
        "material_risk_event_authorization_rate": _rate(
            frame["selected_material_risk_event"].astype(bool), risk_auth
        ),
        "authority_violation_rate": _rate(
            frame["authority_violation"].astype(bool), authorized
        ),
        "normalized_task_utility": float(frame["selected_normalized_task_utility"].mean()),
        "raw_task_utility": float(frame["selected_task_utility"].sum()),
        "missed_benign_opportunity": _rate(frame["missed_benign"].astype(bool), benign),
        "authorized_count": int(authorized.sum()),
        "directional_authorized_count": int(directional_auth.sum()),
        "risk_authorized_count": int(risk_auth.sum()),
        "benign_opportunity_count": int(benign.sum()),
    }


def evaluate_rules(
    frame: pd.DataFrame,
    config: dict[str, Any],
    thresholds_by_rule: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    default_thresholds = {
        key: float(value) for key, value in config["zero_shot_thresholds"].items()
    }
    for rule in registered_rules(config):
        thresholds = dict(default_thresholds)
        if thresholds_by_rule and rule.name in thresholds_by_rule:
            thresholds.update(
                {key: float(value) for key, value in thresholds_by_rule[rule.name].items()}
            )
        ruled = apply_decision(frame, rule.decide(frame, thresholds))
        ruled.insert(0, "rule", rule.name)
        decisions.append(ruled)
        for profile, current in [
            ("overall", ruled),
            ("directional_execution", ruled[ruled["task_id"] == "directional_execution"]),
            ("risk_limit_increase", ruled[ruled["task_id"] == "risk_limit_increase"]),
        ]:
            if current.empty:
                continue
            row = metric_row(current)
            row.update({"rule": rule.name, "profile": profile})
            metrics.append(row)
    return pd.concat(decisions, ignore_index=True), pd.DataFrame(metrics)
