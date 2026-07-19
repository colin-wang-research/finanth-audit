from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from finauth_audit.generators.controlled import ELIGIBLE_ROLES, SOURCE_SPECS


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(index: int, clusters: int, fractions: dict[str, float]) -> str:
    train_end = int(clusters * fractions["train"])
    validation_end = train_end + int(clusters * fractions["validation"])
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def _finalize_cluster(
    cluster_rows: list[dict[str, object]],
    realized_return: float,
    risk_penalty_scale: float,
) -> None:
    for row in cluster_rows:
        gross = float(row["candidate_action"]) * realized_return
        cost = (
            float(row["liquidity_cost_bps"])
            + float(row["turnover_cost_bps"])
            + float(row["fee_bps"])
        ) / 10_000.0
        inventory = abs(float(row.get("inventory_exposure", 0.0)))
        budget = float(row.get("risk_budget_utilization", 0.0))
        risk_penalty = risk_penalty_scale * (
            float(row["volatility_proxy"]) ** 2 + 0.35 * inventory**2 + 0.25 * budget**2
        )
        full_utility = gross - cost - risk_penalty
        reduced_utility = 0.4 * gross - 0.4 * cost - 0.16 * risk_penalty
        row.update(
            {
                "realized_return": realized_return,
                "full_utility": full_utility,
                "reduced_utility": reduced_utility,
                "harm_label": full_utility < 0,
                "tail_loss": min(full_utility, 0.0),
                "future_decoy": full_utility,
            }
        )
    oracle_positive = any(
        bool(row["original_source_eligible"])
        and max(float(row["full_utility"]), float(row["reduced_utility"])) > 0
        for row in cluster_rows
    )
    for row in cluster_rows:
        row["oracle_positive_opportunity"] = oracle_positive


def generate_sequential(config: dict[str, object], family: dict[str, object]) -> pd.DataFrame:
    rng = np.random.default_rng(int(family["seed"]))
    clusters = int(config["clusters_per_family"])
    fractions = config["split_fractions"]
    trend = 0.0
    inventory = 0.0
    volatility = 0.25
    liquidity = 0.90
    start = pd.Timestamp("2018-01-01T00:00:00Z")
    rows: list[dict[str, object]] = []
    for cluster_index in range(clusters):
        regime_jump = rng.random() < 0.035
        tail_event = rng.random() < 0.018
        trend = 0.82 * trend + float(rng.normal(0, 0.55))
        if regime_jump:
            trend += float(rng.normal(0, 2.0))
        volatility = float(np.clip(0.88 * volatility + 0.12 * abs(rng.normal(0.35, 0.18)), 0.08, 1.25))
        if tail_event:
            volatility = min(1.25, volatility + 0.55)
        liquidity = float(np.clip(0.90 * liquidity + 0.10 * rng.normal(0.85, 0.15) - 0.25 * tail_event, 0.05, 1.15))
        inventory = float(np.clip(0.72 * inventory + rng.normal(0, 0.35), -1.5, 1.5))
        latent_move = 0.0011 * trend + float(rng.normal(0, 0.00055 + 0.00085 * volatility))
        if tail_event:
            latent_move += float(rng.choice([-1, 1])) * rng.uniform(0.004, 0.010)
        market_impact_bps = 1.5 + 5.5 * abs(inventory) + 7.0 * volatility + 4.0 * (1.0 - liquidity)
        if tail_event:
            slice_name = "tail_event"
        elif regime_jump:
            slice_name = "regime_shift"
        elif liquidity < 0.40:
            slice_name = "liquidity_shock"
        elif abs(inventory) > 0.95:
            slice_name = "inventory_stress"
        else:
            slice_name = "ordinary"
        opportunity_intent = abs(latent_move) * 10_000 > market_impact_bps + 3.0
        cluster_rows: list[dict[str, object]] = []
        for source_index, (source_id, role, reliability) in enumerate(SOURCE_SPECS):
            signal = latent_move + float(rng.normal(0, 0.0004 + (1 - reliability) * 0.0012 + 0.0005 * volatility))
            direction = 1 if signal >= 0 else -1
            confidence = float(
                np.clip(0.25 + 0.45 * reliability + abs(signal) * 25 + rng.normal(0, 0.06), 0.05, 0.98)
            )
            uncertainty = float(
                np.clip(0.82 - 0.62 * reliability + 0.38 * volatility + rng.normal(0, 0.06), 0.03, 0.98)
            )
            expected_edge_bps = float(direction * signal * 10_000 * rng.uniform(0.85, 1.25))
            liquidity_cost_bps = float(max(0.2, 2.0 + 9.0 * (1.0 - liquidity) + 4.0 * volatility + rng.normal(0, 0.8)))
            turnover_cost_bps = float(max(0.2, 1.5 + 3.0 * source_index + 5.0 * abs(inventory) + rng.normal(0, 0.7)))
            fee_bps = float(0.8 + 0.5 * volatility)
            cluster_rows.append(
                {
                    "row_id": f"seq-{cluster_index:06d}-{source_index}",
                    "event_cluster_id": f"seq-cluster-{cluster_index:06d}",
                    "timestamp": (start + pd.Timedelta(hours=cluster_index)).isoformat(),
                    "split": _split(cluster_index, clusters, fractions),
                    "generator_family": "sequential_market",
                    "financial_task": "trading_authorization",
                    "opportunity_slice": slice_name,
                    "stress_tag": slice_name,
                    "stress_severity": volatility,
                    "certification_eligible": True,
                    "current_source_id": source_id,
                    "source_role": role,
                    "claimed_role": role,
                    "current_role_verified": True,
                    "candidate_action": direction,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "expected_edge_bps": expected_edge_bps,
                    "liquidity_cost_bps": liquidity_cost_bps + market_impact_bps,
                    "turnover_cost_bps": turnover_cost_bps,
                    "fee_bps": fee_bps,
                    "volatility_proxy": volatility,
                    "liquidity_proxy": liquidity,
                    "horizon": int((1, 5, 20, 60)[cluster_index % 4]),
                    "original_source_eligible": role in ELIGIBLE_ROLES,
                    "latent_opportunity_intent": bool(opportunity_intent),
                    "latent_move": latent_move,
                    "inventory_exposure": inventory,
                    "market_impact_bps": market_impact_bps,
                    "risk_budget_utilization": min(1.5, abs(inventory) * volatility),
                    "approval_chain_depth": 1,
                    "review_queue_pressure": 0.0,
                    "compliance_flag": False,
                }
            )
        realized = latent_move + float(rng.normal(0, 0.0005 + 0.0009 * volatility))
        _finalize_cluster(cluster_rows, realized, risk_penalty_scale=0.00035)
        rows.extend(cluster_rows)
    return pd.DataFrame(rows).sort_values(["timestamp", "row_id"]).reset_index(drop=True)


def generate_institutional(config: dict[str, object], family: dict[str, object]) -> pd.DataFrame:
    rng = np.random.default_rng(int(family["seed"]))
    clusters = int(config["clusters_per_family"])
    fractions = config["split_fractions"]
    tasks = ("trading_authorization", "credit_authorization", "research_to_action")
    start = pd.Timestamp("2019-01-01T00:00:00Z")
    rows: list[dict[str, object]] = []
    for cluster_index in range(clusters):
        task = tasks[cluster_index % len(tasks)]
        risk_budget = float(np.clip(rng.beta(2.2, 2.0), 0.02, 1.20))
        review_pressure = float(np.clip(rng.beta(2.0, 3.0), 0.0, 1.0))
        approval_depth = int(rng.integers(1, 6))
        compliance_flag = bool(rng.random() < (0.08 + 0.12 * risk_budget))
        tail_exposure = bool(rng.random() < (0.04 + 0.10 * risk_budget))
        latent_quality = float(rng.normal(0.0018, 0.0030))
        if task == "credit_authorization":
            latent_quality -= 0.0025 * tail_exposure + 0.0015 * compliance_flag
        elif task == "research_to_action":
            latent_quality -= 0.0012 * review_pressure
        else:
            latent_quality -= 0.0010 * risk_budget
        if compliance_flag:
            slice_name = "compliance_conflict"
        elif tail_exposure:
            slice_name = "tail_exposure"
        elif risk_budget > 0.85:
            slice_name = "risk_budget_pressure"
        elif approval_depth >= 5:
            slice_name = "approval_chain_gap"
        else:
            slice_name = "ordinary"
        base_cost = {
            "trading_authorization": 4.0,
            "credit_authorization": 7.0,
            "research_to_action": 3.0,
        }[task]
        opportunity_intent = latent_quality > (base_cost / 10_000.0) and not compliance_flag
        cluster_rows: list[dict[str, object]] = []
        for source_index, (source_id, role, reliability) in enumerate(SOURCE_SPECS):
            role_bias = 0.0005 if role in ELIGIBLE_ROLES else -0.0002
            signal = latent_quality + role_bias + float(rng.normal(0, 0.0008 + (1 - reliability) * 0.0015))
            direction = 1 if signal >= 0 else -1
            confidence = float(
                np.clip(0.30 + 0.42 * reliability + abs(signal) * 20 + 0.10 * compliance_flag + rng.normal(0, 0.07), 0.05, 0.99)
            )
            uncertainty = float(
                np.clip(0.80 - 0.55 * reliability + 0.25 * risk_budget + 0.18 * compliance_flag + rng.normal(0, 0.06), 0.03, 0.99)
            )
            expected_edge_bps = float(direction * signal * 10_000 * rng.uniform(0.85, 1.35))
            liquidity_cost_bps = float(max(0.2, base_cost + 4.0 * risk_budget + 3.0 * tail_exposure + rng.normal(0, 0.8)))
            turnover_cost_bps = float(max(0.2, 1.0 + 1.5 * source_index + 2.5 * review_pressure + rng.normal(0, 0.6)))
            fee_bps = float(0.7 + 0.8 * (task == "credit_authorization"))
            volatility = float(np.clip(0.25 + 0.45 * risk_budget + 0.25 * tail_exposure, 0.05, 1.20))
            liquidity = float(np.clip(1.0 - 0.45 * risk_budget - 0.25 * tail_exposure, 0.05, 1.15))
            cluster_rows.append(
                {
                    "row_id": f"inst-{cluster_index:06d}-{source_index}",
                    "event_cluster_id": f"inst-cluster-{cluster_index:06d}",
                    "timestamp": (start + pd.Timedelta(hours=cluster_index)).isoformat(),
                    "split": _split(cluster_index, clusters, fractions),
                    "generator_family": "institutional_workflow",
                    "financial_task": task,
                    "opportunity_slice": slice_name,
                    "stress_tag": slice_name,
                    "stress_severity": risk_budget,
                    "certification_eligible": True,
                    "current_source_id": source_id,
                    "source_role": role,
                    "claimed_role": role,
                    "current_role_verified": not compliance_flag or role in ELIGIBLE_ROLES,
                    "candidate_action": direction,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "expected_edge_bps": expected_edge_bps,
                    "liquidity_cost_bps": liquidity_cost_bps,
                    "turnover_cost_bps": turnover_cost_bps,
                    "fee_bps": fee_bps,
                    "volatility_proxy": volatility,
                    "liquidity_proxy": liquidity,
                    "horizon": int((5, 20, 60)[cluster_index % 3]),
                    "original_source_eligible": role in ELIGIBLE_ROLES and not compliance_flag,
                    "latent_opportunity_intent": bool(opportunity_intent),
                    "latent_move": latent_quality,
                    "inventory_exposure": 0.0,
                    "market_impact_bps": 0.0,
                    "risk_budget_utilization": risk_budget,
                    "approval_chain_depth": approval_depth,
                    "review_queue_pressure": review_pressure,
                    "compliance_flag": compliance_flag,
                }
            )
        realized = latent_quality + float(rng.normal(0, 0.0014 + 0.0012 * tail_exposure))
        _finalize_cluster(cluster_rows, realized, risk_penalty_scale=0.00045)
        rows.extend(cluster_rows)
    return pd.DataFrame(rows).sort_values(["timestamp", "row_id"]).reset_index(drop=True)


def build(config_path: Path) -> tuple[list[Path], Path]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config["priors_per_cluster"]) != len(SOURCE_SPECS):
        raise ValueError("priors_per_cluster must match the frozen source schema")
    generators = {
        "sequential_market": generate_sequential,
        "institutional_workflow": generate_institutional,
    }
    outputs: list[Path] = []
    family_manifest: dict[str, object] = {}
    for name, family in config["families"].items():
        frame = generators[name](config, family)
        output_path = ROOT / family["output_data"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        outputs.append(output_path)
        family_manifest[name] = {
            "seed": int(family["seed"]),
            "data_path": str(output_path.relative_to(ROOT)),
            "data_sha256": sha256(output_path),
            "rows": len(frame),
            "clusters": int(frame["event_cluster_id"].nunique()),
            "split_counts": frame["split"].value_counts().sort_index().to_dict(),
            "financial_tasks": frame["financial_task"].value_counts().sort_index().to_dict(),
            "slice_counts": frame["opportunity_slice"].value_counts().sort_index().to_dict(),
        }
    manifest_path = ROOT / config["output_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project": config["project"],
        "version": config["version"],
        "mode": config["mode"],
        "evaluation_split": "validation",
        "test_outcomes_evaluated": False,
        "config_sha256": sha256(config_path),
        "families": family_manifest,
        "claim_boundary": "Prospective mechanistically distinct controlled generators; not independent external implementations, market simulations, deployment logs, or test evidence.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outputs, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build prospective mechanistic generator families.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "generator_robustness.yaml"))
    args = parser.parse_args()
    outputs, manifest = build(Path(args.config))
    for output in outputs:
        print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
