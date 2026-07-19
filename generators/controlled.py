from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE_ROLES = {"edge_proposer", "learned_prior", "policy_proposer"}
SOURCE_SPECS = (
    ("heuristic_edge", "edge_proposer", 0.66),
    ("learned_signal", "learned_prior", 0.74),
    ("policy_proposal", "policy_proposer", 0.70),
    ("risk_critic", "risk_critic", 0.55),
)


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


def generate(config: dict[str, object]) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed"]))
    clusters = int(config["clusters"])
    priors_per_cluster = int(config["priors_per_cluster"])
    if priors_per_cluster != len(SOURCE_SPECS):
        raise ValueError(f"priors_per_cluster must be {len(SOURCE_SPECS)}")

    slice_names = list(config["slice_mixture"])
    slice_probabilities = np.asarray([config["slice_mixture"][name] for name in slice_names], dtype=float)
    if not np.isclose(slice_probabilities.sum(), 1.0):
        raise ValueError("slice mixture must sum to one")

    stress_scale = {
        "ordinary": 0.0,
        "moderate_stress": 0.35,
        "severe_stress": 0.85,
        "rare_safe_opportunity": 0.20,
        "high_missed_opportunity": 0.30,
        "stress_only_period": 1.00,
        "no_opportunity_control": 0.95,
    }
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2020-01-01T00:00:00Z")

    for cluster_index in range(clusters):
        slice_name = str(rng.choice(slice_names, p=slice_probabilities))
        intended_rate = float(config["expected_opportunity_rate"][slice_name])
        opportunity_intent = bool(rng.random() < intended_rate)
        true_direction = int(rng.choice([-1, 1]))
        severity = stress_scale[slice_name]
        if slice_name == "no_opportunity_control":
            latent_magnitude = float(rng.uniform(0.0000, 0.0008))
        elif opportunity_intent:
            high = 0.016 if slice_name == "high_missed_opportunity" else 0.010
            latent_magnitude = float(rng.uniform(0.0045, high))
        else:
            # Non-opportunity clusters may still contain persuasive priors, but
            # their latent move is intentionally too small to dominate ordinary
            # execution costs. This parameter is rule-independent.
            latent_magnitude = float(rng.uniform(0.0000, 0.00025))
        latent_move = true_direction * latent_magnitude

        cluster_rows: list[dict[str, object]] = []
        for source_index, (source_id, role, reliability) in enumerate(SOURCE_SPECS):
            source_noise = 0.00035 + (1.0 - reliability) * 0.0009 + severity * 0.00045
            source_signal = latent_move + float(rng.normal(0, source_noise))
            if source_signal == 0:
                source_signal = float(rng.choice([-1, 1])) * 1e-6
            direction = 1 if source_signal > 0 else -1
            signal_strength = abs(source_signal) * 10_000
            confidence = np.clip(
                0.24 + reliability * 0.24 + signal_strength / 65 + rng.normal(0, 0.055),
                0.05,
                0.98,
            )
            if severity >= 0.8 and rng.random() < 0.28:
                confidence = np.clip(confidence + rng.uniform(0.10, 0.24), 0.05, 0.99)
            uncertainty = np.clip(1.03 - confidence + rng.normal(0, 0.06), 0.02, 0.98)
            expected_edge_bps = max(
                0.2,
                signal_strength * (0.55 + reliability * 0.50) + rng.normal(0, 1.2),
            )
            if slice_name in {"severe_stress", "no_opportunity_control"} and rng.random() < 0.35:
                expected_edge_bps *= float(rng.uniform(1.15, 1.55))

            liquidity_cost_bps = max(0.2, float(rng.normal(3.0 + severity * 10.0, 1.2)))
            turnover_cost_bps = max(0.2, float(rng.normal(2.5 + severity * 7.0 + source_index, 1.0)))
            fee_bps = 0.8 + severity * 1.2
            volatility_proxy = float(np.clip(rng.normal(0.35 + severity * 0.48, 0.10), 0.02, 1.20))
            liquidity_proxy = float(np.clip(1.0 - severity * 0.65 + rng.normal(0, 0.10), 0.02, 1.20))

            cluster_rows.append(
                {
                    "row_id": f"audit-{cluster_index:06d}-{source_index}",
                    "event_cluster_id": f"cluster-{cluster_index:06d}",
                    "timestamp": (start + pd.Timedelta(hours=cluster_index)).isoformat(),
                    "split": _split(cluster_index, clusters, config["split_fractions"]),
                    "opportunity_slice": slice_name,
                    "stress_tag": slice_name,
                    "stress_severity": severity,
                    "certification_eligible": slice_name != "no_opportunity_control",
                    "current_source_id": source_id,
                    "source_role": role,
                    "claimed_role": role,
                    "current_role_verified": True,
                    "candidate_action": direction,
                    "confidence": float(confidence),
                    "uncertainty": float(uncertainty),
                    "expected_edge_bps": float(expected_edge_bps),
                    "liquidity_cost_bps": liquidity_cost_bps,
                    "turnover_cost_bps": turnover_cost_bps,
                    "fee_bps": fee_bps,
                    "volatility_proxy": volatility_proxy,
                    "liquidity_proxy": liquidity_proxy,
                    "horizon": int((1, 5, 20, 60)[cluster_index % 4]),
                    "original_source_eligible": role in ELIGIBLE_ROLES,
                    "latent_opportunity_intent": opportunity_intent,
                    "latent_move": latent_move,
                }
            )

        if opportunity_intent:
            realized_return = latent_move + float(rng.normal(0, 0.0007 + severity * 0.0006))
        else:
            realized_return = float(rng.normal(0, 0.00018 + severity * 0.00008))
        if slice_name == "no_opportunity_control":
            realized_return = float(rng.normal(0, 0.00035))
        for row in cluster_rows:
            gross = row["candidate_action"] * realized_return
            cost = (row["liquidity_cost_bps"] + row["turnover_cost_bps"] + row["fee_bps"]) / 10_000
            risk_penalty = 0.0004 * row["volatility_proxy"] ** 2
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
            row["original_source_eligible"] and max(row["full_utility"], row["reduced_utility"]) > 0
            for row in cluster_rows
        )
        for row in cluster_rows:
            row["oracle_positive_opportunity"] = oracle_positive
        rows.extend(cluster_rows)

    frame = pd.DataFrame(rows)
    return frame.sort_values(["timestamp", "row_id"]).reset_index(drop=True)


def build(config_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame = generate(config)
    data_path = ROOT / config["output_data"]
    manifest_path = ROOT / config["output_manifest"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False)
    manifest = {
        "project": config["project"],
        "version": config["version"],
        "mode": config["mode"],
        "seed": int(config["seed"]),
        "rows": len(frame),
        "clusters": int(frame["event_cluster_id"].nunique()),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "slice_counts": frame["opportunity_slice"].value_counts().sort_index().to_dict(),
        "source_role_counts": frame["source_role"].value_counts().sort_index().to_dict(),
        "data_path": str(data_path.relative_to(ROOT)),
        "data_sha256": sha256(data_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "claim_boundary": "Controlled authorization scenarios; not market simulation, deployment logs, or investment evidence.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FinAuth-Audit controlled validation data.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.yaml"))
    args = parser.parse_args()
    data_path, manifest_path = build(Path(args.config))
    print(data_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
