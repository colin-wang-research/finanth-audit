from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from finauth_audit.attacks.provenance import transform


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(base: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed"]))
    attack_names = list(config["attack_mix"])
    attack_probabilities = np.asarray([config["attack_mix"][name] for name in attack_names], dtype=float)
    if not np.isclose(attack_probabilities.sum(), 1.0):
        raise ValueError("attack mixture must sum to one")
    noise_rates = tuple(float(value) for value in config["role_noise_rates"])
    hop_depths = tuple(int(value) for value in config["hop_depths"])

    rows: list[dict[str, object]] = []
    for index, row in base.reset_index(drop=True).iterrows():
        attack_type = str(rng.choice(attack_names, p=attack_probabilities))
        traceability = "traceable" if rng.random() < float(config["traceable_fraction"]) else "untraceable"
        if attack_type in {"clean", "role_noise"}:
            traceability = "traceable"
        role_noise_rate = noise_rates[index % len(noise_rates)] if attack_type == "role_noise" else 0.0
        hop_depth = hop_depths[index % len(hop_depths)] if attack_type == "multi_hop" else 0
        transformed = transform(
            row_id=str(row["row_id"]),
            original_source_id=str(row["current_source_id"]),
            original_role=str(row["source_role"]),
            attack_type=attack_type,
            traceability=traceability,
            role_noise_rate=role_noise_rate,
            hop_depth=hop_depth,
            rng=rng,
        )
        payload = row.to_dict()
        payload.update(transformed.__dict__)
        payload["prior_action"] = payload["candidate_action"]
        payload["decision_timestamp"] = payload["timestamp"]
        payload["outcome_timestamp"] = (
            pd.Timestamp(payload["timestamp"]) + pd.Timedelta(hours=int(payload["horizon"]))
        ).isoformat()
        payload["direct_leakage"] = not transformed.current_source_eligible
        payload["indirect_leakage"] = transformed.authority_laundering and transformed.current_source_eligible
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["timestamp", "row_id"]).reset_index(drop=True)


def build(config_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_data"]
    frame = generate(pd.read_csv(source_path), config)
    output_path = ROOT / config["output_data"]
    manifest_path = ROOT / config["output_manifest"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    manifest = {
        "project": config["project"],
        "version": config["version"],
        "mode": config["mode"],
        "seed": int(config["seed"]),
        "rows": len(frame),
        "clusters": int(frame["event_cluster_id"].nunique()),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "attack_counts": frame["attack_type"].value_counts().sort_index().to_dict(),
        "traceability_counts": frame["traceability"].value_counts().sort_index().to_dict(),
        "role_noise_counts": frame.loc[frame["attack_type"] == "role_noise", "role_noise_rate"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "hop_depth_counts": frame.loc[frame["attack_type"] == "multi_hop", "hop_depth"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "source_data_path": config["source_data"],
        "data_path": config["output_data"],
        "source_data_sha256": sha256(source_path),
        "data_sha256": sha256(output_path),
        "config_sha256": sha256(config_path),
        "claim_boundary": "Controlled provenance attacks; not deployed agent logs or observed institutional delegation.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build provenance laundering validation data.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "provenance_smoke.yaml"))
    args = parser.parse_args()
    data_path, manifest_path = build(Path(args.config))
    print(data_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
