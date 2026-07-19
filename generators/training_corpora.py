from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from finauth_audit.evaluation.seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_training_split(path: Path, required: list[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda name: name in set(required), chunksize=50_000):
        selected = chunk[chunk["split"] == "train"].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"training split is empty: {path}")
    return pd.concat(chunks, ignore_index=True)


def _normalize(frame: pd.DataFrame, features: list[str], layer: str) -> pd.DataFrame:
    result = frame.copy()
    defaults: dict[str, object] = {
        "candidate_action": 0,
        "confidence": 0.5,
        "uncertainty": 1.0,
        "expected_edge_bps": 0.0,
        "liquidity_cost_bps": 0.0,
        "turnover_cost_bps": 0.0,
        "fee_bps": 0.0,
        "volatility_proxy": 0.0,
        "liquidity_proxy": 0.0,
        "horizon": 0,
        "source_role": "unknown",
        "claimed_role": "unknown",
        "current_role_verified": False,
        "verified_current_role": "unknown",
        "hop_depth": 0,
        "transformation_type": "none",
        "traceability": "none",
        "lineage_attested": False,
        "time_to_resolution_hours": 0.0,
        "price_history_points_predecision": 0.0,
        "fee_rate": 0.0,
    }
    for feature in features:
        if feature not in result:
            result[feature] = defaults[feature]
        else:
            result[feature] = result[feature].fillna(defaults[feature])
    result["source_layer"] = layer
    result["training_cluster_id"] = layer + ":" + result["event_cluster_id"].astype(str)
    result["authority_laundering"] = result.get("authority_laundering", False)
    if not isinstance(result["authority_laundering"], pd.Series):
        result["authority_laundering"] = False
    return result


def _authorization_label(frame: pd.DataFrame) -> pd.Series:
    utility_safe = frame["full_utility"].astype(float) > 0.0
    role_safe = frame["original_source_eligible"].astype(bool)
    laundering_safe = ~frame["authority_laundering"].fillna(False).astype(bool)
    return (utility_safe & role_safe & laundering_safe).astype(int)


def _prediction_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["candidate_action"].astype(float) * frame["realized_return"].astype(float)
        > 0.0
    ).astype(int)


def _one_row_per_cluster(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for cluster_id, group in frame.sort_values("row_id").groupby("training_cluster_id"):
        index = derive_seed(seed, f"row/{cluster_id}") % len(group)
        selected.append(group.iloc[int(index)])
    return pd.DataFrame(selected).reset_index(drop=True)


def _sample_clusters(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    unique = _one_row_per_cluster(frame, seed)
    if len(unique) < count:
        raise ValueError(f"pool has {len(unique)} clusters; requested {count}")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(unique), size=count, replace=False))
    return unique.iloc[indices].reset_index(drop=True)


def _exclude_test_holdouts(
    frame: pd.DataFrame,
    controlled_holdouts: list[str],
    provenance_holdouts: list[str],
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if "opportunity_slice" in frame:
        mask &= ~frame["opportunity_slice"].isin(controlled_holdouts)
    if "attack_type" in frame:
        mask &= ~frame["attack_type"].isin(provenance_holdouts)
    return frame.loc[mask].copy()


def _variant_pool(
    variant: str,
    controlled: pd.DataFrame,
    provenance: pd.DataFrame,
    public: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    budget = int(config["training_clusters_per_variant"])
    if variant == "D0":
        return _sample_clusters(controlled, budget, seed)
    if variant == "D1":
        return _sample_clusters(
            controlled[controlled["opportunity_slice"] == "ordinary"], budget, seed
        )
    if variant == "D2":
        return _sample_clusters(
            controlled[controlled["opportunity_slice"] != "ordinary"], budget, seed
        )
    if variant == "D3":
        return _sample_clusters(controlled, budget, seed)
    if variant == "D4":
        return _sample_clusters(provenance, budget, seed)
    if variant == "D5":
        role_pool = provenance[
            provenance["attack_type"].isin(["role_noise", "delegation"])
        ]
        return _sample_clusters(role_pool, budget, seed)
    if variant == "D6":
        cost_pool = controlled[controlled["original_source_eligible"].astype(bool)]
        return _sample_clusters(cost_pool, budget, seed)
    if variant == "D7":
        provenance_count = int(config["full_audit_quotas"]["provenance_seen"])
        public_count = int(config["full_audit_quotas"]["public_train"])
        if provenance_count + public_count != budget:
            raise ValueError("full-audit quotas must equal training budget")
        left = _sample_clusters(provenance, provenance_count, derive_seed(seed, "D7/provenance"))
        right = _sample_clusters(public, public_count, derive_seed(seed, "D7/public"))
        return pd.concat([left, right], ignore_index=True)
    raise KeyError(variant)


def build(config_path: Path) -> tuple[Path, Path]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(
        (ROOT / "manifests" / "feature_access.json").read_text(encoding="utf-8")
    )
    legal_features = feature_manifest["tasks"]["training_utility"]["legal"]
    configured_features = config["features"]["numeric"] + config["features"]["categorical"]
    if len(configured_features) != len(set(configured_features)) or set(
        configured_features
    ) != set(legal_features):
        raise ValueError("training features must match the frozen legal manifest")

    required = list(
        dict.fromkeys(
            [
                "row_id",
                "event_cluster_id",
                "split",
                "opportunity_slice",
                "attack_type",
                "realized_return",
                "full_utility",
                "original_source_eligible",
                "authority_laundering",
            ]
            + configured_features
        )
    )
    controlled = _read_training_split(ROOT / config["inputs"]["controlled"], required)
    provenance = _read_training_split(ROOT / config["inputs"]["provenance"], required)
    public = _read_training_split(ROOT / config["inputs"]["public"], required)
    controlled = _normalize(controlled, configured_features, "controlled")
    provenance = _normalize(provenance, configured_features, "provenance")
    public = _normalize(public, configured_features, "public")

    controlled = _exclude_test_holdouts(
        controlled,
        config["controlled_test_holdouts"],
        config["provenance_test_holdouts"],
    )
    provenance = _exclude_test_holdouts(
        provenance,
        config["controlled_test_holdouts"],
        config["provenance_test_holdouts"],
    )
    public = _exclude_test_holdouts(
        public,
        config["controlled_test_holdouts"],
        config["provenance_test_holdouts"],
    )

    output_dir = ROOT / config["outputs"]["corpora_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    records: list[dict[str, object]] = []
    variants = list(config["variant_definitions"])
    for seed in config["seeds"]:
        for variant in variants:
            pool = _variant_pool(
                variant, controlled, provenance, public, config, int(seed)
            ).copy()
            label_kind = str(config["variant_definitions"][variant]["label"])
            pool["training_label"] = (
                _prediction_label(pool)
                if label_kind == "prediction_correct"
                else _authorization_label(pool)
            )
            pool["variant"] = variant
            pool["seed"] = int(seed)
            pool["label_semantics"] = label_kind
            columns = [
                "training_cluster_id",
                "variant",
                "seed",
                "source_layer",
                "label_semantics",
            ] + configured_features + ["training_label"]
            output = output_dir / f"{variant.lower()}_seed_{int(seed)}.csv"
            pool[columns].sort_values("training_cluster_id").to_csv(output, index=False)
            relative = str(output.relative_to(ROOT))
            files[relative] = sha256(output)
            records.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "rows": len(pool),
                    "clusters": int(pool["training_cluster_id"].nunique()),
                    "positive_rate": float(pool["training_label"].mean()),
                    "source_layers": pool["source_layer"].value_counts().sort_index().to_dict(),
                    "opportunity_slices": (
                        pool["opportunity_slice"].value_counts().sort_index().to_dict()
                        if "opportunity_slice" in pool
                        else {}
                    ),
                    "attack_types": (
                        pool["attack_type"].value_counts().sort_index().to_dict()
                        if "attack_type" in pool
                        else {}
                    ),
                    "label_semantics": label_kind,
                }
            )

    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project": config["project"],
        "version": config["version"],
        "mode": config["mode"],
        "confirmatory": False,
        "test_outcomes_evaluated": False,
        "training_clusters_per_variant": int(config["training_clusters_per_variant"]),
        "variants": variants,
        "seeds": [int(seed) for seed in config["seeds"]],
        "legal_features": configured_features,
        "controlled_test_holdouts": config["controlled_test_holdouts"],
        "provenance_test_holdouts": config["provenance_test_holdouts"],
        "records": records,
        "files": files,
        "config_sha256": sha256(config_path),
        "feature_manifest_sha256": sha256(ROOT / "manifests" / "feature_access.json"),
        "claim_boundary": (
            "Matched validation-development corpora for a tabular authorization learner. "
            "D0 uses prediction-only supervision; D1-D7 use authorization supervision. "
            "No LLM-training or deployment claim is supported."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    print(manifest_path)
    return output_dir, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build matched D0-D7 training corpora.")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "training_utility_smoke.yaml")
    )
    args = parser.parse_args()
    build(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
