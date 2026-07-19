from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finauth_audit.evaluation.external_orderbook_power import (
    COMPARISON_RULES,
    _cluster_burden,
    required_clusters,
    two_sided_power,
)
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.get("version") != "0.4.0":
        raise RuntimeError("Databento powered power gate requires v0.4.0 config")

    source_path = ROOT / "results" / "paper_test" / "controlled" / "decisions.csv"
    v02_freeze_path = ROOT / "manifests" / "paper_test_freeze.json"
    v02_registry_path = ROOT / "manifests" / "paper_test_registry.json"
    v02_freeze = json.loads(v02_freeze_path.read_text(encoding="utf-8"))
    v02_registry = json.loads(v02_registry_path.read_text(encoding="utf-8"))
    expected_source_hash = v02_registry.get("output_hashes", {}).get(
        "controlled/decisions.csv"
    )
    if v02_freeze.get("status") != "FROZEN_BEFORE_PAPER_TEST":
        raise RuntimeError("v0.2 freeze marker is invalid")
    if v02_registry.get("status") != "COMPLETE_INSPECTED":
        raise RuntimeError("v0.2 one-time paper-test registry is invalid")
    if expected_source_hash is None or sha256(source_path) != expected_source_hash:
        raise RuntimeError("v0.2 controlled decision source is not registered")

    frame = pd.read_csv(
        source_path,
        usecols=[
            "rule",
            "event_cluster_id",
            "row_id",
            "selected_harm",
            "selected_laundering",
        ],
    )
    burdens = {
        rule: _cluster_burden(frame, rule).set_index("event_cluster_id")
        for rule in COMPARISON_RULES
    }
    left, right = burdens[COMPARISON_RULES[0]], burdens[COMPARISON_RULES[1]]
    common = left.index.intersection(right.index)
    false_diff = (
        left.loc[common, "false_authorization_burden"]
        - right.loc[common, "false_authorization_burden"]
    )
    laundering_diff = (
        left.loc[common, "laundering_burden"]
        - right.loc[common, "laundering_burden"]
    )
    observed = {
        "clusters": len(common),
        "false_authorization_mean_difference": float(false_diff.mean()),
        "false_authorization_sd": float(false_diff.std(ddof=1)),
        "laundering_mean_difference": float(laundering_diff.mean()),
        "laundering_sd": float(laundering_diff.std(ddof=1)),
    }
    prereg_path = (
        ROOT / "manifests" / "preregistration" / "databento_powered_v04.yaml"
    )
    prereg = load_config(prereg_path)
    expected_sd_false = float(prereg["power_contract"]["false_authorization_sd"])
    expected_sd_laundering = float(prereg["power_contract"]["laundering_sd"])
    if not np.isclose(observed["false_authorization_sd"], expected_sd_false, atol=1e-10):
        raise RuntimeError("frozen false-authorization SD does not match v0.4 registration")
    if not np.isclose(observed["laundering_sd"], expected_sd_laundering, atol=1e-10):
        raise RuntimeError("frozen laundering SD does not match v0.4 registration")

    false_effect = abs(
        float(config["primary_endpoints"]["false_authorization_burden"]["sesoi"])
    )
    laundering_effect = abs(
        float(config["primary_endpoints"]["laundering_burden"]["sesoi"])
    )
    clusters = int(config["databento"]["paper_test_clusters"])
    minimum_power = float(config["primary_endpoints"]["minimum_power"])
    false_power = two_sided_power(false_effect, expected_sd_false, clusters)
    laundering_power = two_sided_power(
        laundering_effect, expected_sd_laundering, clusters
    )
    gate_passed = min(false_power, laundering_power) >= minimum_power
    report = {
        "project": "FinAuth-Audit",
        "version": "0.4.0",
        "external_outcomes_read": False,
        "source": "frozen v0.2 controlled paper-test cluster burdens only",
        "source_path": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256(source_path),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "preregistration_path": str(prereg_path.relative_to(ROOT)),
        "preregistration_sha256": sha256(prereg_path),
        "v02_freeze_manifest": str(v02_freeze_path.relative_to(ROOT)),
        "v02_freeze_manifest_sha256": sha256(v02_freeze_path),
        "v02_freeze_archive_sha256": v02_freeze.get("archive_sha256"),
        "v02_test_registry": str(v02_registry_path.relative_to(ROOT)),
        "v02_test_registry_sha256": sha256(v02_registry_path),
        "comparison": f"{COMPARISON_RULES[0]} minus {COMPARISON_RULES[1]}",
        "observed_v02": observed,
        "sesoi": {
            "false_authorization_burden": -false_effect,
            "laundering_burden": laundering_effect,
        },
        "required_clusters_for_80_percent_power": {
            "false_authorization_burden": required_clusters(
                false_effect, expected_sd_false, minimum_power
            ),
            "laundering_burden": required_clusters(
                laundering_effect, expected_sd_laundering, minimum_power
            ),
        },
        "databento": {
            "clusters": clusters,
            "false_authorization_power": false_power,
            "laundering_power": laundering_power,
            "minimum_endpoint_power": min(false_power, laundering_power),
            "confirmatory_gate_passed": gate_passed,
            "classification": "confirmatory_temporal_replication"
            if gate_passed
            else "descriptive_underpowered",
        },
        "gate_passed": gate_passed,
        "known_v03_direction_used_for_power": False,
        "claim_boundary": (
            "Pre-result design power derived exclusively from frozen v0.2 cluster "
            "burdens. The v0.3 Databento point estimate is not used."
        ),
    }
    output_path = resolve_root_path(config["freeze"]["power_report"])
    write_json(output_path, report)
    if not gate_passed:
        raise RuntimeError("Databento v0.4 confirmatory power gate did not pass")
    print(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute v0.4 Databento power from frozen v0.2 only."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "databento_powered_v04.yaml")
    )
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
