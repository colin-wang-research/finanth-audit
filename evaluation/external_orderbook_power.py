from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


COMPARISON_RULES = ("Cost-Aware Gate", "Lifecycle Checklist")


def two_sided_power(effect: float, standard_deviation: float, clusters: int, alpha: float = 0.05) -> float:
    if effect <= 0 or standard_deviation <= 0 or clusters <= 1:
        raise ValueError("effect, standard deviation, and clusters must be positive")
    noncentrality = effect * math.sqrt(clusters) / standard_deviation
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    return float(norm.cdf(-critical - noncentrality) + 1.0 - norm.cdf(critical - noncentrality))


def required_clusters(effect: float, standard_deviation: float, power: float, alpha: float = 0.05) -> int:
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    target = float(norm.ppf(power))
    return int(math.ceil(((critical + target) * standard_deviation / effect) ** 2))


def _cluster_burden(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    selected = frame[frame["rule"] == rule]
    return (
        selected.groupby("event_cluster_id", as_index=False)
        .agg(
            rows=("row_id", "size"),
            harmful=("selected_harm", "sum"),
            laundering=("selected_laundering", "sum"),
        )
        .assign(
            false_authorization_burden=lambda value: value["harmful"] / value["rows"],
            laundering_burden=lambda value: value["laundering"] / value["rows"],
        )
    )


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
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
        raise RuntimeError("v0.2 controlled decision source is not the registered output")
    columns = [
        "rule",
        "event_cluster_id",
        "row_id",
        "selected_harm",
        "selected_laundering",
    ]
    frame = pd.read_csv(source_path, usecols=columns)
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
    laundering_diff = left.loc[common, "laundering_burden"] - right.loc[common, "laundering_burden"]
    observed = {
        "clusters": len(common),
        "false_authorization_mean_difference": float(false_diff.mean()),
        "false_authorization_sd": float(false_diff.std(ddof=1)),
        "laundering_mean_difference": float(laundering_diff.mean()),
        "laundering_sd": float(laundering_diff.std(ddof=1)),
    }
    prereg_path = ROOT / "manifests" / "preregistration" / "external_orderbook_v03.yaml"
    prereg = load_config(prereg_path)
    expected_sd_false = float(prereg["power_contract"]["false_authorization_sd"])
    expected_sd_laundering = float(prereg["power_contract"]["laundering_sd"])
    if not np.isclose(observed["false_authorization_sd"], expected_sd_false, atol=1e-10):
        raise RuntimeError("frozen false-authorization SD does not match preregistration")
    if not np.isclose(observed["laundering_sd"], expected_sd_laundering, atol=1e-10):
        raise RuntimeError("frozen laundering SD does not match preregistration")
    false_effect = abs(float(config["primary_endpoints"]["false_authorization_burden"]["sesoi"]))
    laundering_effect = abs(float(config["primary_endpoints"]["laundering_burden"]["sesoi"]))
    min_power = float(config["primary_endpoints"]["minimum_power"])
    source_rows: dict[str, dict[str, object]] = {}
    for name, clusters in (
        ("binance", int(config["binance"]["paper_test_clusters"])),
        ("databento", int(config["databento"]["paper_test_clusters"])),
    ):
        false_power = two_sided_power(false_effect, expected_sd_false, clusters)
        laundering_power = two_sided_power(laundering_effect, expected_sd_laundering, clusters)
        source_rows[name] = {
            "clusters": clusters,
            "false_authorization_power": false_power,
            "laundering_power": laundering_power,
            "minimum_endpoint_power": min(false_power, laundering_power),
            "confirmatory_gate_passed": min(false_power, laundering_power) >= min_power,
            "classification": "confirmatory" if min(false_power, laundering_power) >= min_power else "descriptive",
        }
    report = {
        "project": "FinAuth-Audit",
        "version": "0.3.0",
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
            "false_authorization_burden": required_clusters(false_effect, expected_sd_false, min_power),
            "laundering_burden": required_clusters(laundering_effect, expected_sd_laundering, min_power),
        },
        "sources": source_rows,
        "gate_passed": bool(source_rows["binance"]["confirmatory_gate_passed"]),
        "claim_boundary": (
            "Pre-result design power derived exclusively from the frozen v0.2 paper test. "
            "No Binance or Databento outcome is read."
        ),
    }
    output_path = resolve_root_path(config["freeze"]["power_report"])
    write_json(output_path, report)
    if not report["gate_passed"]:
        raise RuntimeError("Binance confirmatory power gate did not pass")
    print(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute v0.3 pre-result power from frozen v0.2 only.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "external_orderbook_v03.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
