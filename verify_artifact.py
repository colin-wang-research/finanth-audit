#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

REAL_AGENT_V06_MODELS = frozenset(
    {"claude-sonnet-4-6", "gpt-5.4-mini", "gpt-5.5"}
)
REAL_AGENT_V06_TASKS = frozenset(
    {"directional_execution", "risk_limit_increase"}
)
REAL_AGENT_V06_RANK_RULES = (
    "Direct Prior",
    "Confidence Gate",
    "Uncertainty Gate",
    "Risk Filter",
    "Cost-Aware Gate",
    "Lifecycle Checklist",
)
REAL_AGENT_V06_PRIMARY_METRICS = (
    "coverage",
    "economic_loss_authorization_rate",
    "material_harm_authorization_rate",
    "tail_harm_authorization_rate",
    "risk_event_authorization_rate",
    "material_risk_event_authorization_rate",
    "authority_violation_rate",
    "normalized_task_utility",
    "missed_benign_opportunity",
    "review_rate",
)
REAL_AGENT_V06_AGGREGATE_OUTPUTS = frozenset(
    {
        "model_source_quality.csv",
        "rank_transfer_zero_shot.json",
        "recalibrated/bootstrap_bounds.csv",
        "recalibrated/metrics.csv",
        "recalibrated/subgroup_metrics.csv",
        "zero_shot/bootstrap_bounds.csv",
        "zero_shot/metrics.csv",
        "zero_shot/subgroup_metrics.csv",
    }
)
REAL_AGENT_V06_UNREAD_OUTPUTS = frozenset(
    {
        "recalibrated/decisions.csv",
        "summary.md",
        "zero_shot/decisions.csv",
    }
)
REAL_AGENT_V06_STRUCTURAL_NA_COLUMNS = (
    "clusters",
    "coverage",
    "execute_rate",
    "reduce_rate",
    "review_rate",
    "abstain_rate",
    "economic_loss_authorization_rate",
    "material_harm_authorization_rate",
    "tail_harm_authorization_rate",
    "risk_event_authorization_rate",
    "material_risk_event_authorization_rate",
    "authority_violation_rate",
    "normalized_task_utility",
    "raw_task_utility",
    "missed_benign_opportunity",
    "authorized_count",
    "directional_authorized_count",
    "risk_authorized_count",
    "benign_opportunity_count",
)


def append_check(
    checks: list[dict[str, object]], check: str, passed: bool, detail: object
) -> None:
    checks.append({"check": check, "passed": bool(passed), "detail": detail})


def is_read_only(path: Path) -> bool:
    return path.is_file() and path.stat().st_mode & 0o222 == 0


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def safe_child(root: Path, relative: object) -> Path | None:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return root / candidate


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def derive_rank_primary_support(rank: dict[str, object]) -> bool | None:
    exact = rank.get("exact_permutation")
    bootstrap = rank.get("date_cluster_bootstrap")
    if not isinstance(exact, dict) or not isinstance(bootstrap, dict):
        return None
    exact_p = exact.get("exact_p_value")
    probability = bootstrap.get("probability_below_zero")
    if exact_p is None or probability is None:
        return None
    return bool(float(probability) >= 0.95 and float(exact_p) <= 0.05)


def find_real_agent_v06_secret_paths(repo: Path) -> list[str]:
    ignored_directories = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    suspicious: list[str] = []
    for current, directories, filenames in os.walk(repo, followlinks=False):
        directories[:] = [
            name for name in directories if name not in ignored_directories
        ]
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(repo)
            normalized = filename.lower().replace("-", "_")
            suffix = path.suffix.lower()
            hidden_plaintext = (
                normalized.startswith("hidden_proposals.")
                and suffix != ".fernet"
            ) or normalized.startswith("community_hidden_proposals.")
            hidden_plaintext = hidden_plaintext or any(
                marker in normalized
                for marker in ("hidden_plaintext", "decrypted_hidden")
            )
            key_material = suffix in {".key", ".pem", ".p12", ".pfx"} or any(
                marker in normalized
                for marker in ("private_key", "secret_key", "fernet_key")
            )
            if hidden_plaintext or key_material:
                suspicious.append(relative.as_posix())
    return sorted(suspicious)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_upstream(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def verify_hash(path: Path, expected: str, checks: list[dict[str, object]]) -> None:
    if not path.exists():
        checks.append({"check": str(path), "passed": False, "detail": "missing"})
        return
    actual = sha256(path)
    checks.append(
        {
            "check": str(path),
            "passed": actual == expected,
            "detail": f"expected={expected} actual={actual}",
        }
    )


def record_observed_hash(path: Path, initial: str, checks: list[dict[str, object]]) -> None:
    if not path.exists():
        checks.append({"check": str(path), "passed": False, "detail": "missing record-only upstream"})
        return
    checks.append(
        {
            "check": f"record-only upstream {path}",
            "passed": True,
            "detail": f"initial={initial} current={sha256(path)}",
        }
    )


def verify_manifest_hashes(manifest_path: Path, checks: list[dict[str, object]]) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in payload.get("outputs", {}).items():
        verify_hash(manifest_path.parent / name, expected, checks)


def run_tests(checks: list[dict[str, object]]) -> None:
    result = subprocess.run(
        [str(REPO / ".venv" / "bin" / "pytest"), "-q", str(ROOT / "tests")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    checks.append(
        {
            "check": "finauth_audit tests",
            "passed": result.returncode == 0,
            "detail": (result.stdout + result.stderr).strip(),
        }
    )


def verify_smoke(checks: list[dict[str, object]]) -> None:
    data_manifest_path = ROOT / "manifests" / "smoke_data_manifest.json"
    result_manifest_path = ROOT / "results" / "smoke" / "manifest.json"
    shortcut_manifest_path = ROOT / "results" / "smoke" / "shortcut_manifest.json"
    for path in (data_manifest_path, result_manifest_path, shortcut_manifest_path):
        checks.append({"check": str(path), "passed": path.exists(), "detail": "exists" if path.exists() else "missing"})
    if not all(path.exists() for path in (data_manifest_path, result_manifest_path, shortcut_manifest_path)):
        return

    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_path = ROOT / data_manifest["data_path"]
    verify_hash(data_path, data_manifest["data_sha256"], checks)
    verify_hash(Path(data_manifest["config_path"]), data_manifest["config_sha256"], checks)
    verify_manifest_hashes(result_manifest_path, checks)
    verify_manifest_hashes(shortcut_manifest_path, checks)

    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "smoke remains non-confirmatory validation",
            "passed": result_manifest.get("evaluation_split") == "validation"
            and result_manifest.get("confirmatory") is False,
            "detail": f"split={result_manifest.get('evaluation_split')} confirmatory={result_manifest.get('confirmatory')}",
        }
    )
    checks.append(
        {
            "check": "cluster bootstrap protocol",
            "passed": result_manifest.get("bootstrap_unit") == "event_cluster_id"
            and result_manifest.get("bootstrap_replicates") == 2000,
            "detail": f"unit={result_manifest.get('bootstrap_unit')} reps={result_manifest.get('bootstrap_replicates')}",
        }
    )

    metrics = pd.read_csv(ROOT / "results" / "smoke" / "metrics.csv")
    no_action = metrics[(metrics["rule"] == "No Action") & (metrics["profile"] == "overall")].iloc[0]
    checks.append(
        {
            "check": "zero action is N/A risk",
            "passed": no_action["coverage"] == 0.0 and pd.isna(no_action["far"]) and pd.isna(no_action["alr"]),
            "detail": f"coverage={no_action['coverage']} far={no_action['far']} alr={no_action['alr']}",
        }
    )

    certification = pd.read_csv(ROOT / "results" / "smoke" / "certification_summary.csv")
    no_action_volume = certification.loc[
        certification["rule"] == "No Action", "worst_profile_certification_volume"
    ].iloc[0]
    checks.append(
        {
            "check": "zero action cannot certify",
            "passed": no_action_volume == 0.0,
            "detail": f"volume={no_action_volume}",
        }
    )

    opportunity = pd.read_csv(ROOT / "results" / "smoke" / "opportunity_by_slice.csv")
    control = opportunity[opportunity["opportunity_slice"] == "no_opportunity_control"].iloc[0]
    scored = opportunity[opportunity["certification_eligible"]]
    checks.append(
        {
            "check": "opportunity controls",
            "passed": control["oracle_positive_rate"] == 0.0
            and scored["oracle_positive_rate"].between(0.10, 0.80).all(),
            "detail": f"control={control['oracle_positive_rate']} scored_min={scored['oracle_positive_rate'].min()} scored_max={scored['oracle_positive_rate'].max()}",
        }
    )

    access = pd.read_csv(ROOT / "results" / "smoke" / "feature_access_audit.csv")
    checks.append(
        {
            "check": "feature access",
            "passed": (access["status"] == "VALID").all(),
            "detail": f"valid={(access['status'] == 'VALID').sum()} total={len(access)}",
        }
    )
    structural = json.loads(
        (ROOT / "results" / "smoke" / "shortcut_structural_audit.json").read_text(encoding="utf-8")
    )
    checks.append(
        {
            "check": "shortcut structural audit",
            "passed": not structural["cluster_split_leakage"]
            and not structural["future_decoy_used_by_deployable_rule"]
            and not structural["identifier_used_by_deployable_rule"],
            "detail": json.dumps(structural, sort_keys=True),
        }
    )


def verify_provenance(checks: list[dict[str, object]]) -> None:
    data_manifest_path = ROOT / "manifests" / "provenance_smoke_manifest.json"
    result_manifest_path = ROOT / "results" / "provenance_smoke" / "manifest.json"
    for path in (data_manifest_path, result_manifest_path):
        checks.append(
            {"check": str(path), "passed": path.exists(), "detail": "exists" if path.exists() else "missing"}
        )
    if not data_manifest_path.exists() or not result_manifest_path.exists():
        return

    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_path = ROOT / "data" / "provenance_laundering" / "smoke.csv"
    verify_hash(data_path, data_manifest["data_sha256"], checks)
    verify_manifest_hashes(result_manifest_path, checks)
    attacks = set(data_manifest["attack_counts"])
    checks.append(
        {
            "check": "provenance attack coverage",
            "passed": attacks == {"clean", "role_noise", "delegation", "paraphrase", "multi_hop"},
            "detail": sorted(attacks),
        }
    )
    noise_rates = {float(value) for value in data_manifest["role_noise_counts"]}
    hop_depths = {int(float(value)) for value in data_manifest["hop_depth_counts"]}
    checks.append(
        {
            "check": "role noise grid",
            "passed": noise_rates == {0.0, 0.05, 0.10, 0.20, 0.40},
            "detail": sorted(noise_rates),
        }
    )
    checks.append(
        {
            "check": "hop depth grid",
            "passed": hop_depths == {1, 2, 3, 4, 5},
            "detail": sorted(hop_depths),
        }
    )

    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "provenance smoke remains validation-only",
            "passed": result_manifest.get("evaluation_split") == "validation"
            and result_manifest.get("confirmatory") is False,
            "detail": f"split={result_manifest.get('evaluation_split')} confirmatory={result_manifest.get('confirmatory')}",
        }
    )
    access = pd.read_csv(ROOT / "results" / "provenance_smoke" / "feature_access_audit.csv")
    checks.append(
        {
            "check": "provenance feature access",
            "passed": (access["status"] == "VALID").all(),
            "detail": f"valid={(access['status'] == 'VALID').sum()} total={len(access)}",
        }
    )
    traceability = pd.read_csv(ROOT / "results" / "provenance_smoke" / "by_traceability.csv")
    checks.append(
        {
            "check": "traceability strata reported separately",
            "passed": set(traceability["traceability"]) == {"traceable", "untraceable"},
            "detail": sorted(set(traceability["traceability"])),
        }
    )
    summary = pd.read_csv(ROOT / "results" / "provenance_smoke" / "summary.csv")
    no_action = summary[summary["rule"] == "No Action"].iloc[0]
    checks.append(
        {
            "check": "provenance zero action is N/A risk",
            "passed": no_action["coverage"] == 0.0 and pd.isna(no_action["far"]) and pd.isna(no_action["alr"]),
            "detail": f"coverage={no_action['coverage']} far={no_action['far']} alr={no_action['alr']}",
        }
    )
    feature_manifest = json.loads((ROOT / "manifests" / "feature_access.json").read_text(encoding="utf-8"))
    provenance_legal = set(feature_manifest["tasks"]["provenance"]["legal"])
    checks.append(
        {
            "check": "hidden original role remains illegal",
            "passed": "original_verified_role" not in provenance_legal,
            "detail": f"legal_count={len(provenance_legal)}",
        }
    )
    checks.append(
        {
            "check": "provenance cluster bootstrap protocol",
            "passed": result_manifest.get("bootstrap_unit") == "event_cluster_id"
            and result_manifest.get("bootstrap_replicates") == 2000,
            "detail": f"unit={result_manifest.get('bootstrap_unit')} reps={result_manifest.get('bootstrap_replicates')}",
        }
    )

    def trace_row(rule: str, stratum: str) -> pd.Series:
        selected = traceability[
            (traceability["rule"] == rule) & (traceability["traceability"] == stratum)
        ]
        if len(selected) != 1:
            raise AssertionError(f"expected one traceability row for {rule}/{stratum}, found {len(selected)}")
        return selected.iloc[0]

    provenance_hard_traceable = trace_row("Provenance Hard Gate", "traceable")
    provenance_hard_untraceable = trace_row("Provenance Hard Gate", "untraceable")
    checks.append(
        {
            "check": "provenance hard gate eliminates traceable laundering",
            "passed": provenance_hard_traceable["coverage"] > 0.0
            and provenance_hard_traceable["alr"] == 0.0,
            "detail": (
                f"coverage={provenance_hard_traceable['coverage']} "
                f"alr={provenance_hard_traceable['alr']}"
            ),
        }
    )
    checks.append(
        {
            "check": "untraceable hard-gate abstention is N/A risk",
            "passed": provenance_hard_untraceable["coverage"] == 0.0
            and pd.isna(provenance_hard_untraceable["alr"])
            and provenance_hard_untraceable["false_block_rate"] == 1.0,
            "detail": (
                f"coverage={provenance_hard_untraceable['coverage']} "
                f"alr={provenance_hard_untraceable['alr']} "
                f"false_block={provenance_hard_untraceable['false_block_rate']}"
            ),
        }
    )

    indexed = summary.set_index("rule")
    hard_role = indexed.loc["Hard Role Gate"]
    no_role = indexed.loc["No Role Gate"]
    learned = indexed.loc["Provenance Learned Gate"]
    checks.append(
        {
            "check": "current-role gate does not solve indirect laundering",
            "passed": hard_role["direct_leakage_rate"] == 0.0
            and hard_role["indirect_leakage_rate"] > 0.0
            and hard_role["alr"] > 0.0,
            "detail": (
                f"direct={hard_role['direct_leakage_rate']} "
                f"indirect={hard_role['indirect_leakage_rate']} alr={hard_role['alr']}"
            ),
        }
    )
    checks.append(
        {
            "check": "no-role gate exposes direct and indirect laundering",
            "passed": no_role["direct_leakage_rate"] > 0.0
            and no_role["indirect_leakage_rate"] > 0.0,
            "detail": (
                f"direct={no_role['direct_leakage_rate']} "
                f"indirect={no_role['indirect_leakage_rate']}"
            ),
        }
    )
    learned_untraceable = trace_row("Provenance Learned Gate", "untraceable")
    checks.append(
        {
            "check": "learned provenance gate exposes delegation-leakage trade-off",
            "passed": learned["alr"] > 0.0
            and learned["alr"] < hard_role["alr"]
            and learned_untraceable["safe_delegation_coverage"] > 0.0
            and learned_untraceable["coverage"] > 0.0,
            "detail": (
                f"overall_alr={learned['alr']} hard_role_alr={hard_role['alr']} "
                f"untraceable_coverage={learned_untraceable['coverage']} "
                f"untraceable_safe_delegation={learned_untraceable['safe_delegation_coverage']}"
            ),
        }
    )


def verify_public(checks: list[dict[str, object]]) -> None:
    fetch_manifest_path = ROOT / "manifests" / "public_polymarket_fetch.json"
    dataset_manifest_path = ROOT / "manifests" / "public_polymarket_dataset.json"
    results_dir = ROOT / "results" / "public_audit"
    validation_manifest_path = results_dir / "public_validation_manifest.json"
    power_gate_path = results_dir / "public_power_gate.json"
    required = (
        fetch_manifest_path,
        dataset_manifest_path,
        validation_manifest_path,
        power_gate_path,
        results_dir / "point_in_time_checks.csv",
        results_dir / "source_classification.csv",
        results_dir / "feature_access_audit.csv",
    )
    for path in required:
        checks.append(
            {
                "check": str(path),
                "passed": path.exists(),
                "detail": "exists" if path.exists() else "missing",
            }
        )
    if not all(path.exists() for path in required):
        return

    fetch_manifest = json.loads(fetch_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    for relative, expected in fetch_manifest["outputs"].items():
        verify_hash(ROOT / relative, expected, checks)
    for relative, expected in dataset_manifest["outputs"].items():
        verify_hash(ROOT / relative, expected, checks)
    for relative, expected in dataset_manifest["inputs"].items():
        verify_hash(ROOT / relative, expected, checks)
    checks.append(
        {
            "check": "public fetch has no API or window errors",
            "passed": int(fetch_manifest.get("window_error_count", -1)) == 0
            and len(fetch_manifest.get("errors", [])) == 0
            and len(fetch_manifest.get("truncated_window_audit_errors", [])) == 0,
            "detail": (
                f"window_errors={fetch_manifest.get('window_error_count')} "
                f"request_errors={len(fetch_manifest.get('errors', []))}"
            ),
        }
    )
    checks.append(
        {
            "check": "public truncated windows are explicitly enumerated",
            "passed": len(fetch_manifest.get("truncated_window_names", []))
            == int(fetch_manifest.get("truncated_windows", -1)),
            "detail": (
                f"count={fetch_manifest.get('truncated_windows')} "
                f"listed={len(fetch_manifest.get('truncated_window_names', []))}"
            ),
        }
    )
    checks.append(
        {
            "check": "public history suffices for frozen dataset target",
            "passed": int(fetch_manifest.get("history_tokens_with_minimum_points", 0))
            >= int(dataset_manifest.get("event_clusters", 0)),
            "detail": (
                f"history={fetch_manifest.get('history_tokens_with_minimum_points')} "
                f"clusters={dataset_manifest.get('event_clusters')}"
            ),
        }
    )

    dataset_path = ROOT / "data" / "public_replay" / "polymarket" / "point_in_time.csv"
    frame = pd.read_csv(dataset_path, low_memory=False)
    source = pd.to_datetime(frame["source_timestamp"], utc=True, errors="coerce", format="mixed")
    decision = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce", format="mixed")
    action = pd.to_datetime(frame["action_timestamp"], utc=True, errors="coerce", format="mixed")
    outcome = pd.to_datetime(frame["outcome_timestamp"], utc=True, errors="coerce", format="mixed")
    ordered = (source < decision) & (decision < action) & (action < outcome)
    checks.append(
        {
            "check": "public timestamp ordering",
            "passed": bool(ordered.all()),
            "detail": f"violations={int((~ordered).sum())}",
        }
    )
    cluster_splits = frame.groupby("event_cluster_id")["split"].nunique()
    checks.append(
        {
            "check": "public event clusters and splits",
            "passed": frame["event_cluster_id"].nunique() == len(frame)
            and int(frame.groupby("event_cluster_id").size().max()) == 1
            and bool((cluster_splits == 1).all()),
            "detail": f"rows={len(frame)} clusters={frame['event_cluster_id'].nunique()}",
        }
    )
    allowed_sources = {"clob_prices_history", "data_api_trade"}
    checks.append(
        {
            "check": "public historical provenance and omitted microstructure",
            "passed": set(frame["historical_probability_source"].astype(str)) <= allowed_sources
            and "spread" not in frame.columns
            and "depth" not in frame.columns
            and not frame["historical_spread_observed"].astype(bool).any()
            and not frame["historical_depth_observed"].astype(bool).any(),
            "detail": f"sources={sorted(set(frame['historical_probability_source'].astype(str)))}",
        }
    )
    checks.append(
        {
            "check": "public series concentration bound",
            "passed": float(dataset_manifest.get("max_series_share", 1.0)) <= 0.10,
            "detail": f"max_share={dataset_manifest.get('max_series_share')}",
        }
    )
    point_checks = pd.read_csv(results_dir / "point_in_time_checks.csv")
    checks.append(
        {
            "check": "point-in-time audit",
            "passed": bool(point_checks["passed"].all()),
            "detail": f"passed={int(point_checks['passed'].sum())}/{len(point_checks)}",
        }
    )
    classifications = pd.read_csv(results_dir / "source_classification.csv")
    legacy = classifications[classifications["source"] != "polymarket_point_in_time"]
    checks.append(
        {
            "check": "legacy public layers remain non-confirmatory",
            "passed": not legacy["confirmatory_eligible"].astype(bool).any(),
            "detail": ", ".join(
                f"{row.source}={row.classification}" for row in legacy.itertuples()
            ),
        }
    )
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    access = pd.read_csv(results_dir / "feature_access_audit.csv")
    checks.append(
        {
            "check": "public evaluation and feature access remain validation-only",
            "passed": validation_manifest.get("evaluation_split") == "validation"
            and validation_manifest.get("confirmatory") is False
            and bool((access["status"] == "VALID").all()),
            "detail": f"rules={len(access)}",
        }
    )
    power = json.loads(power_gate_path.read_text(encoding="utf-8"))
    expected = "confirmatory_eligible" if all(power["conditions"].values()) else "exploratory_only"
    checks.append(
        {
            "check": "public power gate is internally consistent",
            "passed": power.get("classification") == expected
            and power.get("passed") == all(power["conditions"].values())
            and power.get("confirmatory_test_evaluated") is False,
            "detail": f"classification={power.get('classification')} power={power.get('estimated_power')}",
        }
    )


def verify_training(checks: list[dict[str, object]]) -> None:
    corpora_manifest_path = ROOT / "manifests" / "training_utility_smoke_manifest.json"
    results_dir = ROOT / "results" / "training_utility_smoke"
    results_manifest_path = results_dir / "manifest.json"
    required = (
        corpora_manifest_path,
        results_manifest_path,
        ROOT / "docs" / "training_utility_report.md",
        results_dir / "primary_endpoint_by_seed.csv",
        results_dir / "holm_validation_contrasts.csv",
        results_dir / "feature_ablation_aggregate.csv",
        results_dir / "negative_control_aggregate.csv",
        results_dir / "negative_control_primary_by_seed.csv",
    )
    for path in required:
        checks.append(
            {
                "check": str(path),
                "passed": path.exists(),
                "detail": "exists" if path.exists() else "missing",
            }
        )
    if not all(path.exists() for path in required):
        return
    feature_manifest = json.loads(
        (ROOT / "manifests" / "feature_access.json").read_text(encoding="utf-8")
    )
    legal = set(feature_manifest["tasks"]["training_utility"]["legal"])
    forbidden = set(feature_manifest["global_forbidden"])
    corpora = json.loads(corpora_manifest_path.read_text(encoding="utf-8"))
    for relative, expected in corpora["files"].items():
        verify_hash(ROOT / relative, expected, checks)
    records = pd.DataFrame(corpora["records"])
    checks.append(
        {
            "check": "D0-D7 matched training budgets",
            "passed": records["rows"].nunique() == 1
            and records["clusters"].nunique() == 1
            and int(records["rows"].iloc[0])
            == int(corpora["training_clusters_per_variant"]),
            "detail": f"rows={sorted(records['rows'].unique())} clusters={sorted(records['clusters'].unique())}",
        }
    )
    checks.append(
        {
            "check": "training features match legal manifest",
            "passed": set(corpora["legal_features"]) == legal
            and not (set(corpora["legal_features"]) & forbidden),
            "detail": f"features={len(corpora['legal_features'])}",
        }
    )
    holdout_slices = set(corpora["controlled_test_holdouts"])
    holdout_attacks = set(corpora["provenance_test_holdouts"])
    leaked_slices: set[str] = set()
    leaked_attacks: set[str] = set()
    for record in corpora["records"]:
        leaked_slices |= set(record.get("opportunity_slices", {})) & holdout_slices
        leaked_attacks |= set(record.get("attack_types", {})) & holdout_attacks
    checks.append(
        {
            "check": "training corpora exclude frozen test mechanisms",
            "passed": not leaked_slices and not leaked_attacks,
            "detail": f"slices={sorted(leaked_slices)} attacks={sorted(leaked_attacks)}",
        }
    )
    checks.append(
        {
            "check": "training corpora preserve test sealing",
            "passed": corpora.get("confirmatory") is False
            and corpora.get("test_outcomes_evaluated") is False,
            "detail": f"test={corpora.get('test_outcomes_evaluated')}",
        }
    )
    results = json.loads(results_manifest_path.read_text(encoding="utf-8"))
    for name, expected in results["outputs"].items():
        verify_hash(results_dir / name, expected, checks)
    for relative, expected in results["inputs"].items():
        verify_hash(ROOT / relative, expected, checks)
    for relative, expected in results.get("docs_report", {}).items():
        verify_hash(ROOT / relative, expected, checks)
    checks.append(
        {
            "check": "training evaluation remains validation-only",
            "passed": results.get("evaluation_split") == "validation"
            and results.get("confirmatory") is False
            and results.get("test_outcomes_evaluated") is False,
            "detail": f"split={results.get('evaluation_split')} test={results.get('test_outcomes_evaluated')}",
        }
    )
    checks.append(
        {
            "check": "training primary registry is frozen",
            "passed": len(results.get("primary_evaluations", [])) == 4
            and float(results.get("min_primary_mechanism_coverage", -1)) == 0.05
            and results.get("prediction_diagnostic_variant") == "D0"
            and results.get("holm_reference_variant") == "D1"
            and len(results.get("holm_family", [])) == 6
            and results.get("secondary_learners_status")
            == "not_registered_in_v0.2.0_validation_smoke"
            and results.get("negative_controls") == ["D7_role_neutral"]
            and results.get("negative_controls_in_holm_family") is False,
            "detail": (
                f"mechanisms={results.get('primary_evaluations')} "
                f"reference={results.get('holm_reference_variant')} "
                f"family={results.get('holm_family')}"
            ),
        }
    )
    primary = pd.read_csv(results_dir / "primary_endpoint_by_seed.csv")
    invalid = ~primary["primary_endpoint_valid"].astype(bool)
    checks.append(
        {
            "check": "invalid primary endpoints remain N/A",
            "passed": bool(primary.loc[invalid, "primary_far"].isna().all()),
            "detail": f"invalid={int(invalid.sum())}",
        }
    )
    holm = pd.read_csv(results_dir / "holm_validation_contrasts.csv")
    invalid_variants = set(
        primary.groupby("variant")["primary_endpoint_valid"].all().loc[lambda values: ~values].index
    )
    invalid_holm = holm[holm["variant"].isin(invalid_variants)]
    checks.append(
        {
            "check": "coverage-collapsed variants cannot win Holm contrasts",
            "passed": not invalid_holm["reject"].astype(bool).any(),
            "detail": f"invalid_variants={sorted(invalid_variants)}",
        }
    )


def _decision_splits(path: Path) -> set[str]:
    values: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["split"], chunksize=100_000):
        values.update(chunk["split"].dropna().astype(str).unique())
    return values


def verify_paper_test_protocol(
    checks: list[dict[str, object]],
    require_frozen: bool = False,
    require_complete: bool = False,
) -> None:
    partition_path = ROOT / "manifests" / "paper_test_partition" / "manifest.json"
    rehearsal_path = ROOT / "results" / "paper_test_rehearsal" / "manifest.json"
    for path in (partition_path, rehearsal_path):
        checks.append(
            {
                "check": str(path),
                "passed": path.exists(),
                "detail": "exists" if path.exists() else "missing",
            }
        )
    if not partition_path.exists() or not rehearsal_path.exists():
        return

    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    for relative, expected in partition["files"].items():
        verify_hash(ROOT / relative, expected, checks)
    for relative, expected in partition["source_files"].items():
        verify_hash(ROOT / relative, expected, checks)
    verify_hash(
        ROOT / "evaluation" / "paper_test_partition.py",
        partition["partition_code_sha256"],
        checks,
    )
    paper_clusters = pd.read_csv(
        ROOT / "manifests" / "paper_test_partition" / "paper_test_clusters.csv"
    )["event_cluster_id"].astype(str)
    community_clusters = pd.read_csv(
        ROOT / "manifests" / "paper_test_partition" / "community_hidden_clusters.csv"
    )["event_cluster_id"].astype(str)
    overlap = set(paper_clusters) & set(community_clusters)
    checks.append(
        {
            "check": "paper/community cluster partition is exact and disjoint",
            "passed": len(paper_clusters) == 5_000
            and len(community_clusters) == 5_000
            and not overlap
            and partition.get("paper_test_clusters") == 5_000
            and partition.get("community_hidden_clusters") == 5_000,
            "detail": (
                f"paper={len(paper_clusters)} community={len(community_clusters)} "
                f"overlap={len(overlap)}"
            ),
        }
    )
    checks.append(
        {
            "check": "partition assignment used identifiers only",
            "passed": partition.get("partition_fields_inspected")
            == ["event_cluster_id", "split"]
            and partition.get("outcome_fields_inspected") is False
            and partition.get("opaque_rows_copied_without_semantic_field_access")
            is True,
            "detail": (
                f"fields={partition.get('partition_fields_inspected')} "
                f"outcomes={partition.get('outcome_fields_inspected')}"
            ),
        }
    )
    paper_controlled = pd.read_csv(
        ROOT / "data" / "paper_test" / "controlled.csv",
        usecols=["event_cluster_id", "split"],
    )
    paper_provenance = pd.read_csv(
        ROOT / "data" / "paper_test" / "provenance.csv",
        usecols=["event_cluster_id", "split"],
    )
    checks.append(
        {
            "check": "paper-test data contain only assigned test clusters",
            "passed": set(paper_controlled["split"]) == {"test"}
            and set(paper_provenance["split"]) == {"test"}
            and set(paper_controlled["event_cluster_id"].astype(str))
            == set(paper_clusters)
            and set(paper_provenance["event_cluster_id"].astype(str))
            == set(paper_clusters),
            "detail": (
                f"controlled_rows={len(paper_controlled)} "
                f"provenance_rows={len(paper_provenance)}"
            ),
        }
    )

    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    verify_manifest_hashes(rehearsal_path, checks)
    checks.append(
        {
            "check": "paper-test evaluator validation rehearsal is exact and sealed",
            "passed": rehearsal.get("equivalence_passed") is True
            and rehearsal.get("evaluation_split") == "validation"
            and rehearsal.get("test_outcomes_evaluated") is False
            and rehearsal.get("community_hidden_outcomes_evaluated") is False,
            "detail": (
                f"equivalence={rehearsal.get('equivalence_passed')} "
                f"split={rehearsal.get('evaluation_split')}"
            ),
        }
    )

    freeze_path = ROOT / "manifests" / "paper_test_freeze.json"
    checks.append(
        {
            "check": str(freeze_path),
            "passed": freeze_path.exists() or not require_frozen,
            "detail": "exists" if freeze_path.exists() else "not yet required",
        }
    )
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        archive_path = ROOT / freeze["archive_path"]
        verify_hash(archive_path, freeze["archive_sha256"], checks)
        for relative, expected in freeze["surface_hashes"].items():
            verify_hash(ROOT / relative, expected, checks)
        checks.append(
            {
                "check": "paper-test freeze remains active and pre-inspection",
                "passed": freeze.get("status") == "FROZEN_BEFORE_PAPER_TEST"
                and freeze.get("paper_test_outcomes_evaluated") is False
                and freeze.get("community_hidden_outcomes_evaluated") is False,
                "detail": (
                    f"status={freeze.get('status')} "
                    f"paper={freeze.get('paper_test_outcomes_evaluated')} "
                    f"community={freeze.get('community_hidden_outcomes_evaluated')}"
                ),
            }
        )

    registry_path = ROOT / "manifests" / "paper_test_registry.json"
    checks.append(
        {
            "check": str(registry_path),
            "passed": registry_path.exists() or not require_complete,
            "detail": "exists" if registry_path.exists() else "not yet required",
        }
    )
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        checks.append(
            {
                "check": "one-time paper-test registry is complete and community sealed",
                "passed": registry.get("status")
                in {"COMPLETE_UNINSPECTED", "COMPLETE_INSPECTED"}
                and registry.get("community_hidden_outcomes_evaluated") is False,
                "detail": (
                    f"status={registry.get('status')} "
                    f"community={registry.get('community_hidden_outcomes_evaluated')}"
                ),
            }
        )
        result_manifest_path = ROOT / registry["result_manifest"]
        verify_hash(result_manifest_path, registry["result_manifest_sha256"], checks)
        result_manifest = json.loads(
            result_manifest_path.read_text(encoding="utf-8")
        )
        for relative, expected in result_manifest["outputs"].items():
            verify_hash(result_manifest_path.parent / relative, expected, checks)
        checks.append(
            {
                "check": "paper-test result scale and access boundary",
                "passed": result_manifest.get("evaluation_split") == "paper_test"
                and result_manifest.get("confirmatory") is True
                and result_manifest.get("controlled_clusters") == 5_000
                and result_manifest.get("provenance_clusters") == 5_000
                and result_manifest.get("community_hidden_outcomes_evaluated")
                is False,
                "detail": (
                    f"controlled={result_manifest.get('controlled_clusters')} "
                    f"provenance={result_manifest.get('provenance_clusters')} "
                    f"community={result_manifest.get('community_hidden_outcomes_evaluated')}"
                ),
            }
        )


def verify_main(checks: list[dict[str, object]]) -> None:
    controlled_manifest_path = ROOT / "manifests" / "main_data_manifest.json"
    controlled_results_manifest_path = ROOT / "results" / "main" / "manifest.json"
    shortcut_manifest_path = ROOT / "results" / "main" / "shortcut_manifest.json"
    provenance_manifest_path = ROOT / "manifests" / "provenance_main_manifest.json"
    provenance_results_manifest_path = (
        ROOT / "results" / "provenance_main" / "manifest.json"
    )
    summary_manifest_path = ROOT / "results" / "main_validation" / "manifest.json"
    robustness_manifest_paths = (
        ROOT / "results" / "certification_robustness" / "manifest.json",
        ROOT / "results" / "review_workload" / "manifest.json",
        ROOT / "results" / "baseline_governance" / "manifest.json",
        ROOT / "results" / "generator_robustness" / "manifest.json",
        ROOT / "results" / "provenance_identifiability" / "manifest.json",
        ROOT / "results" / "training_robustness_v03" / "manifest.json",
    )
    generator_manifest_path = ROOT / "manifests" / "generator_robustness_manifest.json"
    required = (
        controlled_manifest_path,
        controlled_results_manifest_path,
        shortcut_manifest_path,
        provenance_manifest_path,
        provenance_results_manifest_path,
        summary_manifest_path,
        generator_manifest_path,
        *robustness_manifest_paths,
    )
    for path in required:
        checks.append(
            {
                "check": str(path),
                "passed": path.exists(),
                "detail": "exists" if path.exists() else "missing",
            }
        )
    if not all(path.exists() for path in required):
        return

    controlled = json.loads(controlled_manifest_path.read_text(encoding="utf-8"))
    controlled_data = ROOT / controlled["data_path"]
    verify_hash(controlled_data, controlled["data_sha256"], checks)
    verify_hash(Path(controlled["config_path"]), controlled["config_sha256"], checks)
    checks.append(
        {
            "check": "main controlled scale and split freeze",
            "passed": controlled.get("rows") == 200_000
            and controlled.get("clusters") == 50_000
            and controlled.get("split_counts")
            == {"test": 40_000, "train": 120_000, "validation": 40_000},
            "detail": (
                f"rows={controlled.get('rows')} clusters={controlled.get('clusters')} "
                f"splits={controlled.get('split_counts')}"
            ),
        }
    )

    for manifest_path in robustness_manifest_paths:
        verify_manifest_hashes(manifest_path, checks)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks.append(
            {
                "check": f"Round 75 robustness remains validation-only: {manifest_path.parent.name}",
                "passed": manifest.get("evaluation_split") == "validation"
                and manifest.get("test_outcomes_evaluated") is False,
                "detail": (
                    f"split={manifest.get('evaluation_split')} "
                    f"test={manifest.get('test_outcomes_evaluated')}"
                ),
            }
        )

    generator_manifest = json.loads(
        generator_manifest_path.read_text(encoding="utf-8")
    )
    verify_hash(
        ROOT / "configs" / "generator_robustness.yaml",
        generator_manifest["config_sha256"],
        checks,
    )
    for family, record in generator_manifest["families"].items():
        verify_hash(ROOT / record["data_path"], record["data_sha256"], checks)
        checks.append(
            {
                "check": f"prospective generator scale and split: {family}",
                "passed": record["rows"] == 48_000
                and record["clusters"] == 12_000
                and record["split_counts"]
                == {"test": 9_600, "train": 28_800, "validation": 9_600},
                "detail": (
                    f"rows={record['rows']} clusters={record['clusters']} "
                    f"splits={record['split_counts']}"
                ),
            }
        )
    checks.append(
        {
            "check": "prospective generators preserve test sealing",
            "passed": generator_manifest.get("evaluation_split") == "validation"
            and generator_manifest.get("test_outcomes_evaluated") is False,
            "detail": (
                f"split={generator_manifest.get('evaluation_split')} "
                f"test={generator_manifest.get('test_outcomes_evaluated')}"
            ),
        }
    )

    training_robustness = json.loads(
        (ROOT / "results" / "training_robustness_v03" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for relative, expected in training_robustness.get("inputs", {}).items():
        verify_hash(ROOT / relative, expected, checks)
    for relative, expected in training_robustness.get("docs_report", {}).items():
        verify_hash(ROOT / relative, expected, checks)
    checks.append(
        {
            "check": "prospective RQ4 preserves the registered v0.2 primary",
            "passed": training_robustness.get("registered_primary_preserved")
            == "configs/training_utility_smoke.yaml"
            and training_robustness.get("registered_primary_preserved_sha256")
            == sha256(ROOT / "configs" / "training_utility_smoke.yaml")
            and training_robustness.get("min_primary_mechanism_coverage") == 0.05,
            "detail": (
                f"primary={training_robustness.get('registered_primary_preserved')} "
                f"coverage_floor={training_robustness.get('min_primary_mechanism_coverage')}"
            ),
        }
    )
    checks.append(
        {
            "check": "prospective RQ4 learner and budget matrix is complete",
            "passed": training_robustness.get("budgets") == [400, 1000, 5000, 20000]
            and len(training_robustness.get("learners", [])) == 4
            and len(training_robustness.get("curricula", {})) == 3
            and training_robustness.get("fit_count") == 144,
            "detail": (
                f"budgets={training_robustness.get('budgets')} "
                f"learners={training_robustness.get('learners')} "
                f"fit_count={training_robustness.get('fit_count')}"
            ),
        }
    )
    checks.append(
        {
            "check": "prospective RQ4 figure text QA passes",
            "passed": all(
                training_robustness.get("figure_text_qa", {}).get(key) == 0
                for key in (
                    "all_text_overlap_count",
                    "qa_label_point_overlap_count",
                    "qa_label_boundary_violation_count",
                    "legend_axes_overlap_count",
                )
            )
            and training_robustness.get("figure_text_qa", {}).get(
                "minimum_font_points", 0
            )
            >= 8.0,
            "detail": str(training_robustness.get("figure_text_qa")),
        }
    )

    governance_splits = _decision_splits(
        ROOT / "results" / "baseline_governance" / "decisions.csv"
    )
    checks.append(
        {
            "check": "baseline-governance decisions exclude train and test rows",
            "passed": governance_splits == {"validation"},
            "detail": f"splits={sorted(governance_splits)}",
        }
    )
    robustness = json.loads(robustness_manifest_paths[0].read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "certification robustness profiles frozen before paper test",
            "passed": len(robustness.get("policy_grids", {})) == 5
            and robustness.get("continuous_reference")
            == {"far": 1.0, "alr": 0.25, "coverage": 0.0},
            "detail": (
                f"profiles={sorted(robustness.get('policy_grids', {}))} "
                f"reference={robustness.get('continuous_reference')}"
            ),
        }
    )
    review = json.loads(robustness_manifest_paths[1].read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "review workload scenarios are frozen diagnostics",
            "passed": review.get("capacity_fractions") == [0.05, 0.1, 0.2, 0.4]
            and review.get("review_cost_bps") == [0.5, 1.0, 2.0, 5.0],
            "detail": (
                f"capacity={review.get('capacity_fractions')} "
                f"cost={review.get('review_cost_bps')}"
            ),
        }
    )
    governance = json.loads(robustness_manifest_paths[2].read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "registered lifecycle is reproduced exactly before ablation",
            "passed": governance.get("registered_lifecycle_reproduced_exactly") is True,
            "detail": f"exact={governance.get('registered_lifecycle_reproduced_exactly')}",
        }
    )

    verify_manifest_hashes(controlled_results_manifest_path, checks)
    verify_manifest_hashes(shortcut_manifest_path, checks)
    controlled_results = json.loads(
        controlled_results_manifest_path.read_text(encoding="utf-8")
    )
    checks.append(
        {
            "check": "main controlled evaluation remains validation-only",
            "passed": controlled_results.get("evaluation_split") == "validation"
            and controlled_results.get("confirmatory") is False
            and controlled_results.get("rows_evaluated") == 40_000
            and controlled_results.get("clusters_evaluated") == 10_000
            and controlled_results.get("bootstrap_replicates") == 5_000,
            "detail": (
                f"split={controlled_results.get('evaluation_split')} "
                f"rows={controlled_results.get('rows_evaluated')} "
                f"clusters={controlled_results.get('clusters_evaluated')} "
                f"reps={controlled_results.get('bootstrap_replicates')}"
            ),
        }
    )
    controlled_decision_splits = _decision_splits(ROOT / "results" / "main" / "decisions.csv")
    checks.append(
        {
            "check": "main controlled decisions exclude train and test rows",
            "passed": controlled_decision_splits == {"validation"},
            "detail": f"splits={sorted(controlled_decision_splits)}",
        }
    )
    controlled_metrics = pd.read_csv(ROOT / "results" / "main" / "metrics.csv")
    controlled_zero = controlled_metrics[controlled_metrics["coverage"] == 0]
    checks.append(
        {
            "check": "main zero-coverage controlled FAR remains N/A",
            "passed": controlled_zero["far"].isna().all(),
            "detail": f"zero_coverage_rows={len(controlled_zero)}",
        }
    )

    provenance = json.loads(provenance_manifest_path.read_text(encoding="utf-8"))
    provenance_data = ROOT / provenance["data_path"]
    verify_hash(provenance_data, provenance["data_sha256"], checks)
    verify_hash(ROOT / "configs" / "provenance_main.yaml", provenance["config_sha256"], checks)
    checks.append(
        {
            "check": "main provenance scale, source, and split freeze",
            "passed": provenance.get("rows") == 200_000
            and provenance.get("clusters") == 50_000
            and provenance.get("split_counts")
            == {"test": 40_000, "train": 120_000, "validation": 40_000}
            and provenance.get("source_data_sha256") == controlled.get("data_sha256"),
            "detail": (
                f"rows={provenance.get('rows')} clusters={provenance.get('clusters')} "
                f"splits={provenance.get('split_counts')} "
                f"source_match={provenance.get('source_data_sha256') == controlled.get('data_sha256')}"
            ),
        }
    )

    verify_manifest_hashes(provenance_results_manifest_path, checks)
    provenance_results = json.loads(
        provenance_results_manifest_path.read_text(encoding="utf-8")
    )
    checks.append(
        {
            "check": "main provenance evaluation remains validation-only",
            "passed": provenance_results.get("evaluation_split") == "validation"
            and provenance_results.get("confirmatory") is False
            and provenance_results.get("rows_evaluated") == 40_000
            and provenance_results.get("clusters_evaluated") == 10_000
            and provenance_results.get("bootstrap_replicates") == 5_000,
            "detail": (
                f"split={provenance_results.get('evaluation_split')} "
                f"rows={provenance_results.get('rows_evaluated')} "
                f"clusters={provenance_results.get('clusters_evaluated')} "
                f"reps={provenance_results.get('bootstrap_replicates')}"
            ),
        }
    )
    provenance_decision_splits = _decision_splits(
        ROOT / "results" / "provenance_main" / "decisions.csv"
    )
    checks.append(
        {
            "check": "main provenance decisions exclude train and test rows",
            "passed": provenance_decision_splits == {"validation"},
            "detail": f"splits={sorted(provenance_decision_splits)}",
        }
    )
    provenance_summary = pd.read_csv(ROOT / "results" / "provenance_main" / "summary.csv")
    provenance_zero = provenance_summary[provenance_summary["coverage"] == 0]
    checks.append(
        {
            "check": "main zero-coverage provenance FAR and ALR remain N/A",
            "passed": provenance_zero["far"].isna().all()
            and provenance_zero["alr"].isna().all(),
            "detail": f"zero_coverage_rules={len(provenance_zero)}",
        }
    )

    summary_manifest = json.loads(summary_manifest_path.read_text(encoding="utf-8"))
    verify_manifest_hashes(summary_manifest_path, checks)
    for relative, expected in summary_manifest.get("inputs", {}).items():
        verify_hash(ROOT / relative, expected, checks)
    checks.append(
        {
            "check": "main validation summary remains descriptive and test-sealed",
            "passed": summary_manifest.get("confirmatory") is False
            and summary_manifest.get("test_outcomes_evaluated") is False,
            "detail": (
                f"confirmatory={summary_manifest.get('confirmatory')} "
                f"test={summary_manifest.get('test_outcomes_evaluated')}"
            ),
        }
    )
    verify_paper_test_protocol(checks)


def verify_external_orderbook(checks: list[dict[str, object]]) -> None:
    config_path = ROOT / "configs" / "external_orderbook_v03.yaml"
    preregistration_path = ROOT / "manifests" / "preregistration" / "external_orderbook_v03.yaml"
    feature_manifest_path = ROOT / "manifests" / "external_orderbook_v03_feature_access.json"
    structural_path = ROOT / "results" / "external_orderbook_v03" / "structural_audit.json"
    power_path = ROOT / "results" / "external_orderbook_v03" / "preresult_power.json"
    freeze_path = ROOT / "manifests" / "external_orderbook_v03_freeze.json"
    registry_path = ROOT / "manifests" / "external_orderbook_v03_test_registry.json"
    required = (
        config_path,
        preregistration_path,
        feature_manifest_path,
        structural_path,
        power_path,
        freeze_path,
        registry_path,
    )
    for path in required:
        checks.append(
            {
                "check": str(path),
                "passed": path.exists(),
                "detail": "exists" if path.exists() else "missing",
            }
        )
    if not all(path.exists() for path in required):
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "external order-book version and evidence tiers",
            "passed": config.get("version") == "0.3.0"
            and config.get("confirmatory_source") == "binance_public_depth"
            and config.get("descriptive_source") == "databento_mes_bbo",
            "detail": (
                f"version={config.get('version')} confirmatory={config.get('confirmatory_source')} "
                f"descriptive={config.get('descriptive_source')}"
            ),
        }
    )
    for source_name in ("binance", "databento"):
        source = config[source_name]
        source_manifest_path = ROOT / source["source_manifest"]
        dataset_manifest_path = ROOT / source["dataset_manifest"]
        for path in (source_manifest_path, dataset_manifest_path):
            checks.append(
                {
                    "check": str(path),
                    "passed": path.exists(),
                    "detail": "exists" if path.exists() else "missing",
                }
            )
        if not source_manifest_path.exists() or not dataset_manifest_path.exists():
            continue
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        for relative, expected in dataset_manifest.get("outputs", {}).items():
            verify_hash(ROOT / relative, expected, checks)
        checks.append(
            {
                "check": f"{source_name} construction remained pre-result",
                "passed": source_manifest.get("outcome_metrics_computed") is False
                and dataset_manifest.get("outcome_metrics_computed") is False,
                "detail": (
                    f"source={source_manifest.get('outcome_metrics_computed')} "
                    f"dataset={dataset_manifest.get('outcome_metrics_computed')}"
                ),
            }
        )
        expected_test = int(source["paper_test_clusters"])
        checks.append(
            {
                "check": f"{source_name} frozen paper-test cluster count",
                "passed": dataset_manifest.get("splits", {}).get("paper_test") == expected_test,
                "detail": f"splits={dataset_manifest.get('splits')} expected={expected_test}",
            }
        )
        if source_name == "databento":
            checks.append(
                {
                    "check": "Databento release boundary",
                    "passed": source_manifest.get("raw_or_row_level_redistribution") is False
                    and dataset_manifest.get("raw_or_row_level_redistribution") is False
                    and dataset_manifest.get("entitlement_required") is True,
                    "detail": (
                        f"source_release={source_manifest.get('raw_or_row_level_redistribution')} "
                        f"dataset_release={dataset_manifest.get('raw_or_row_level_redistribution')} "
                        f"entitlement={dataset_manifest.get('entitlement_required')}"
                    ),
                }
            )
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    power = json.loads(power_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "external structural audit passed without outcomes",
            "passed": structural.get("passed") is True
            and structural.get("outcome_columns_read") is False
            and structural.get("outcome_metrics_computed") is False,
            "detail": (
                f"passed={structural.get('passed')} columns={structural.get('outcome_columns_read')} "
                f"metrics={structural.get('outcome_metrics_computed')}"
            ),
        }
    )
    checks.append(
        {
            "check": "external power gate uses v0.2 only",
            "passed": power.get("gate_passed") is True
            and power.get("external_outcomes_read") is False
            and power.get("sources", {}).get("binance", {}).get("classification") == "confirmatory"
            and power.get("sources", {}).get("databento", {}).get("classification") == "descriptive",
            "detail": json.dumps(power.get("sources", {}), sort_keys=True),
        }
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "external content-addressed freeze status",
            "passed": freeze.get("status") == "FROZEN_BEFORE_EXTERNAL_TEST"
            and freeze.get("paper_test_outcomes_evaluated") is False
            and freeze.get("community_hidden_outcomes_evaluated") is False,
            "detail": (
                f"status={freeze.get('status')} test={freeze.get('paper_test_outcomes_evaluated')} "
                f"hidden={freeze.get('community_hidden_outcomes_evaluated')}"
            ),
        }
    )
    archive_path = ROOT / freeze["archive_path"]
    verify_hash(archive_path, freeze["archive_sha256"], checks)
    successor_path = ROOT / "manifests" / "real_agent_v05_freeze.json"
    successor_hashes = {}
    if successor_path.exists():
        successor_hashes = json.loads(successor_path.read_text(encoding="utf-8")).get(
            "surface_hashes", {}
        )
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archived_members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for relative, expected in freeze.get("surface_hashes", {}).items():
            current_path = ROOT / relative
            current = sha256(current_path) if current_path.exists() else None
            if current == expected:
                checks.append(
                    {
                        "check": str(current_path),
                        "passed": True,
                        "detail": f"expected={expected} actual={current}",
                    }
                )
                continue
            member = archived_members.get(relative)
            handle = archive.extractfile(member) if member is not None else None
            archived = hashlib.sha256(handle.read()).hexdigest() if handle is not None else None
            successor = successor_hashes.get(relative)
            checks.append(
                {
                    "check": f"versioned frozen surface {current_path}",
                    "passed": archived == expected and current == successor,
                    "detail": (
                        f"v03_archive={archived} expected={expected} "
                        f"current={current} v05_successor={successor}"
                    ),
                }
            )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "external test completed once and hidden remained sealed",
            "passed": registry.get("status") == "COMPLETED"
            and registry.get("paper_test_outcomes_evaluated") is True
            and registry.get("community_hidden_outcomes_evaluated") is False,
            "detail": (
                f"status={registry.get('status')} test={registry.get('paper_test_outcomes_evaluated')} "
                f"hidden={registry.get('community_hidden_outcomes_evaluated')}"
            ),
        }
    )
    results_root = ROOT / "results" / "external_orderbook_v03" / "paper_test"
    for source_name in ("binance", "databento"):
        for name, expected in registry.get("outputs", {}).get(source_name, {}).items():
            verify_hash(results_root / source_name / name, expected, checks)
    cross_expected = registry.get("outputs", {}).get("cross_source_rank_correlation.json")
    if cross_expected:
        verify_hash(results_root / "cross_source_rank_correlation.json", cross_expected, checks)
    databento_outputs = set(registry.get("outputs", {}).get("databento", {}))
    checks.append(
        {
            "check": "Databento test outputs are aggregate-only",
            "passed": "decisions.csv" not in databento_outputs
            and "paired_cluster_burdens.csv" not in databento_outputs,
            "detail": sorted(databento_outputs),
        }
    )


def verify_real_agent(checks: list[dict[str, object]]) -> None:
    config_path = ROOT / "configs" / "real_agent_v05.yaml"
    source_manifest_path = ROOT / "manifests" / "real_agent_v05" / "source_manifest.json"
    dataset_manifest_path = ROOT / "manifests" / "real_agent_v05" / "dataset_manifest.json"
    proposal_manifest_path = ROOT / "manifests" / "real_agent_v05" / "proposal_manifest.json"
    feature_access_path = ROOT / "manifests" / "real_agent_v05_feature_access.json"
    structural_path = ROOT / "results" / "real_agent_v05" / "structural_audit.json"
    freeze_path = ROOT / "manifests" / "real_agent_v05_freeze.json"
    registry_path = ROOT / "manifests" / "real_agent_v05_test_registry.json"
    required = (
        config_path,
        source_manifest_path,
        dataset_manifest_path,
        proposal_manifest_path,
        feature_access_path,
        structural_path,
        freeze_path,
        registry_path,
    )
    for path in required:
        checks.append(
            {
                "check": str(path),
                "passed": path.exists(),
                "detail": "exists" if path.exists() else "missing",
            }
        )
    if not all(path.exists() for path in required):
        return

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "real-agent registered version and source boundary",
            "passed": str(config.get("version")) == "0.5.0"
            and str(config["binance"]["start_date"])
            > str(config["binance"]["prior_inspected_period_end"])
            and config["generation"].get("prohibit_hidden_generation") is True
            and config["generation"].get("rationale_is_execution_evidence") is False,
            "detail": (
                f"version={config.get('version')} start={config['binance']['start_date']} "
                f"prior_end={config['binance']['prior_inspected_period_end']} "
                f"hidden={config['generation'].get('prohibit_hidden_generation')} "
                f"rationale_evidence={config['generation'].get('rationale_is_execution_evidence')}"
            ),
        }
    )

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_failures: list[str] = []
    for record in source_manifest.get("records", []):
        relative = record.get("path")
        expected = record.get("sha256")
        if not relative or not expected:
            source_failures.append(str(record.get("url", "missing-path")))
            continue
        path = ROOT / str(relative)
        if not path.exists() or sha256(path) != str(expected):
            source_failures.append(str(relative))
    checks.append(
        {
            "check": "real-agent public source archives and checksums",
            "passed": source_manifest.get("requested_files") == 248
            and len(source_manifest.get("records", [])) == 248
            and source_manifest.get("outcome_metrics_computed") is False
            and not source_failures,
            "detail": (
                f"requested={source_manifest.get('requested_files')} "
                f"records={len(source_manifest.get('records', []))} "
                f"failures={source_failures[:5]}"
            ),
        }
    )

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    for relative, expected in dataset_manifest.get("outputs", {}).items():
        verify_hash(ROOT / relative, expected, checks)
    checks.append(
        {
            "check": "real-agent cluster and split contract",
            "passed": dataset_manifest.get("contexts") == 100
            and dataset_manifest.get("event_clusters") == 100
            and dataset_manifest.get("splits")
            == {"community_hidden": 20, "development": 20, "paper_test": 60}
            and dataset_manifest.get("symbols") == {"BTCUSDT": 50, "ETHUSDT": 50}
            and dataset_manifest.get("calendar_overlap_with_v03") is False
            and dataset_manifest.get("outcome_metrics_computed") is False
            and dataset_manifest.get("community_hidden_outcomes_evaluated") is False,
            "detail": (
                f"contexts={dataset_manifest.get('contexts')} "
                f"splits={dataset_manifest.get('splits')} symbols={dataset_manifest.get('symbols')}"
            ),
        }
    )

    feature_access = json.loads(feature_access_path.read_text(encoding="utf-8"))
    context_frames = []
    for split in ("development", "paper_test", "community_hidden"):
        path = ROOT / "data" / "real_agent_v05" / "derived" / f"{split}_contexts.csv"
        frame = pd.read_csv(path)
        context_frames.append(frame)
    contexts = pd.concat(context_frames, ignore_index=True)
    forbidden = set(feature_access["global_forbidden"])
    legal_context = set(feature_access["legal_prompt_fields"])
    checks.append(
        {
            "check": "real-agent prompt legality and split isolation",
            "passed": len(contexts) == 100
            and contexts["event_cluster_id"].nunique() == 100
            and not (forbidden & set(contexts.columns))
            and legal_context.issubset(contexts.columns)
            and set(contexts["split"]) == {"development", "paper_test", "community_hidden"},
            "detail": (
                f"rows={len(contexts)} clusters={contexts['event_cluster_id'].nunique()} "
                f"forbidden={sorted(forbidden & set(contexts.columns))}"
            ),
        }
    )

    proposal_manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    proposal_path = ROOT / str(proposal_manifest["proposal_file"])
    verify_hash(proposal_path, proposal_manifest["proposal_sha256"], checks)
    proposals = pd.read_csv(proposal_path)
    registered_models = {str(item["model_id"]) for item in config["models"]}
    expected_contexts = contexts[contexts["split"].isin(["development", "paper_test"])][
        "context_id"
    ].nunique()
    per_model_contexts = proposals.groupby("model_id")["context_id"].nunique().to_dict()
    checks.append(
        {
            "check": "real-agent proposal cache contract",
            "passed": len(proposals) == 240
            and set(proposals["model_id"]) == registered_models
            and set(proposals["split"]) == {"development", "paper_test"}
            and "community_hidden" not in set(proposals["split"])
            and all(int(value) == expected_contexts for value in per_model_contexts.values())
            and set(proposals["source_role"]) == {"edge_proposer"}
            and set(proposals["original_source_eligible"].astype(str).str.lower()) == {"true"}
            and set(proposals["rationale_not_execution_evidence"].astype(str).str.lower())
            == {"true"}
            and not (forbidden & set(proposals.columns))
            and proposal_manifest.get("community_hidden_contexts_generated") == 0
            and proposal_manifest.get("outcome_fields_read") is False
            and proposal_manifest.get("raw_schema_validity") == 1.0
            and proposal_manifest.get("repairs_applied") == 0,
            "detail": (
                f"rows={len(proposals)} models={sorted(set(proposals['model_id']))} "
                f"splits={sorted(set(proposals['split']))} per_model={per_model_contexts}"
            ),
        }
    )

    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    failed_structural = [
        item.get("check") for item in structural.get("checks", []) if not item.get("passed")
    ]
    checks.append(
        {
            "check": "real-agent pre-result structural audit",
            "passed": structural.get("passed") is True
            and structural.get("outcome_files_read") is False
            and structural.get("outcome_metrics_computed") is False
            and len(structural.get("checks", [])) == 22
            and not failed_structural,
            "detail": (
                f"checks={len(structural.get('checks', []))} failures={failed_structural} "
                f"outcomes_read={structural.get('outcome_files_read')}"
            ),
        }
    )

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "real-agent content-addressed freeze status",
            "passed": freeze.get("status") == "FROZEN_BEFORE_REAL_AGENT_TEST"
            and freeze.get("paper_test_outcomes_evaluated") is False
            and freeze.get("community_hidden_outcomes_evaluated") is False,
            "detail": (
                f"status={freeze.get('status')} test={freeze.get('paper_test_outcomes_evaluated')} "
                f"hidden={freeze.get('community_hidden_outcomes_evaluated')}"
            ),
        }
    )
    for relative, expected in freeze.get("surface_hashes", {}).items():
        verify_hash(ROOT / relative, expected, checks)
    archive_path = ROOT / str(freeze["archive_path"])
    verify_hash(archive_path, freeze["archive_sha256"], checks)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "real-agent one-time test completed and hidden outcomes remained sealed",
            "passed": registry.get("status") == "COMPLETED"
            and registry.get("paper_test_clusters") == 60
            and registry.get("proposal_rows") == 180
            and registry.get("paper_test_outcomes_evaluated") is True
            and registry.get("community_hidden_outcomes_evaluated") is False
            and sha256(freeze_path) == registry.get("freeze_manifest_sha256"),
            "detail": (
                f"status={registry.get('status')} clusters={registry.get('paper_test_clusters')} "
                f"rows={registry.get('proposal_rows')} hidden={registry.get('community_hidden_outcomes_evaluated')}"
            ),
        }
    )
    results_root = ROOT / "results" / "real_agent_v05" / "paper_test"
    for name, expected in registry.get("outputs", {}).items():
        verify_hash(results_root / name, expected, checks)

    metrics = pd.read_csv(results_root / "rule_metrics.csv")
    route_sum = metrics[["execute_rate", "reduce_rate", "review_rate", "abstain_rate"]].sum(axis=1)
    zero_action = metrics["authorized_count"] == 0
    execution_na = metrics["far"].isna() & metrics["upa"].isna()
    checks.append(
        {
            "check": "real-agent routing and structural N/A semantics",
            "passed": ((route_sum - 1.0).abs() < 1e-9).all()
            and (execution_na == zero_action).all()
            and metrics.loc[~zero_action, "alr"].eq(0.0).all(),
            "detail": (
                f"max_route_residual={(route_sum - 1.0).abs().max()} "
                f"zero_action_rows={int(zero_action.sum())}"
            ),
        }
    )
    source_quality = pd.read_csv(results_root / "model_source_quality.csv")
    checks.append(
        {
            "check": "real-agent source-quality reporting",
            "passed": set(source_quality["model_id"]) == registered_models
            and source_quality["contexts"].eq(60).all()
            and source_quality["raw_schema_validity"].eq(1.0).all()
            and source_quality["repairs_applied"].eq(0).all(),
            "detail": source_quality[
                ["model_id", "contexts", "raw_schema_validity", "repairs_applied"]
            ].to_dict(orient="records"),
        }
    )
    ranking = json.loads((results_root / "ranking_transfer.json").read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "real-agent transfer result is descriptive and non-leaderboard",
            "passed": ranking.get("classification") == "descriptive prospective transfer"
            and len(ranking.get("rules_compared", [])) == 6
            and float(ranking.get("far_spearman_rho")) < 0.0
            and "not a model ranking" in str(ranking.get("claim_boundary")),
            "detail": (
                f"far_rho={ranking.get('far_spearman_rho')} "
                f"rules={ranking.get('rules_compared')}"
            ),
        }
    )


def verify_real_agent_v06(
    checks: list[dict[str, object]], root: Path = ROOT, repo: Path = REPO
) -> None:
    config_path = root / "configs" / "real_agent_v06.yaml"
    freeze_path = root / "manifests" / "real_agent_v06_freeze.json"
    registry_path = root / "manifests" / "real_agent_v06_test_registry.json"
    hidden_snapshot_path = (
        root / "manifests" / "real_agent_v06" / "hidden_proposal_snapshot.json"
    )
    required = (config_path, freeze_path, registry_path, hidden_snapshot_path)
    for path in required:
        append_check(
            checks,
            str(path),
            path.is_file(),
            "exists" if path.is_file() else "missing",
        )
    if not all(path.is_file() for path in required):
        return

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    append_check(
        checks,
        "real-agent-v06 freeze status",
        freeze.get("version") == "0.6.0"
        and freeze.get("status") == "FROZEN_BEFORE_V06_PAPER_TEST"
        and freeze.get("paper_test_outcomes_evaluated") is False
        and freeze.get("community_hidden_outcomes_evaluated") is False
        and freeze.get("hidden_proposals_decrypted") is False,
        (
            f"version={freeze.get('version')} status={freeze.get('status')} "
            f"paper_test={freeze.get('paper_test_outcomes_evaluated')} "
            f"community_hidden={freeze.get('community_hidden_outcomes_evaluated')} "
            f"hidden_decrypted={freeze.get('hidden_proposals_decrypted')}"
        ),
    )
    append_check(
        checks,
        "real-agent-v06 freeze manifest is read-only",
        is_read_only(freeze_path),
        f"mode={oct(freeze_path.stat().st_mode & 0o777)}",
    )

    surface_hashes = freeze.get("surface_hashes")
    append_check(
        checks,
        "real-agent-v06 frozen surface registry",
        isinstance(surface_hashes, dict) and bool(surface_hashes),
        f"registered={len(surface_hashes) if isinstance(surface_hashes, dict) else 0}",
    )
    if isinstance(surface_hashes, dict):
        for relative, expected in sorted(surface_hashes.items()):
            path = safe_child(root, relative)
            if path is None:
                append_check(
                    checks,
                    f"frozen surface {relative}",
                    False,
                    "path escapes artifact root",
                )
            elif Path(str(relative)).name in {"decisions.csv", "summary.md"}:
                append_check(
                    checks,
                    f"frozen surface {relative}",
                    False,
                    "forbidden row-level or free-text output registration",
                )
            elif not is_sha256(expected):
                append_check(
                    checks,
                    f"frozen surface {relative}",
                    False,
                    f"invalid sha256={expected}",
                )
            else:
                verify_hash(path, str(expected), checks)

    archive_path = safe_child(root, freeze.get("archive_path"))
    if archive_path is None or not is_sha256(freeze.get("archive_sha256")):
        append_check(
            checks,
            "real-agent-v06 freeze archive contract",
            False,
            (
                f"path={freeze.get('archive_path')} "
                f"sha256={freeze.get('archive_sha256')}"
            ),
        )
    else:
        verify_hash(archive_path, str(freeze["archive_sha256"]), checks)
        append_check(
            checks,
            "real-agent-v06 freeze archive is read-only",
            is_read_only(archive_path),
            (
                f"mode={oct(archive_path.stat().st_mode & 0o777)}"
                if archive_path.exists()
                else "missing"
            ),
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    append_check(
        checks,
        "real-agent-v06 registry completed and read-only",
        registry.get("version") == "0.6.0"
        and registry.get("status") == "COMPLETED"
        and registry.get("paper_test_outcomes_evaluated") is True
        and is_read_only(registry_path),
        (
            f"version={registry.get('version')} status={registry.get('status')} "
            f"paper_test={registry.get('paper_test_outcomes_evaluated')} "
            f"mode={oct(registry_path.stat().st_mode & 0o777)}"
        ),
    )
    append_check(
        checks,
        "real-agent-v06 registry cardinality and exact model/task sets",
        registry.get("paper_test_clusters") == 200
        and registry.get("proposal_rows") == 1200
        and set(registry.get("models", [])) == REAL_AGENT_V06_MODELS
        and set(registry.get("tasks", [])) == REAL_AGENT_V06_TASKS,
        (
            f"clusters={registry.get('paper_test_clusters')} "
            f"rows={registry.get('proposal_rows')} "
            f"models={registry.get('models')} tasks={registry.get('tasks')}"
        ),
    )
    append_check(
        checks,
        "real-agent-v06 registry-to-freeze digest",
        registry.get("freeze_manifest") == "manifests/real_agent_v06_freeze.json"
        and is_sha256(registry.get("freeze_manifest_sha256"))
        and sha256(freeze_path) == registry.get("freeze_manifest_sha256"),
        (
            f"manifest={registry.get('freeze_manifest')} "
            f"expected={registry.get('freeze_manifest_sha256')}"
        ),
    )
    append_check(
        checks,
        "real-agent-v06 hidden outcomes and proposals remained sealed",
        registry.get("community_hidden_outcomes_evaluated") is False
        and registry.get("hidden_proposals_decrypted") is False,
        (
            f"community_hidden={registry.get('community_hidden_outcomes_evaluated')} "
            f"hidden_decrypted={registry.get('hidden_proposals_decrypted')}"
        ),
    )

    outputs = registry.get("outputs")
    expected_output_names = (
        REAL_AGENT_V06_AGGREGATE_OUTPUTS | REAL_AGENT_V06_UNREAD_OUTPUTS
    )
    append_check(
        checks,
        "real-agent-v06 registered output contract",
        isinstance(outputs, dict) and set(outputs) == expected_output_names,
        (
            f"registered={sorted(outputs) if isinstance(outputs, dict) else None}"
        ),
    )
    results_root = root / "results" / "real_agent_v06" / "paper_test"
    if isinstance(outputs, dict):
        for relative in sorted(REAL_AGENT_V06_AGGREGATE_OUTPUTS):
            expected = outputs.get(relative)
            path = safe_child(results_root, relative)
            if path is None or not is_sha256(expected):
                append_check(
                    checks,
                    f"registered aggregate {relative}",
                    False,
                    f"path={path} sha256={expected}",
                )
            else:
                verify_hash(path, str(expected), checks)
        unread_valid = all(
            is_sha256(outputs.get(relative))
            and (results_root / relative).is_file()
            for relative in REAL_AGENT_V06_UNREAD_OUTPUTS
        )
        append_check(
            checks,
            "real-agent-v06 row-level decisions and free-text summary left unread",
            unread_valid,
            sorted(REAL_AGENT_V06_UNREAD_OUTPUTS),
        )

    rank_path = results_root / "rank_transfer_zero_shot.json"
    if rank_path.is_file():
        rank = json.loads(rank_path.read_text(encoding="utf-8"))
        derived_support = derive_rank_primary_support(rank)
        exact = rank.get("exact_permutation", {})
        bootstrap = rank.get("date_cluster_bootstrap", {})
        append_check(
            checks,
            "real-agent-v06 rank aggregate matches registry",
            rank == registry.get("rank_transfer"),
            f"registered={rank == registry.get('rank_transfer')}",
        )
        append_check(
            checks,
            "real-agent-v06 primary_support semantics and current result",
            rank.get("primary_support") is derived_support
            and derived_support is False
            and tuple(rank.get("rules", [])) == REAL_AGENT_V06_RANK_RULES
            and rank.get("metric") == "economic_loss_authorization_rate"
            and rank.get("community_hidden_outcomes_evaluated") is False,
            (
                f"reported={rank.get('primary_support')} derived={derived_support} "
                f"exact_p={exact.get('exact_p_value') if isinstance(exact, dict) else None} "
                f"probability_below_zero="
                f"{bootstrap.get('probability_below_zero') if isinstance(bootstrap, dict) else None}"
            ),
        )

    reason = "NO_ELIGIBLE_CANDIDATE_COVERAGE_FLOOR"
    calibration = registry.get("calibration_validity_by_rule", {})
    lifecycle_calibration = (
        calibration.get("Lifecycle Checklist", {})
        if isinstance(calibration, dict)
        else {}
    )
    append_check(
        checks,
        "real-agent-v06 Lifecycle registry reason",
        lifecycle_calibration.get("calibration_valid") is False
        and lifecycle_calibration.get("status") == reason
        and lifecycle_calibration.get("reason") == reason
        and lifecycle_calibration.get("selected_candidate") is None
        and lifecycle_calibration.get("candidates_meeting_floor") == 0,
        lifecycle_calibration,
    )

    metrics_path = results_root / "recalibrated" / "metrics.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        lifecycle = metrics[metrics["rule"] == "Lifecycle Checklist"].copy()
        profiles = {"overall", *REAL_AGENT_V06_TASKS}
        required_columns = set(REAL_AGENT_V06_STRUCTURAL_NA_COLUMNS)
        markers_valid = (
            lifecycle["calibration_valid"]
            .astype(str)
            .str.lower()
            .eq("false")
            .all()
            and lifecycle["calibration_status"].eq(reason).all()
            and lifecycle["structural_n_a_reason"].eq(reason).all()
            and lifecycle["track"].eq("recalibrated").all()
        )
        rows_by_profile = {
            str(row.profile): int(row.rows)
            for row in lifecycle[["profile", "rows"]].itertuples(index=False)
        }
        append_check(
            checks,
            "real-agent-v06 Lifecycle recalibrated metrics are structural N/A",
            len(lifecycle) == 3
            and set(lifecycle["profile"]) == profiles
            and rows_by_profile
            == {
                "overall": 1200,
                "directional_execution": 600,
                "risk_limit_increase": 600,
            }
            and required_columns <= set(lifecycle.columns)
            and lifecycle[list(REAL_AGENT_V06_STRUCTURAL_NA_COLUMNS)]
            .isna()
            .all()
            .all()
            and markers_valid,
            f"rows={rows_by_profile} profiles={sorted(set(lifecycle['profile']))}",
        )

    bootstrap_path = results_root / "recalibrated" / "bootstrap_bounds.csv"
    if bootstrap_path.is_file():
        bounds = pd.read_csv(bootstrap_path)
        lifecycle_bounds = bounds[bounds["rule"] == "Lifecycle Checklist"].copy()
        expected_pairs = {
            (profile, metric)
            for profile in {"overall", *REAL_AGENT_V06_TASKS}
            for metric in REAL_AGENT_V06_PRIMARY_METRICS
        }
        actual_pairs = set(
            lifecycle_bounds[["profile", "metric"]].itertuples(
                index=False, name=None
            )
        )
        append_check(
            checks,
            "real-agent-v06 Lifecycle recalibrated bootstrap bounds are structural N/A",
            actual_pairs == expected_pairs
            and lifecycle_bounds[
                ["point", "lcb95", "ucb95", "valid_replicate_fraction"]
            ]
            .isna()
            .all()
            .all()
            and lifecycle_bounds["replicates"].eq(0).all()
            and lifecycle_bounds["clusters"].eq(200).all()
            and lifecycle_bounds["calibration_valid"]
            .astype(str)
            .str.lower()
            .eq("false")
            .all()
            and lifecycle_bounds["calibration_status"].eq(reason).all()
            and lifecycle_bounds["structural_n_a_reason"].eq(reason).all(),
            f"rows={len(lifecycle_bounds)} pairs={len(actual_pairs)}",
        )

    subgroup_path = results_root / "recalibrated" / "subgroup_metrics.csv"
    if subgroup_path.is_file():
        subgroups = pd.read_csv(subgroup_path)
        lifecycle_subgroups = subgroups[
            subgroups["rule"] == "Lifecycle Checklist"
        ].copy()
        expected_dimension_values = {
            "model_id": REAL_AGENT_V06_MODELS,
            "symbol": frozenset(
                {"BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
            ),
            "volatility_regime": frozenset({"high", "low"}),
            "task_id": REAL_AGENT_V06_TASKS,
        }
        actual_dimension_values = {
            dimension: frozenset(
                lifecycle_subgroups.loc[
                    lifecycle_subgroups["dimension"] == dimension, "value"
                ].astype(str)
            )
            for dimension in expected_dimension_values
        }
        dimension_detail = {
            dimension: sorted(values)
            for dimension, values in actual_dimension_values.items()
        }
        append_check(
            checks,
            "real-agent-v06 Lifecycle recalibrated subgroups are structural N/A",
            len(lifecycle_subgroups) == 12
            and actual_dimension_values == expected_dimension_values
            and lifecycle_subgroups[list(REAL_AGENT_V06_STRUCTURAL_NA_COLUMNS)]
            .isna()
            .all()
            .all()
            and lifecycle_subgroups["rows"].notna().all()
            and lifecycle_subgroups["calibration_valid"]
            .astype(str)
            .str.lower()
            .eq("false")
            .all()
            and lifecycle_subgroups["calibration_status"].eq(reason).all()
            and lifecycle_subgroups["structural_n_a_reason"].eq(reason).all(),
            dimension_detail,
        )

    hidden_snapshot = json.loads(hidden_snapshot_path.read_text(encoding="utf-8"))
    ciphertext_path = safe_child(root, hidden_snapshot.get("ciphertext_path"))
    ciphertext_valid = (
        ciphertext_path is not None
        and is_sha256(hidden_snapshot.get("ciphertext_sha256"))
        and ciphertext_path.is_file()
        and sha256(ciphertext_path) == hidden_snapshot.get("ciphertext_sha256")
    )
    configured_key = Path(str(config.get("generation", {}).get("hidden_key_path", "")))
    configured_ciphertext = str(
        config.get("generation", {}).get("hidden_ciphertext", "")
    )
    append_check(
        checks,
        "real-agent-v06 hidden snapshot is ciphertext-only with external key custody",
        hidden_snapshot.get("status") == "ENCRYPTED_BEFORE_OUTCOME_EVALUATION"
        and hidden_snapshot.get("repository_plaintext_persisted") is False
        and hidden_snapshot.get("outcome_fields_read") is False
        and hidden_snapshot.get("hidden_outcomes_evaluated") is False
        and hidden_snapshot.get("key_location") == "outside repository and release"
        and configured_key.is_absolute()
        and not path_is_within(configured_key, repo)
        and configured_ciphertext == hidden_snapshot.get("ciphertext_path")
        and ciphertext_valid,
        (
            f"ciphertext={hidden_snapshot.get('ciphertext_path')} "
            f"key_path={configured_key} key_in_repo={path_is_within(configured_key, repo)}"
        ),
    )
    suspicious_paths = find_real_agent_v06_secret_paths(repo)
    append_check(
        checks,
        "real-agent-v06 repository contains no hidden proposal plaintext or key",
        not suspicious_paths,
        suspicious_paths or "none",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the isolated FinAuth-Audit artifact.")
    parser.add_argument(
        "--phase",
        choices=[
            "smoke",
            "provenance",
            "public",
            "training",
            "main",
            "paper-test",
            "external-orderbook",
            "real-agent",
            "real-agent-v06",
        ],
        default="training",
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    upstream = json.loads((ROOT / "manifests" / "upstream_freeze.json").read_text(encoding="utf-8"))
    for artifact in upstream["artifacts"]:
        path = resolve_upstream(artifact["path"])
        if artifact.get("verification") == "record_only":
            record_observed_hash(path, artifact["sha256"], checks)
        else:
            verify_hash(path, artifact["sha256"], checks)

    hidden_archives = sorted((REPO / "finauth_worlds").rglob("*2025*.zip"))
    checks.append(
        {
            "check": "sealed FinAuth-Worlds period remains unopened",
            "passed": not hidden_archives,
            "detail": ", ".join(str(path) for path in hidden_archives) or "no matching archives",
        }
    )

    overlap_path = ROOT / "results" / "overlap_audit" / "aaai_overlap_audit.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8")) if overlap_path.exists() else {"blocking": True}
    checks.append(
        {
            "check": "AAAI overlap audit",
            "passed": overlap.get("blocking") is False,
            "detail": f"blocking={overlap.get('blocking')}",
        }
    )
    for source_path, audited_hash in overlap.get("source_hashes", {}).items():
        verify_hash(Path(source_path), audited_hash, checks)

    if args.phase == "smoke":
        verify_smoke(checks)
    elif args.phase == "provenance":
        verify_smoke(checks)
        verify_provenance(checks)
    elif args.phase == "public":
        verify_smoke(checks)
        verify_provenance(checks)
        verify_public(checks)
    elif args.phase == "training":
        verify_smoke(checks)
        verify_provenance(checks)
        verify_public(checks)
        verify_training(checks)
    elif args.phase == "main":
        verify_smoke(checks)
        verify_provenance(checks)
        verify_public(checks)
        verify_training(checks)
        verify_main(checks)
    elif args.phase == "paper-test":
        verify_smoke(checks)
        verify_provenance(checks)
        verify_public(checks)
        verify_training(checks)
        verify_main(checks)
        verify_paper_test_protocol(
            checks, require_frozen=True, require_complete=True
        )
    elif args.phase == "external-orderbook":
        verify_smoke(checks)
        verify_provenance(checks)
        verify_public(checks)
        verify_training(checks)
        verify_main(checks)
        verify_paper_test_protocol(
            checks, require_frozen=True, require_complete=True
        )
        verify_external_orderbook(checks)
    elif args.phase == "real-agent":
        verify_smoke(checks)
        verify_provenance(checks)
        verify_public(checks)
        verify_training(checks)
        verify_main(checks)
        verify_paper_test_protocol(
            checks, require_frozen=True, require_complete=True
        )
        verify_external_orderbook(checks)
        verify_real_agent(checks)
    elif args.phase == "real-agent-v06":
        verify_real_agent_v06(checks)
    if args.run_tests:
        run_tests(checks)

    passed = sum(bool(check["passed"]) for check in checks)
    report = {
        "phase": args.phase,
        "passed": passed,
        "total": len(checks),
        "success": passed == len(checks),
        "checks": checks,
    }
    internal_output_dir = ROOT / "results" / "verification"
    internal_output_dir.mkdir(parents=True, exist_ok=True)
    internal_output = internal_output_dir / f"{args.phase}_verification.json"
    internal_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    markdown_lines = [
        f"# FinAuth-Audit {args.phase.title()} Verification",
        "",
        f"- Passed: {passed}/{len(checks)}",
        f"- Success: {passed == len(checks)}",
        f"- Tests invoked: {args.run_tests}",
        "",
        "## Checks",
        "",
        "| # | Status | Check | Detail |",
        "|---:|:---:|---|---|",
    ]
    for index, check in enumerate(checks, start=1):
        status = "PASS" if check["passed"] else "FAIL"
        name = str(check["check"]).replace("|", "\\|").replace("\n", "<br>")
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", "<br>")
        markdown_lines.append(f"| {index} | {status} | {name} | {detail} |")
    internal_markdown_output = internal_output_dir / f"{args.phase}_verification.md"
    internal_markdown_output.write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )

    output = internal_output
    markdown_output = internal_markdown_output
    if args.phase == "real-agent-v06":
        freeze = json.loads(
            (ROOT / "manifests" / "real_agent_v06_freeze.json").read_text(
                encoding="utf-8"
            )
        )
        registry = json.loads(
            (ROOT / "manifests" / "real_agent_v06_test_registry.json").read_text(
                encoding="utf-8"
            )
        )
        checks_by_name = {
            str(check["check"]): bool(check["passed"]) for check in checks
        }
        public_checks = [
            {
                "check": "aggregate-only internal verification",
                "passed": report["success"],
                "detail": f"{passed}/{len(checks)} checks passed",
            },
            {
                "check": "frozen evaluation surface integrity",
                "passed": checks_by_name.get("real-agent-v06 frozen surface registry", False)
                and report["success"],
                "detail": f"registered_hashes={len(freeze.get('surface_hashes', {}))}",
            },
            {
                "check": "exactly-once paper-test registry",
                "passed": checks_by_name.get(
                    "real-agent-v06 registry completed and read-only", False
                ),
                "detail": (
                    f"status={registry.get('status')} "
                    f"clusters={registry.get('paper_test_clusters')} "
                    f"proposal_rows={registry.get('proposal_rows')}"
                ),
            },
            {
                "check": "community-hidden split remains sealed",
                "passed": checks_by_name.get(
                    "real-agent-v06 hidden outcomes and proposals remained sealed", False
                ),
                "detail": "outcomes_evaluated=false proposals_decrypted=false",
            },
            {
                "check": "registered paper-test aggregate hashes",
                "passed": report["success"],
                "detail": f"aggregate_files={len(REAL_AGENT_V06_AGGREGATE_OUTPUTS)}",
            },
            {
                "check": "preregistered rank-transfer inference semantics",
                "passed": checks_by_name.get(
                    "real-agent-v06 primary_support semantics and current result",
                    False,
                ),
                "detail": "primary_support=false",
            },
            {
                "check": "invalid recalibration remains structural N/A",
                "passed": checks_by_name.get(
                    "real-agent-v06 Lifecycle recalibrated metrics are structural N/A",
                    False,
                )
                and checks_by_name.get(
                    "real-agent-v06 Lifecycle recalibrated bootstrap bounds are structural N/A",
                    False,
                )
                and checks_by_name.get(
                    "real-agent-v06 Lifecycle recalibrated subgroups are structural N/A",
                    False,
                ),
                "detail": "Lifecycle Checklist did not meet the frozen development coverage floor",
            },
            {
                "check": "hidden proposal plaintext and key exclusion",
                "passed": checks_by_name.get(
                    "real-agent-v06 repository contains no hidden proposal plaintext or key",
                    False,
                ),
                "detail": "repository scan passed",
            },
        ]
        public_passed = sum(bool(check["passed"]) for check in public_checks)
        public_report = {
            "phase": args.phase,
            "schema_version": "0.6",
            "passed": public_passed,
            "total": len(public_checks),
            "success": public_passed == len(public_checks),
            "internal_passed": passed,
            "internal_total": len(checks),
            "checks": public_checks,
        }
        public_output_dir = ROOT / "results" / "verification" / "public"
        public_output_dir.mkdir(parents=True, exist_ok=True)
        output = public_output_dir / "real-agent-v06_verification.json"
        output.write_text(
            json.dumps(public_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        public_markdown_lines = [
            "# FinAuth-Audit Real-Agent v0.6 Public Verification",
            "",
            f"- Passed: {public_passed}/{len(public_checks)}",
            f"- Success: {public_report['success']}",
            f"- Internal aggregate-only checks: {passed}/{len(checks)}",
            "",
            "## Checks",
            "",
            "| # | Status | Check | Detail |",
            "|---:|:---:|---|---|",
        ]
        for index, check in enumerate(public_checks, start=1):
            status = "PASS" if check["passed"] else "FAIL"
            public_markdown_lines.append(
                f"| {index} | {status} | {check['check']} | {check['detail']} |"
            )
        markdown_output = public_output_dir / "real-agent-v06_verification.md"
        markdown_output.write_text(
            "\n".join(public_markdown_lines) + "\n", encoding="utf-8"
        )
    print(f"FinAuth-Audit verification: {passed}/{len(checks)}")
    for check in checks:
        if not check["passed"]:
            print(f"FAIL: {check['check']}: {check['detail']}", file=sys.stderr)
    print(output)
    print(markdown_output)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
