from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from finauth_audit.evaluation.external_orderbook_test import (
    _write_outputs,
    evaluate_frame,
    materialize_outcomes,
)
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


def _timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _verify_freeze(freeze_path: Path) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_DATABENTO_POWERED_TEST":
        raise RuntimeError("Databento v0.4 freeze manifest is not active")
    if freeze.get("paper_test_outcomes_evaluated") is not False:
        raise RuntimeError("v0.4 freeze records prior paper-test execution")
    if freeze.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("v0.4 freeze records hidden-set access")
    for relative, expected in freeze.get("surface_hashes", {}).items():
        path = resolve_root_path(relative)
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"frozen v0.4 surface changed: {relative}")
    for relative, expected in freeze.get("dataset_hashes", {}).items():
        path = resolve_root_path(relative)
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"frozen v0.4 dataset changed: {relative}")
    v03_freeze = json.loads(
        (ROOT / "manifests" / "external_orderbook_v03_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    for relative, expected in freeze.get("imported_v03_surface_hashes", {}).items():
        if v03_freeze.get("surface_hashes", {}).get(relative) != expected:
            raise RuntimeError(f"v0.4 imported v0.3 hash mismatch: {relative}")
    return freeze


def _result_wording(primary: dict[str, object]) -> dict[str, str]:
    false_pass = bool(primary["false_authorization_burden"]["passed"])
    laundering_pass = bool(primary["laundering_burden"]["passed"])
    if false_pass and laundering_pass:
        classification = "PASS"
        wording = (
            "The powered, temporally non-overlapping MES test reproduced both "
            "registered co-primary arms of the frozen v0.3 Databento protocol. "
            "It replicates the prior descriptive Databento arm, not the failed "
            "v0.3 Binance intersection-union result, and is not independent discovery."
        )
    elif false_pass or laundering_pass:
        classification = "MIXED_FAIL"
        passed = "false-authorization" if false_pass else "laundering"
        failed = "laundering" if false_pass else "false-authorization"
        wording = (
            f"The powered MES intersection-union test did not pass: only the {passed} "
            f"arm met its registered criterion, while the {failed} arm did not. "
            "The partial result is retained as mechanism-dependent descriptive evidence."
        )
    else:
        classification = "FAIL"
        wording = (
            "The powered, temporally non-overlapping MES test did not reproduce the "
            "registered co-primary trade-off. The frozen null or adverse result is "
            "retained as evidence of period or mechanism dependence."
        )
    return {"classification": classification, "permitted_summary": wording}


def execute_once(config_path: Path, freeze_path: Path) -> Path:
    config_path = config_path.resolve()
    freeze_path = freeze_path.resolve()
    config = load_config(config_path)
    _verify_freeze(freeze_path)
    registry_path = resolve_root_path(config["freeze"]["test_registry"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    started = {
        "project": "FinAuth-Audit",
        "version": "0.4.0",
        "status": "STARTED",
        "started_at": _timestamp(),
        "freeze_manifest": str(freeze_path.relative_to(ROOT)),
        "freeze_manifest_sha256": sha256(freeze_path),
        "community_hidden_outcomes_evaluated": False,
    }
    try:
        with registry_path.open("x", encoding="utf-8") as handle:
            json.dump(started, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError("Databento v0.4 test registry exists; rerun prohibited") from exc

    try:
        source = config["databento"]
        derived_dir = resolve_root_path(source["derived_dir"])
        features = pd.read_parquet(derived_dir / "paper_test.parquet")
        sealed = pd.read_parquet(derived_dir / "paper_test_outcomes.parquet")
        if "outcome_timestamp" in features.columns:
            raise RuntimeError("post-decision outcome_timestamp is present in feature rows")
        frame = materialize_outcomes(features, sealed, config, "databento")
        result = evaluate_frame(frame, config, "databento-v04")
        output_dir = resolve_root_path(config["results_dir"]) / "paper_test"
        output_hashes = _write_outputs(output_dir, result, save_row_level=False)
        interpretation = _result_wording(result["primary"])
        manifest = {
            "project": "FinAuth-Audit",
            "version": "0.4.0",
            "status": "COMPLETED",
            "started_at": started["started_at"],
            "completed_at": _timestamp(),
            "freeze_manifest": started["freeze_manifest"],
            "freeze_manifest_sha256": started["freeze_manifest_sha256"],
            "paper_test_outcomes_evaluated": True,
            "community_hidden_outcomes_evaluated": False,
            "classification": "confirmatory_temporal_replication",
            "known_v03_direction_before_registration": True,
            "v03_binance_intersection_union_passed": False,
            "primary_result": result["primary"],
            "result_interpretation": interpretation,
            "outputs": output_hashes,
            "claim_boundary": (
                "One-time powered temporal replication of the frozen v0.3 Databento "
                "protocol. All null, adverse, or mixed results are retained. It must "
                "not be conflated with the failed v0.3 Binance co-primary result. "
                "Community-hidden outcomes remain unevaluated."
            ),
        }
        write_json(registry_path, manifest)
        os.chmod(registry_path, 0o444)
        print(registry_path)
        return registry_path
    except Exception as exc:
        failure = {
            **started,
            "status": "FAILED_RETAINED_NO_RERUN",
            "failed_at": _timestamp(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "paper_test_outcomes_may_have_been_read": True,
            "community_hidden_outcomes_evaluated": False,
        }
        write_json(registry_path, failure)
        os.chmod(registry_path, 0o444)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute the frozen Databento v0.4 paper test exactly once."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "databento_powered_v04.yaml")
    )
    parser.add_argument("--execute-frozen-test", action="store_true")
    parser.add_argument("--freeze-manifest")
    args = parser.parse_args()
    if not args.execute_frozen_test or not args.freeze_manifest:
        raise SystemExit("--execute-frozen-test and --freeze-manifest are required")
    execute_once(Path(args.config), Path(args.freeze_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
