from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


REPO = ROOT.parent

STATIC_SURFACE = (
    "pyproject.toml",
    "VERSION",
    "requirements-lock.txt",
    "Dockerfile",
    ".dockerignore",
    "configs/real_agent_v06.yaml",
    "manifests/real_agent_v06_design_freeze.json",
    "manifests/preregistration/real_agent_v06.yaml",
    "manifests/preregistration/real_agent_v06_malformed_output_amendment.json",
    "manifests/preregistration/real_agent_v06_historical_memory_amendment.json",
    "manifests/preregistration/real_agent_v06_historical_memory_merge_correction.json",
    "manifests/preregistration/real_agent_v06_recalibration_infeasibility_amendment.json",
    "manifests/preregistration/real_agent_v06_context_attachment_correction.json",
    "manifests/preregistration/real_agent_v06_context_attachment_test_fixture_correction.json",
    "manifests/preregistration/real_agent_v06_context_registry_correction.json",
    "manifests/preregistration/real_agent_v06_outcome_identity_merge_correction.json",
    "manifests/preregistration/real_agent_v06_freeze_chain_correction.json",
    "manifests/preregistration/real_agent_v06_freeze_chain_fixture_correction.json",
    "manifests/real_agent_v06_feature_access.json",
    "schemas/real_agent_proposals_v06.schema.json",
    "docs/real_agent_multitask_protocol_v06.md",
    "docs/real_agent_v06_implementation.md",
    "generators/external_orderbook_v03.py",
    "generators/fetch_real_agent_v06.py",
    "generators/build_real_agent_contexts_v06.py",
    "generators/generate_real_agent_proposals_v06.py",
    "evaluation/real_agent_v06_common.py",
    "evaluation/real_agent_v06_structural_audit.py",
    "evaluation/real_agent_v06_calibrate.py",
    "evaluation/real_agent_v06_historical_memory_audit.py",
    "evaluation/rank_transfer_robustness.py",
    "evaluation/real_agent_v06_freeze.py",
    "evaluation/real_agent_v06_test.py",
    "tests/test_real_agent_v06_design.py",
    "tests/test_real_agent_v06_source_context.py",
    "tests/test_real_agent_v06_generation.py",
    "tests/test_real_agent_v06_evaluation.py",
    "tests/test_real_agent_v06_inference.py",
    "tests/test_real_agent_v06_historical_memory_audit.py",
    "reviews/prompts/real_agent_v06_historical_memory_review.md",
    "reviews/evidence/real_agent_v06_historical_memory_review_gateway_failed.md",
    "reviews/evidence/real_agent_v06_historical_memory_review_raw.md",
    "reviews/prompts/real_agent_v06_historical_memory_followup_review.md",
    "reviews/evidence/real_agent_v06_historical_memory_followup_review_raw.md",
    "reviews/prompts/real_agent_v06_historical_memory_final_review.md",
    "reviews/evidence/real_agent_v06_historical_memory_final_review_budget10_failed.md",
    "reviews/evidence/real_agent_v06_historical_memory_final_review_raw.md",
    "reviews/prompts/real_agent_v06_historical_memory_approval_review.md",
    "reviews/evidence/real_agent_v06_historical_memory_approval_review_raw.md",
    "reviews/real_agent_v06_historical_memory_review.md",
    "reviews/evidence/real_agent_v06_historical_memory_runtime_failure.md",
    "reviews/prompts/real_agent_v06_historical_memory_merge_correction_review.md",
    "reviews/evidence/real_agent_v06_historical_memory_merge_correction_review_raw.md",
    "reviews/real_agent_v06_historical_memory_merge_correction_review.md",
    "reviews/evidence/real_agent_v06_recalibration_coverage_precheck.md",
    "reviews/prompts/real_agent_v06_recalibration_infeasibility_review.md",
    "reviews/evidence/real_agent_v06_recalibration_infeasibility_review_raw.md",
    "reviews/prompts/real_agent_v06_recalibration_implementation_review.md",
    "reviews/evidence/real_agent_v06_recalibration_implementation_review_raw.md",
    "reviews/real_agent_v06_recalibration_review.md",
    "reviews/evidence/real_agent_v06_development_calibration_runtime_failure.md",
    "reviews/prompts/real_agent_v06_context_attachment_review.md",
    "reviews/evidence/real_agent_v06_context_attachment_review_raw.md",
    "reviews/prompts/real_agent_v06_context_attachment_implementation_review.md",
    "reviews/evidence/real_agent_v06_context_attachment_implementation_review_raw.md",
    "reviews/real_agent_v06_context_attachment_review.md",
    "reviews/evidence/real_agent_v06_context_attachment_fixture_test_failure.md",
    "reviews/prompts/real_agent_v06_context_attachment_fixture_review.md",
    "reviews/evidence/real_agent_v06_context_attachment_fixture_review_raw.md",
    "reviews/real_agent_v06_context_attachment_fixture_review.md",
    "reviews/evidence/real_agent_v06_context_registry_runtime_failure.md",
    "reviews/prompts/real_agent_v06_context_registry_correction_review.md",
    "reviews/evidence/real_agent_v06_context_registry_correction_review_raw.md",
    "reviews/prompts/real_agent_v06_context_registry_implementation_review.md",
    "reviews/evidence/real_agent_v06_context_registry_implementation_review_raw.md",
    "reviews/real_agent_v06_context_registry_review.md",
    "reviews/evidence/real_agent_v06_outcome_identity_merge_runtime_failure.md",
    "reviews/prompts/real_agent_v06_outcome_identity_merge_review.md",
    "reviews/evidence/real_agent_v06_outcome_identity_merge_review_raw.md",
    "reviews/prompts/real_agent_v06_outcome_identity_merge_implementation_review.md",
    "reviews/evidence/real_agent_v06_outcome_identity_merge_implementation_review_raw.md",
    "reviews/prompts/real_agent_v06_outcome_identity_merge_followup_review.md",
    "reviews/evidence/real_agent_v06_outcome_identity_merge_followup_review_raw.md",
    "reviews/real_agent_v06_outcome_identity_merge_review.md",
    "reviews/evidence/real_agent_v06_freeze_chain_runtime_failure.md",
    "reviews/prompts/real_agent_v06_freeze_chain_review.md",
    "reviews/evidence/real_agent_v06_freeze_chain_review_raw.md",
    "reviews/prompts/real_agent_v06_freeze_chain_followup_review.md",
    "reviews/evidence/real_agent_v06_freeze_chain_followup_review_raw.md",
    "reviews/real_agent_v06_freeze_chain_review.md",
    "reviews/evidence/real_agent_v06_freeze_chain_fixture_failure.md",
    "reviews/prompts/real_agent_v06_freeze_chain_fixture_review.md",
    "reviews/evidence/real_agent_v06_freeze_chain_fixture_review_raw.md",
    "reviews/prompts/real_agent_v06_freeze_chain_fixture_implementation_review.md",
    "reviews/evidence/real_agent_v06_freeze_chain_fixture_implementation_review_raw.md",
    "reviews/real_agent_v06_freeze_chain_fixture_review.md",
    "reviews/prompts/real_agent_v06_implementation_review.md",
    "reviews/evidence/real_agent_v06_implementation_review_budget8_failed.md",
    "reviews/evidence/real_agent_v06_implementation_review_raw.md",
    "reviews/evidence/real_agent_v06_implementation_review.stderr.log",
    "reviews/real_agent_v06_implementation_review.md",
    "reviews/supervisor_round_84_implementation.md",
    "reviews/prompts/real_agent_v06_malformed_output_review.md",
    "reviews/evidence/real_agent_v06_malformed_output_review_raw.md",
    "reviews/prompts/real_agent_v06_malformed_output_followup_review.md",
    "reviews/evidence/real_agent_v06_malformed_output_followup_review_raw.md",
    "reviews/real_agent_v06_malformed_output_review.md",
)


def _timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _run_gate(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "passed": result.returncode == 0,
    }


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o444
    archive.addfile(info, io.BytesIO(payload))


def _archive(
    paths: list[str], metadata: dict[str, object], output_dir: Path
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finauth-real-agent-v06-freeze-") as temporary:
        tar_path = Path(temporary) / "snapshot.tar"
        with tarfile.open(tar_path, "w") as archive:
            _add_bytes(
                archive,
                "SNAPSHOT_CONTENTS.json",
                (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            for relative in sorted(paths):
                _add_bytes(archive, relative, resolve_root_path(relative).read_bytes())
        compressed = Path(temporary) / "snapshot.tar.gz"
        with tar_path.open("rb") as source, compressed.open("wb") as destination:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as handle:
                shutil.copyfileobj(source, handle)
        digest = sha256(compressed)
        final = output_dir / f"real_agent_v06_freeze_{digest}.tar.gz"
        if not final.exists():
            shutil.copy2(compressed, final)
        if sha256(final) != digest:
            raise RuntimeError("content-addressed v0.6 freeze archive mismatch")
        os.chmod(final, 0o444)
    return final, digest


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _dataset_surface(config: dict[str, Any], dataset: dict[str, Any]) -> list[str]:
    derived = resolve_root_path(config["binance"]["derived_dir"])
    paths = [
        _relative(derived / "split_registry.csv"),
        *[
            _relative(derived / f"{split}_{suffix}")
            for split in ("development", "paper_test", "community_hidden")
            for suffix in ("contexts.csv", "outcomes.parquet")
        ],
    ]
    declared = dataset.get("outputs", {})
    if isinstance(declared, dict):
        paths.extend(str(value) for value in declared.keys())
    elif isinstance(declared, list):
        paths.extend(str(value) for value in declared)
    return list(dict.fromkeys(paths))


def _proposal_surface(config: dict[str, Any], proposals: dict[str, Any]) -> list[str]:
    paths = [
        str(proposals["proposal_file"]),
        str(config["generation"]["hidden_ciphertext"]),
        str(config["generation"]["hidden_manifest"]),
    ]
    for record in proposals.get("raw_outputs", []):
        raw_path = record.get("raw_path")
        if raw_path:
            paths.append(str(raw_path))
            log_path = str(Path(raw_path).with_suffix(".log"))
            if resolve_root_path(log_path).is_file():
                paths.append(log_path)
    return list(dict.fromkeys(paths))


def _verify_design_surface(config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_root_path(config["freeze"]["design_freeze_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_SOURCE_ACQUISITION":
        raise RuntimeError("v0.6 design freeze is not active")
    for relative, expected in manifest.get("surface_hashes", {}).items():
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"frozen v0.6 design changed: {relative}")
    return manifest


def _verify_historical_memory_amendment(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = ROOT / "manifests" / "preregistration" / "real_agent_v06_historical_memory_amendment.json"
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_BEFORE_ANY_OUTCOME_EVALUATION":
        raise RuntimeError("v0.6 historical-memory amendment is not active")
    correction_path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_historical_memory_merge_correction.json"
    )
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    if correction.get("status") != "HASH_LOCKED_AFTER_OUTPUT_BLIND_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 historical-memory merge correction is not active")
    later_superseded = set(extra_superseded or set())
    superseded = set(correction.get("superseded_surface_paths", [])) | later_superseded
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 historical-memory amendment changed: {relative}")
    for relative, expected in correction.get("surface_hashes", {}).items():
        if relative in later_superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 historical-memory merge correction changed: {relative}")
    return {"base": amendment, "correction": correction}


def _verify_malformed_output_amendment(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = ROOT / "manifests" / "preregistration" / "real_agent_v06_malformed_output_amendment.json"
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_BEFORE_GENERATION_RESUME":
        raise RuntimeError("v0.6 malformed-output amendment is not active")
    superseded = set(extra_superseded or set())
    for relative, expected in amendment.get("hash_locked_protocol_surface", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 malformed-output amendment changed: {relative}")
    return amendment


def _verify_recalibration_infeasibility_amendment(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_recalibration_infeasibility_amendment.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_BEFORE_DEVELOPMENT_OUTCOME_ACCESS":
        raise RuntimeError("v0.6 recalibration infeasibility amendment is not active")
    superseded = set(extra_superseded or set())
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 recalibration infeasibility amendment changed: {relative}")
    return amendment


def _verify_context_attachment_correction(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_context_attachment_correction.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_AFTER_DEVELOPMENT_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 context-attachment correction is not active")
    superseded = set(extra_superseded or set())
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 context-attachment correction changed: {relative}")
    return amendment


def _verify_context_attachment_fixture_correction(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_context_attachment_test_fixture_correction.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_AFTER_FIXTURE_ONLY_TEST_FAILURE":
        raise RuntimeError("v0.6 context-attachment fixture correction is not active")
    superseded = set(extra_superseded or set())
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 context-attachment fixture correction changed: {relative}")
    return amendment


def _verify_context_registry_correction(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_context_registry_correction.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_AFTER_CONTEXT_REGISTRY_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 context-registry correction is not active")
    superseded = set(extra_superseded or set())
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 context-registry correction changed: {relative}")
    return amendment


def _verify_outcome_identity_merge_correction(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_outcome_identity_merge_correction.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_AFTER_OUTCOME_IDENTITY_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 outcome-identity merge correction is not active")
    superseded = set(extra_superseded or set())
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 outcome-identity merge correction changed: {relative}")
    return amendment


def _verify_freeze_chain_correction(
    extra_superseded: set[str] | None = None,
) -> dict[str, Any]:
    path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_freeze_chain_correction.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_AFTER_FULL_FREEZE_RUNTIME_FAILURE":
        raise RuntimeError("v0.6 full-freeze chain correction is not active")
    superseded = set(extra_superseded or set())
    for relative, expected in amendment.get("surface_hashes", {}).items():
        if relative in superseded:
            continue
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 full-freeze chain correction changed: {relative}")
    return amendment


def _verify_freeze_chain_fixture_correction() -> dict[str, Any]:
    path = (
        ROOT
        / "manifests"
        / "preregistration"
        / "real_agent_v06_freeze_chain_fixture_correction.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("status") != "HASH_LOCKED_AFTER_FREEZE_CHAIN_FIXTURE_FAILURE":
        raise RuntimeError("v0.6 freeze-chain fixture correction is not active")
    for relative, expected in amendment.get("surface_hashes", {}).items():
        current = resolve_root_path(relative)
        if not current.is_file() or sha256(current) != expected:
            raise RuntimeError(f"v0.6 freeze-chain fixture correction changed: {relative}")
    return amendment


def freeze(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    freeze_path = resolve_root_path(config["freeze"]["freeze_manifest"])
    registry_path = resolve_root_path(config["freeze"]["test_registry"])
    if freeze_path.exists():
        raise FileExistsError("v0.6 paper-test freeze already exists")
    if registry_path.exists():
        raise FileExistsError("v0.6 paper-test registry exists before freeze")
    design = _verify_design_surface(config)
    freeze_chain_fixture_amendment = _verify_freeze_chain_fixture_correction()
    freeze_chain_fixture_superseded = set(
        freeze_chain_fixture_amendment.get("superseded_surface_paths", [])
    )
    freeze_chain_amendment = _verify_freeze_chain_correction(
        freeze_chain_fixture_superseded
    )
    freeze_chain_superseded = set(
        freeze_chain_amendment.get("superseded_surface_paths", [])
    ) | freeze_chain_fixture_superseded
    outcome_identity_amendment = _verify_outcome_identity_merge_correction(
        freeze_chain_superseded
    )
    outcome_superseded = set(
        outcome_identity_amendment.get("superseded_surface_paths", [])
    ) | freeze_chain_superseded
    context_registry_amendment = _verify_context_registry_correction(
        outcome_superseded
    )
    registry_superseded = set(
        context_registry_amendment.get("superseded_surface_paths", [])
    ) | outcome_superseded
    context_attachment_fixture = _verify_context_attachment_fixture_correction(
        registry_superseded
    )
    fixture_superseded = set(
        context_attachment_fixture.get("superseded_surface_paths", [])
    ) | registry_superseded
    context_attachment_amendment = _verify_context_attachment_correction(
        fixture_superseded
    )
    context_superseded = set(
        context_attachment_amendment.get("superseded_surface_paths", [])
    ) | fixture_superseded
    recalibration_amendment = _verify_recalibration_infeasibility_amendment(
        context_superseded
    )
    superseded = set(recalibration_amendment.get("superseded_surface_paths", [])) | context_superseded
    historical_memory_amendment = _verify_historical_memory_amendment(superseded)
    malformed_output_amendment = _verify_malformed_output_amendment(superseded)

    source_manifest_path = resolve_root_path(config["binance"]["source_manifest"])
    dataset_manifest_path = resolve_root_path(config["binance"]["dataset_manifest"])
    proposal_manifest_path = resolve_root_path(config["binance"]["proposal_manifest"])
    structural_path = resolve_root_path(config["freeze"]["structural_audit"])
    calibration_path = resolve_root_path(config["freeze"]["development_calibration"])
    historical_memory_path = (
        resolve_root_path(config["results_dir"])
        / "historical_memory_audit"
        / "manifest.json"
    )
    required = [
        source_manifest_path,
        dataset_manifest_path,
        proposal_manifest_path,
        structural_path,
        historical_memory_path,
        calibration_path,
    ]
    missing_required = [str(path) for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(f"v0.6 freeze prerequisites are missing: {missing_required}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    historical_memory = json.loads(
        historical_memory_path.read_text(encoding="utf-8")
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if dataset.get("outcome_metrics_computed") is not False:
        raise RuntimeError("dataset manifest crossed the pre-result boundary")
    if proposals.get("outcome_fields_read") is not False:
        raise RuntimeError("proposal manifest crossed the pre-result boundary")
    if structural.get("passed") is not True or structural.get("outcome_files_read") is not False:
        raise RuntimeError("v0.6 structural gate is not satisfied")
    if historical_memory.get("status") != "COMPLETED_BEFORE_ANY_OUTCOME_EVALUATION":
        raise RuntimeError("v0.6 historical-memory audit is not outcome-blind")
    if any(
        historical_memory.get(field) is not False
        for field in (
            "development_outcomes_read",
            "paper_test_outcomes_read",
            "community_hidden_outcomes_read",
            "community_hidden_proposals_decrypted",
            "rationale_text_persisted_in_outputs",
        )
    ):
        raise RuntimeError("v0.6 historical-memory audit crossed a prohibited boundary")
    if calibration.get("status") != "FROZEN_BEFORE_PAPER_TEST":
        raise RuntimeError("v0.6 development calibration is not frozen")
    if calibration.get("paper_test_outcomes_read") is not False:
        raise RuntimeError("development calibration read paper-test outcomes")
    if calibration.get("community_hidden_outcomes_read") is not False:
        raise RuntimeError("development calibration read hidden outcomes")
    if proposals.get("community_hidden_plaintext_in_repository") is not False:
        raise RuntimeError("hidden proposal plaintext entered the repository")

    if source_manifest.get("complete_registered_request") is not True:
        raise RuntimeError("v0.6 source manifest is not the complete registered request")
    for record in source_manifest.get("records", []):
        if record.get("status") not in {"CACHED_VERIFIED", "DOWNLOADED_VERIFIED"}:
            continue
        relative = record.get("path")
        expected = record.get("sha256")
        if not relative or not expected:
            raise RuntimeError(f"verified source record lacks path/hash: {record}")
        path = resolve_root_path(relative)
        checksum_path = path.with_suffix(path.suffix + ".CHECKSUM")
        if not path.is_file() or sha256(path) != expected or not checksum_path.is_file():
            raise RuntimeError(f"verified source record changed before freeze: {relative}")
        checksum_parts = checksum_path.read_text(encoding="utf-8").split()
        if not checksum_parts or checksum_parts[0].lower() != str(expected).lower():
            raise RuntimeError(f"source checksum payload changed before freeze: {relative}")

    declared_outputs = dataset.get("outputs", {})
    if isinstance(declared_outputs, dict):
        for relative, expected in declared_outputs.items():
            path = resolve_root_path(relative)
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"dataset output hash mismatch before freeze: {relative}")
    for record in proposals.get("raw_outputs", []):
        relative = record.get("raw_path")
        expected = record.get("raw_sha256")
        if relative and expected:
            path = resolve_root_path(relative)
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"proposal raw-output hash mismatch before freeze: {relative}")

    dynamic = [
        _relative(source_manifest_path),
        _relative(dataset_manifest_path),
        _relative(proposal_manifest_path),
        _relative(structural_path),
        _relative(historical_memory_path),
        _relative(calibration_path),
        *_dataset_surface(config, dataset),
        *_proposal_surface(config, proposals),
        *[
            _relative(historical_memory_path.parent / relative)
            for relative in historical_memory.get("outputs", {})
        ],
    ]
    surface = list(dict.fromkeys((*STATIC_SURFACE, *dynamic)))
    missing = [relative for relative in surface if not resolve_root_path(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v0.6 freeze surface is incomplete: {missing}")

    focused_tests = _run_gate(
        [
            str(REPO / ".venv" / "bin" / "pytest"),
            "-q",
            *[
                str(ROOT / "tests" / name)
                for name in (
                    "test_real_agent_v06_design.py",
                    "test_real_agent_v06_source_context.py",
                    "test_real_agent_v06_generation.py",
                    "test_real_agent_v06_evaluation.py",
                    "test_real_agent_v06_inference.py",
                    "test_real_agent_v06_historical_memory_audit.py",
                )
            ],
        ]
    )
    if not focused_tests["passed"]:
        raise RuntimeError(f"v0.6 focused tests failed: {focused_tests}")

    surface_hashes = {relative: sha256(resolve_root_path(relative)) for relative in surface}
    metadata = {
        "project": "FinAuth-Audit",
        "version": "0.6.0",
        "status": "FROZEN_BEFORE_V06_PAPER_TEST",
        "surface_hashes": surface_hashes,
        "source_manifest_sha256": sha256(source_manifest_path),
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "proposal_manifest_sha256": sha256(proposal_manifest_path),
        "structural_audit_sha256": sha256(structural_path),
        "historical_memory_audit_sha256": sha256(historical_memory_path),
        "development_calibration_sha256": sha256(calibration_path),
        "design_freeze_sha256": sha256(
            resolve_root_path(config["freeze"]["design_freeze_manifest"])
        ),
        "historical_memory_amendment_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_historical_memory_amendment.json"
        ),
        "recalibration_infeasibility_amendment_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_recalibration_infeasibility_amendment.json"
        ),
        "context_attachment_correction_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_context_attachment_correction.json"
        ),
        "context_attachment_fixture_correction_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_context_attachment_test_fixture_correction.json"
        ),
        "context_registry_correction_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_context_registry_correction.json"
        ),
        "outcome_identity_merge_correction_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_outcome_identity_merge_correction.json"
        ),
        "freeze_chain_correction_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_freeze_chain_correction.json"
        ),
        "freeze_chain_fixture_correction_sha256": sha256(
            ROOT
            / "manifests"
            / "preregistration"
            / "real_agent_v06_freeze_chain_fixture_correction.json"
        ),
        "paper_test_outcomes_evaluated": False,
        "community_hidden_outcomes_evaluated": False,
        "hidden_proposals_decrypted": False,
        "prohibitions": [
            "no rule-form, threshold, endpoint, split, proposal, or outcome change after freeze",
            "no paper-test rerun",
            "no community-hidden outcome evaluation or proposal decryption",
            "no outcome-conditioned regeneration, recalibration, or result suppression",
        ],
    }
    archive_paths = [
        relative
        for relative in surface
        if not relative.endswith("community_hidden_outcomes.parquet")
        and not relative.startswith(str(config["binance"]["raw_dir"]))
    ]
    archive_path, archive_hash = _archive(
        archive_paths,
        metadata,
        resolve_root_path(config["freeze"]["archive_dir"]),
    )
    manifest = {
        **metadata,
        "frozen_at": _timestamp(),
        "archive_path": _relative(archive_path),
        "archive_sha256": archive_hash,
        "archive_excludes": [
            "community_hidden_outcomes.parquet",
            "raw Binance archives (hashes retained in source manifest)",
            "hidden decryption key",
        ],
        "focused_test_gate": focused_tests,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "design_archive_sha256": design.get("archive_sha256"),
        "historical_memory_review_verdict": historical_memory_amendment["base"].get(
            "fable_final_verdict"
        ),
        "historical_memory_correction_verdict": historical_memory_amendment[
            "correction"
        ].get("fable_final_verdict"),
        "recalibration_infeasibility_verdict": recalibration_amendment.get(
            "fable_final_verdict"
        ),
        "context_attachment_verdict": context_attachment_amendment.get(
            "fable_final_verdict"
        ),
        "context_attachment_fixture_verdict": context_attachment_fixture.get(
            "fable_final_verdict"
        ),
        "context_registry_verdict": context_registry_amendment.get(
            "fable_final_verdict"
        ),
        "outcome_identity_merge_verdict": outcome_identity_amendment.get(
            "fable_final_verdict"
        ),
        "freeze_chain_verdict": freeze_chain_amendment.get(
            "fable_final_verdict"
        ),
        "freeze_chain_fixture_verdict": freeze_chain_fixture_amendment.get(
            "fable_final_verdict"
        ),
        "malformed_output_amendment_status": malformed_output_amendment.get("status"),
        "claim_boundary": (
            "Content-addressed v0.6 freeze after development-only calibration and "
            "before the one-time 200-date paper test."
        ),
    }
    write_json(freeze_path, manifest)
    os.chmod(freeze_path, 0o444)
    print(freeze_path)
    print(archive_path)
    return freeze_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the v0.6 one-time paper-test surface.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_agent_v06.yaml"))
    args = parser.parse_args()
    freeze(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
