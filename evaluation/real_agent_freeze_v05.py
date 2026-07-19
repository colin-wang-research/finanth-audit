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
    "configs/real_agent_v05.yaml",
    "manifests/preregistration/real_agent_v05.yaml",
    "manifests/preregistration/real_agent_v05_transport_amendment.json",
    "manifests/preregistration/real_agent_v05_claude_envelope_amendment.json",
    "manifests/preregistration/real_agent_v05_claude_result_fallback_amendment.json",
    "manifests/preregistration/real_agent_v05_schema_metadata_amendment.json",
    "manifests/real_agent_v05_feature_access.json",
    "schemas/real_agent_proposals_v05.schema.json",
    "docs/real_agent_proposal_protocol_v05.md",
    "baselines/rules.py",
    "evaluation/metrics.py",
    "evaluation/cluster_bootstrap.py",
    "evaluation/seeds.py",
    "evaluation/real_agent_structural_audit_v05.py",
    "evaluation/real_agent_freeze_v05.py",
    "evaluation/real_agent_test_v05.py",
    "generators/external_orderbook_v03.py",
    "generators/fetch_binance_depth_v03.py",
    "generators/build_binance_depth_v03.py",
    "generators/build_real_agent_contexts_v05.py",
    "generators/generate_real_agent_proposals_v05.py",
    "tests/test_real_agent_v05.py",
    "reviews/evidence/real_agent_v05_design_review_raw.md",
    "reviews/evidence/real_agent_v05_transport_amendment_review_raw.md",
    "reviews/evidence/real_agent_v05_claude_envelope_review_raw.md",
    "reviews/evidence/real_agent_v05_claude_result_fallback_review_raw.md",
    "reviews/evidence/real_agent_v05_schema_compatibility_review_raw.md",
    "reviews/real_agent_v05_design_review.md",
    "reviews/real_agent_v05_transport_amendment_review.md",
    "reviews/real_agent_v05_claude_envelope_review.md",
    "reviews/real_agent_v05_claude_result_fallback_review.md",
    "reviews/real_agent_v05_schema_compatibility_review.md",
    "reviews/supervisor_round_83_design.md",
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
    archive_paths: list[str], metadata: dict[str, object], output_dir: Path
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finauth-real-agent-v05-") as temporary:
        tar_path = Path(temporary) / "snapshot.tar"
        with tarfile.open(tar_path, "w") as archive:
            _add_bytes(
                archive,
                "SNAPSHOT_CONTENTS.json",
                (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            for relative in sorted(archive_paths):
                _add_bytes(archive, relative, resolve_root_path(relative).read_bytes())
        compressed = Path(temporary) / "snapshot.tar.gz"
        with tar_path.open("rb") as source, compressed.open("wb") as destination:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as handle:
                shutil.copyfileobj(source, handle)
        digest = sha256(compressed)
        final = output_dir / f"real_agent_v05_freeze_{digest}.tar.gz"
        if not final.exists():
            shutil.copy2(compressed, final)
        if sha256(final) != digest:
            raise RuntimeError("content-addressed real-agent freeze archive mismatch")
        os.chmod(final, 0o444)
    return final, digest


def _dynamic_surface(config: dict[str, object]) -> tuple[list[str], dict[str, str]]:
    source = config["binance"]
    dataset_manifest_path = resolve_root_path(source["dataset_manifest"])
    proposal_manifest_path = resolve_root_path(source["proposal_manifest"])
    structural_path = resolve_root_path(config["freeze"]["structural_audit"])
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    paths = [
        str(resolve_root_path(source["source_manifest"]).relative_to(ROOT)),
        str(dataset_manifest_path.relative_to(ROOT)),
        str(proposal_manifest_path.relative_to(ROOT)),
        str(structural_path.relative_to(ROOT)),
        str((resolve_root_path(source["derived_dir"]) / "split_registry.csv").relative_to(ROOT)),
        str((ROOT / proposals["proposal_file"]).relative_to(ROOT)),
    ]
    paths.extend(str(value) for value in dataset["outputs"])
    paths.extend(str(record["raw_path"]) for record in proposals["raw_outputs"])
    unique = list(dict.fromkeys(paths))
    return unique, {str(key): str(value) for key, value in dataset["outputs"].items()}


def freeze(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    freeze_path = resolve_root_path(config["freeze"]["freeze_manifest"])
    registry_path = resolve_root_path(config["freeze"]["test_registry"])
    if freeze_path.exists():
        raise FileExistsError("real-agent v0.5 freeze already exists")
    if registry_path.exists():
        raise FileExistsError("real-agent test registry exists before freeze")

    source = config["binance"]
    dataset_manifest_path = resolve_root_path(source["dataset_manifest"])
    proposal_manifest_path = resolve_root_path(source["proposal_manifest"])
    structural_path = resolve_root_path(config["freeze"]["structural_audit"])
    for required in (dataset_manifest_path, proposal_manifest_path, structural_path):
        if not required.is_file():
            raise FileNotFoundError(f"real-agent freeze prerequisite is missing: {required}")
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    if dataset.get("outcome_metrics_computed") is not False:
        raise RuntimeError("dataset manifest crossed the pre-result boundary")
    if proposals.get("outcome_fields_read") is not False:
        raise RuntimeError("proposal manifest crossed the pre-result boundary")
    if structural.get("passed") is not True or structural.get("outcome_files_read") is not False:
        raise RuntimeError("real-agent structural gate is not satisfied")
    if proposals.get("community_hidden_contexts_generated") != 0:
        raise RuntimeError("community-hidden contexts reached model generation")

    dynamic, dataset_hashes = _dynamic_surface(config)
    surface = list(dict.fromkeys((*STATIC_SURFACE, *dynamic)))
    missing = [relative for relative in surface if not resolve_root_path(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"real-agent freeze surface is incomplete: {missing}")
    for relative, expected in dataset_hashes.items():
        path = resolve_root_path(relative)
        if sha256(path) != expected:
            raise RuntimeError(f"dataset output hash mismatch before freeze: {relative}")

    focused_tests = _run_gate(
        [
            str(REPO / ".venv" / "bin" / "pytest"),
            "-q",
            str(ROOT / "tests" / "test_real_agent_v05.py"),
        ]
    )
    if not focused_tests["passed"]:
        raise RuntimeError(f"real-agent focused tests failed: {focused_tests}")

    surface_hashes = {relative: sha256(resolve_root_path(relative)) for relative in surface}
    metadata = {
        "project": "FinAuth-Audit",
        "version": "0.5.0",
        "status": "FROZEN_BEFORE_REAL_AGENT_TEST",
        "surface_hashes": surface_hashes,
        "dataset_hashes": dataset_hashes,
        "proposal_sha256": proposals["proposal_sha256"],
        "structural_audit_sha256": sha256(structural_path),
        "prohibitions": [
            "no prompt, model set, rule, threshold, metric, split, or outcome change after freeze",
            "no paper-test rerun",
            "no community-hidden model generation or evaluation",
            "no rationale inspection before this freeze",
            "no outcome-conditioned regeneration or result suppression",
        ],
    }
    archive_paths = [
        relative
        for relative in surface
        if not relative.endswith("community_hidden_outcomes.parquet")
    ]
    archive_path, archive_hash = _archive(
        archive_paths,
        metadata,
        resolve_root_path(config["freeze"]["archive_dir"]),
    )
    manifest = {
        **metadata,
        "frozen_at": _timestamp(),
        "archive_path": str(archive_path.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "archive_excludes": ["community_hidden_outcomes.parquet"],
        "focused_test_gate": focused_tests,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "paper_test_outcomes_evaluated": False,
        "community_hidden_outcomes_evaluated": False,
        "claim_boundary": (
            "Content-addressed v0.5 freeze before the one-time real-agent paper test. "
            "It does not authorize community-hidden generation or evaluation."
        ),
    }
    write_json(freeze_path, manifest)
    os.chmod(freeze_path, 0o444)
    print(freeze_path)
    print(archive_path)
    return freeze_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the real-agent v0.5 test surface.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_agent_v05.yaml"))
    args = parser.parse_args()
    freeze(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
