from __future__ import annotations

import argparse
import gzip
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

FREEZE_SURFACE = (
    "configs/main.yaml",
    "configs/provenance_main.yaml",
    "configs/training_utility_smoke.yaml",
    "configs/training_robustness_v03.yaml",
    "baselines/rules.py",
    "baselines/provenance_rules.py",
    "evaluation/paper_test.py",
    "evaluation/paper_test_partition.py",
    "evaluation/paper_test_freeze.py",
    "evaluation/metrics.py",
    "evaluation/laundering_metrics.py",
    "evaluation/coverage_audit.py",
    "evaluation/provenance_audit.py",
    "evaluation/certification_surface.py",
    "evaluation/certification_robustness.py",
    "evaluation/review_workload.py",
    "evaluation/baseline_governance.py",
    "evaluation/cluster_bootstrap.py",
    "evaluation/feature_access_audit.py",
    "evaluation/seeds.py",
    "manifests/feature_access.json",
    "manifests/seed_registry.json",
    "manifests/main_data_manifest.json",
    "manifests/provenance_main_manifest.json",
    "manifests/paper_test_partition/manifest.json",
    "manifests/paper_test_partition/paper_test_clusters.csv",
    "manifests/paper_test_partition/community_hidden_clusters.csv",
    "docs/test_freeze_protocol.md",
    "docs/test_freeze_protocol_amendment_round_75.md",
    "docs/baseline_governance.md",
    "docs/generator_robustness_protocol.md",
    "docs/training_robustness_v03.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


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


def _build_deterministic_archive(
    surface_hashes: dict[str, str],
    snapshot_metadata: dict[str, object],
) -> tuple[Path, str]:
    snapshot_dir = ROOT / "manifests" / "paper_test_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finauth-freeze-") as temporary:
        tar_path = Path(temporary) / "snapshot.tar"
        with tarfile.open(tar_path, "w") as archive:
            _add_bytes(
                archive,
                "SNAPSHOT_CONTENTS.json",
                (json.dumps(snapshot_metadata, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
            for relative in sorted(surface_hashes):
                _add_bytes(archive, relative, (ROOT / relative).read_bytes())
        temporary_gzip = Path(temporary) / "snapshot.tar.gz"
        with tar_path.open("rb") as source, temporary_gzip.open("wb") as destination:
            with gzip.GzipFile(
                filename="", fileobj=destination, mode="wb", mtime=0
            ) as compressor:
                shutil.copyfileobj(source, compressor)
        archive_hash = sha256(temporary_gzip)
        final_path = snapshot_dir / f"paper_test_freeze_{archive_hash}.tar.gz"
        if final_path.exists() and sha256(final_path) != archive_hash:
            raise RuntimeError("content-addressed archive collision")
        if not final_path.exists():
            shutil.copy2(temporary_gzip, final_path)
        os.chmod(final_path, 0o444)
    return final_path, archive_hash


def _run_gate(command: list[str]) -> dict[str, object]:
    result = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "passed": result.returncode == 0,
    }


def freeze() -> Path:
    manifest_path = ROOT / "manifests" / "paper_test_freeze.json"
    registry_path = ROOT / "manifests" / "paper_test_registry.json"
    if manifest_path.exists():
        raise FileExistsError("paper-test freeze already exists")
    if registry_path.exists():
        raise FileExistsError("paper-test registry exists before freeze")

    partition_path = ROOT / "manifests" / "paper_test_partition" / "manifest.json"
    rehearsal_path = ROOT / "results" / "paper_test_rehearsal" / "manifest.json"
    if not partition_path.exists() or not rehearsal_path.exists():
        raise FileNotFoundError("partition and validation rehearsal must exist before freeze")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if partition.get("outcome_fields_inspected") is not False:
        raise RuntimeError("partition outcome-field boundary is not satisfied")
    if partition.get("paper_test_outcomes_evaluated") is not False:
        raise RuntimeError("paper-test outcomes were evaluated before freeze")
    if partition.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("community-hidden outcomes were evaluated before freeze")
    if rehearsal.get("equivalence_passed") is not True:
        raise RuntimeError("validation rehearsal did not reproduce registered outputs")
    if rehearsal.get("test_outcomes_evaluated") is not False:
        raise RuntimeError("rehearsal accessed test outcomes")

    missing = [relative for relative in FREEZE_SURFACE if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"freeze surface is incomplete: {missing}")

    focused_tests = _run_gate(
        [
            str(REPO / ".venv" / "bin" / "pytest"),
            "-q",
            str(ROOT / "tests" / "test_paper_test_partition.py"),
            str(ROOT / "tests" / "test_paper_test.py"),
        ]
    )
    strict_verifier = _run_gate(
        [
            str(REPO / ".venv" / "bin" / "python"),
            str(ROOT / "verify_artifact.py"),
            "--phase",
            "main",
            "--run-tests",
        ]
    )
    if not focused_tests["passed"] or not strict_verifier["passed"]:
        raise RuntimeError(
            "pre-freeze gate failed: "
            f"focused={focused_tests['returncode']} strict={strict_verifier['returncode']}"
        )

    surface_hashes = {
        relative: sha256(ROOT / relative) for relative in FREEZE_SURFACE
    }
    bound_data_hashes = {
        relative: expected
        for relative, expected in partition["files"].items()
    }
    snapshot_metadata = {
        "project": "FinAuth-Audit",
        "status": "FROZEN_BEFORE_PAPER_TEST",
        "surface_hashes": surface_hashes,
        "bound_data_hashes": bound_data_hashes,
        "partition_manifest_sha256": sha256(partition_path),
        "rehearsal_manifest_sha256": sha256(rehearsal_path),
        "prohibitions": [
            "no generator changes after inspection",
            "no rule or threshold changes after inspection",
            "no metric or certification-profile changes after inspection",
            "no community-hidden evaluation",
            "no public-test evaluation",
            "no FinAuth-Worlds hidden-period access",
        ],
    }
    archive_path, archive_hash = _build_deterministic_archive(
        surface_hashes, snapshot_metadata
    )
    manifest = {
        **snapshot_metadata,
        "frozen_at": _timestamp(),
        "archive_path": str(archive_path.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "focused_test_gate": focused_tests,
        "strict_verifier_gate": strict_verifier,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "paper_test_clusters": partition["paper_test_clusters"],
        "community_hidden_clusters": partition["community_hidden_clusters"],
        "paper_test_outcomes_evaluated": False,
        "community_hidden_outcomes_evaluated": False,
        "claim_boundary": (
            "Immutable content-addressed freeze before the one-time controlled and "
            "provenance paper test. It does not authorize public, community-hidden, "
            "or FinAuth-Worlds hidden evaluation."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(manifest_path, 0o444)
    print(manifest_path)
    print(archive_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a content-addressed pre-paper-test freeze."
    )
    parser.parse_args()
    freeze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
