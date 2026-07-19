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

DESIGN_SURFACE = (
    "configs/real_agent_v06.yaml",
    "manifests/preregistration/real_agent_v06.yaml",
    "manifests/real_agent_v06_feature_access.json",
    "schemas/real_agent_proposals_v06.schema.json",
    "docs/real_agent_multitask_protocol_v06.md",
    "reviews/prompts/real_agent_v06_design_review.md",
    "reviews/evidence/real_agent_v06_design_review_raw.md",
    "reviews/real_agent_v06_design_review.md",
    "reviews/prompts/real_agent_v06_design_followup_review.md",
    "reviews/evidence/real_agent_v06_design_followup_review_raw.md",
    "reviews/real_agent_v06_design_followup_review.md",
    "reviews/supervisor_round_84_design.md",
    "plans/evening_plan_round_84.md",
    "evaluation/real_agent_v06_design_freeze.py",
    "tests/test_real_agent_v06_design.py",
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
    surface: tuple[str, ...], metadata: dict[str, object], output_dir: Path
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finauth-real-agent-v06-design-") as temporary:
        tar_path = Path(temporary) / "snapshot.tar"
        with tarfile.open(tar_path, "w") as archive:
            _add_bytes(
                archive,
                "SNAPSHOT_CONTENTS.json",
                (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            for relative in sorted(surface):
                _add_bytes(archive, relative, resolve_root_path(relative).read_bytes())
        compressed = Path(temporary) / "snapshot.tar.gz"
        with tar_path.open("rb") as source, compressed.open("wb") as destination:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as handle:
                shutil.copyfileobj(source, handle)
        digest = sha256(compressed)
        final = output_dir / f"real_agent_v06_design_{digest}.tar.gz"
        if not final.exists():
            shutil.copy2(compressed, final)
        if sha256(final) != digest:
            raise RuntimeError("content-addressed v0.6 design archive mismatch")
        os.chmod(final, 0o444)
    return final, digest


def _assert_pre_source_boundary(config: dict[str, object]) -> None:
    forbidden = [
        config["binance"]["source_manifest"],
        config["binance"]["dataset_manifest"],
        config["binance"]["proposal_manifest"],
        config["freeze"]["development_calibration"],
        config["freeze"]["freeze_manifest"],
        config["freeze"]["test_registry"],
        config["results_dir"],
    ]
    existing = [str(resolve_root_path(value)) for value in forbidden if resolve_root_path(value).exists()]
    if existing:
        raise RuntimeError(f"v0.6 design freeze crossed the pre-source boundary: {existing}")


def freeze(config_path: Path) -> Path:
    config = load_config(config_path.resolve())
    manifest_path = resolve_root_path(config["freeze"]["design_freeze_manifest"])
    if manifest_path.exists():
        raise FileExistsError("v0.6 design freeze already exists")

    preregistration = load_config(resolve_root_path(config["freeze"]["preregistration"]))
    followup = resolve_root_path(
        "reviews/evidence/real_agent_v06_design_followup_review_raw.md"
    ).read_text(encoding="utf-8")
    if preregistration.get("status") != "FROZEN_BEFORE_SOURCE_ACQUISITION":
        raise RuntimeError("v0.6 preregistration is not in the approved freeze state")
    if "APPROVED_TO_FREEZE" not in followup:
        raise RuntimeError("Fable follow-up review did not approve the design freeze")

    missing = [relative for relative in DESIGN_SURFACE if not resolve_root_path(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v0.6 design surface is incomplete: {missing}")
    _assert_pre_source_boundary(config)

    focused_tests = _run_gate(
        [
            str(REPO / ".venv" / "bin" / "pytest"),
            "-q",
            str(ROOT / "tests" / "test_real_agent_v06_design.py"),
        ]
    )
    if not focused_tests["passed"]:
        raise RuntimeError(f"v0.6 design tests failed: {focused_tests}")

    surface_hashes = {relative: sha256(resolve_root_path(relative)) for relative in DESIGN_SURFACE}
    metadata = {
        "project": "FinAuth-Audit",
        "version": "0.6.0",
        "status": "FROZEN_BEFORE_SOURCE_ACQUISITION",
        "surface_hashes": surface_hashes,
        "source_data_acquired": False,
        "model_proposals_generated": False,
        "outcomes_evaluated": False,
        "prohibitions": [
            "no scientific-question, split, task, model, rule-form, endpoint, or primary inference change after this freeze",
            "no paper-test outcome access before development calibration is frozen",
            "no outcome-conditioned proposal regeneration or result suppression",
            "no community-hidden outcome evaluation or repository plaintext persistence",
        ],
    }
    archive_path, archive_hash = _archive(
        DESIGN_SURFACE,
        metadata,
        resolve_root_path(config["freeze"]["design_archive_dir"]),
    )
    manifest = {
        **metadata,
        "frozen_at": _timestamp(),
        "archive_path": str(archive_path.relative_to(ROOT)),
        "archive_sha256": archive_hash,
        "focused_test_gate": focused_tests,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "claim_boundary": (
            "Prospective design freeze before v0.6 source acquisition, actual-model "
            "generation, development calibration, or outcome evaluation."
        ),
    }
    write_json(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    print(manifest_path)
    print(archive_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the prospective v0.6 scientific design.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_agent_v06.yaml"))
    args = parser.parse_args()
    freeze(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
