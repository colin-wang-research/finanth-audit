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

IMPORTED_V03_SURFACE = (
    "baselines/rules.py",
    "generators/external_orderbook_v03.py",
    "generators/build_databento_bbo_v03.py",
    "evaluation/external_orderbook_power.py",
    "evaluation/external_orderbook_test.py",
)

STATIC_SURFACE = (
    "configs/databento_powered_v04.yaml",
    "manifests/preregistration/databento_powered_v04.yaml",
    "manifests/databento_powered_v04_feature_access.json",
    "manifests/databento_powered_v04_preflight.json",
    "docs/databento_powered_protocol_v04.md",
    "generators/build_databento_powered_v04.py",
    "evaluation/databento_powered_power_v04.py",
    "evaluation/databento_powered_structural_audit_v04.py",
    "evaluation/databento_powered_freeze_v04.py",
    "evaluation/databento_powered_test_v04.py",
    "tests/test_databento_powered_v04.py",
    "reviews/evidence/fable_round_81_databento_v04_design_raw.md",
    "reviews/evidence/fable_round_81_databento_v04_followup_raw.md",
    "reviews/fable_round_81_databento_v04_design.md",
    "reviews/supervisor_round_81.md",
    "manifests/external_orderbook_v03_freeze.json",
    *IMPORTED_V03_SURFACE,
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
    surface_hashes: dict[str, str], metadata: dict[str, object], output_dir: Path
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finauth-databento-v04-") as temporary:
        tar_path = Path(temporary) / "snapshot.tar"
        with tarfile.open(tar_path, "w") as archive:
            _add_bytes(
                archive,
                "SNAPSHOT_CONTENTS.json",
                (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
            for relative in sorted(surface_hashes):
                _add_bytes(archive, relative, resolve_root_path(relative).read_bytes())
        compressed = Path(temporary) / "snapshot.tar.gz"
        with tar_path.open("rb") as source, compressed.open("wb") as destination:
            with gzip.GzipFile(
                filename="", fileobj=destination, mode="wb", mtime=0
            ) as handle:
                shutil.copyfileobj(source, handle)
        digest = sha256(compressed)
        final = output_dir / f"databento_powered_v04_freeze_{digest}.tar.gz"
        if not final.exists():
            shutil.copy2(compressed, final)
        if sha256(final) != digest:
            raise RuntimeError("content-addressed v0.4 freeze archive mismatch")
        os.chmod(final, 0o444)
    return final, digest


def _verify_imported_v03_surface() -> dict[str, str]:
    freeze_path = ROOT / "manifests" / "external_orderbook_v03_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_EXTERNAL_TEST":
        raise RuntimeError("v0.3 external freeze is not active")
    verified: dict[str, str] = {}
    for relative in IMPORTED_V03_SURFACE:
        expected = freeze.get("surface_hashes", {}).get(relative)
        if expected is None:
            raise RuntimeError(f"v0.3 freeze omits imported surface: {relative}")
        actual = sha256(resolve_root_path(relative))
        if actual != expected:
            raise RuntimeError(f"imported v0.3 surface changed: {relative}")
        verified[relative] = actual
    return verified


def freeze(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    freeze_path = resolve_root_path(config["freeze"]["freeze_manifest"])
    registry_path = resolve_root_path(config["freeze"]["test_registry"])
    if freeze_path.exists():
        raise FileExistsError("Databento v0.4 freeze already exists")
    if registry_path.exists():
        raise FileExistsError("Databento v0.4 test registry exists before freeze")

    prereg = load_config(resolve_root_path(config["freeze"]["preregistration"]))
    if prereg.get("status") != "PREREGISTERED_BEFORE_DOWNLOAD_AND_OUTCOME_EVALUATION":
        raise RuntimeError("v0.4 preregistration design gate is not finalized")
    if prereg.get("design_review", {}).get("fable_decision") != "APPROVE":
        raise RuntimeError("v0.4 Fable design gate is not approved")
    if prereg.get("design_review", {}).get("supervisor_internal_completion") != "10.00/10":
        raise RuntimeError("v0.4 Supervisor implementation gate is not complete")

    structural_path = resolve_root_path(config["freeze"]["structural_audit"])
    power_path = resolve_root_path(config["freeze"]["power_report"])
    if not structural_path.exists() or not power_path.exists():
        raise FileNotFoundError("v0.4 structural audit and power report are required")
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    power = json.loads(power_path.read_text(encoding="utf-8"))
    if structural.get("passed") is not True:
        raise RuntimeError("v0.4 structural audit gate is not satisfied")
    if structural.get("outcome_metrics_computed") is not False:
        raise RuntimeError("v0.4 structural audit crossed the pre-result boundary")
    if power.get("gate_passed") is not True:
        raise RuntimeError("v0.4 power gate is not satisfied")
    if power.get("external_outcomes_read") is not False:
        raise RuntimeError("v0.4 power report crossed the pre-result boundary")
    imported_v03_hashes = _verify_imported_v03_surface()

    source = config["databento"]
    dynamic = [
        str(resolve_root_path(source["source_manifest"]).relative_to(ROOT)),
        str(resolve_root_path(source["dataset_manifest"]).relative_to(ROOT)),
        str(structural_path.relative_to(ROOT)),
        str(power_path.relative_to(ROOT)),
        str(
            (resolve_root_path(source["derived_dir"]) / "split_registry.csv").relative_to(
                ROOT
            )
        ),
    ]
    surface = tuple(dict.fromkeys((*STATIC_SURFACE, *dynamic)))
    missing = [relative for relative in surface if not resolve_root_path(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v0.4 freeze surface is incomplete: {missing}")
    focused_tests = _run_gate(
        [
            str(REPO / ".venv" / "bin" / "pytest"),
            "-q",
            str(ROOT / "tests" / "test_databento_powered_v04.py"),
        ]
    )
    if not focused_tests["passed"]:
        raise RuntimeError(f"v0.4 focused tests failed: {focused_tests}")
    surface_hashes = {relative: sha256(resolve_root_path(relative)) for relative in surface}
    dataset_manifest_path = resolve_root_path(source["dataset_manifest"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("outcome_metrics_computed") is not False:
        raise RuntimeError("v0.4 dataset manifest violates pre-result boundary")
    if dataset_manifest.get("calendar_overlap_with_v03") is not False:
        raise RuntimeError("v0.4 dataset overlaps the inspected v0.3 period")

    metadata = {
        "project": "FinAuth-Audit",
        "version": "0.4.0",
        "status": "FROZEN_BEFORE_DATABENTO_POWERED_TEST",
        "surface_hashes": surface_hashes,
        "dataset_hashes": dataset_manifest["outputs"],
        "imported_v03_surface_hashes": imported_v03_hashes,
        "structural_audit_sha256": sha256(structural_path),
        "power_report_sha256": sha256(power_path),
        "prohibitions": [
            "no prior, rule, threshold, cost, metric, endpoint, SESOI, timing, or split changes after freeze",
            "no v0.4 paper-test rerun",
            "no community-hidden evaluation",
            "no result suppression",
            "no Databento row-level redistribution",
            "no conflation with the failed v0.3 Binance intersection-union result",
        ],
    }
    archive_dir = resolve_root_path(config["freeze"]["archive_dir"])
    archive_path, archive_hash = _archive(surface_hashes, metadata, archive_dir)
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
        "paper_test_outcomes_evaluated": False,
        "community_hidden_outcomes_evaluated": False,
        "claim_boundary": (
            "Content-addressed v0.4 freeze before the one-time powered MES test. "
            "It does not authorize community-hidden evaluation."
        ),
    }
    write_json(freeze_path, manifest)
    os.chmod(freeze_path, 0o444)
    print(freeze_path)
    print(archive_path)
    return freeze_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the Databento powered v0.4 test surface."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "databento_powered_v04.yaml")
    )
    args = parser.parse_args()
    freeze(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
