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
    "configs/external_orderbook_v03.yaml",
    "manifests/preregistration/external_orderbook_v03.yaml",
    "manifests/external_orderbook_v03_feature_access.json",
    "docs/external_orderbook_protocol_v03.md",
    "baselines/rules.py",
    "generators/external_orderbook_v03.py",
    "generators/fetch_binance_depth_v03.py",
    "generators/build_binance_depth_v03.py",
    "generators/build_databento_bbo_v03.py",
    "evaluation/external_orderbook_structural_audit.py",
    "evaluation/external_orderbook_power.py",
    "evaluation/external_orderbook_test.py",
    "evaluation/external_orderbook_freeze.py",
    "tests/test_external_orderbook_v03.py",
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


def _archive(surface_hashes: dict[str, str], metadata: dict[str, object], output_dir: Path) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finauth-external-v03-") as temporary:
        tar_path = Path(temporary) / "snapshot.tar"
        with tarfile.open(tar_path, "w") as archive:
            _add_bytes(
                archive,
                "SNAPSHOT_CONTENTS.json",
                (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            for relative in sorted(surface_hashes):
                _add_bytes(archive, relative, resolve_root_path(relative).read_bytes())
        compressed = Path(temporary) / "snapshot.tar.gz"
        with tar_path.open("rb") as source, compressed.open("wb") as destination:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as handle:
                shutil.copyfileobj(source, handle)
        digest = sha256(compressed)
        final = output_dir / f"external_orderbook_v03_freeze_{digest}.tar.gz"
        if not final.exists():
            shutil.copy2(compressed, final)
        if sha256(final) != digest:
            raise RuntimeError("content-addressed external freeze archive mismatch")
        os.chmod(final, 0o444)
    return final, digest


def freeze(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    freeze_path = resolve_root_path(config["freeze"]["freeze_manifest"])
    registry_path = resolve_root_path(config["freeze"]["test_registry"])
    if freeze_path.exists():
        raise FileExistsError("external order-book freeze already exists")
    if registry_path.exists():
        raise FileExistsError("external test registry exists before freeze")
    structural_path = resolve_root_path(config["results_dir"]) / "structural_audit.json"
    power_path = resolve_root_path(config["freeze"]["power_report"])
    if not structural_path.exists() or not power_path.exists():
        raise FileNotFoundError("structural audit and pre-result power report are required")
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    power = json.loads(power_path.read_text(encoding="utf-8"))
    if structural.get("passed") is not True or structural.get("outcome_metrics_computed") is not False:
        raise RuntimeError("structural audit gate is not satisfied")
    if power.get("gate_passed") is not True or power.get("external_outcomes_read") is not False:
        raise RuntimeError("pre-result power gate is not satisfied")
    dynamic = [
        str(resolve_root_path(config["binance"]["source_manifest"]).relative_to(ROOT)),
        str(resolve_root_path(config["binance"]["dataset_manifest"]).relative_to(ROOT)),
        str(resolve_root_path(config["databento"]["source_manifest"]).relative_to(ROOT)),
        str(resolve_root_path(config["databento"]["dataset_manifest"]).relative_to(ROOT)),
        str(structural_path.relative_to(ROOT)),
        str(power_path.relative_to(ROOT)),
        str((resolve_root_path(config["binance"]["derived_dir"]) / "split_registry.csv").relative_to(ROOT)),
        str((resolve_root_path(config["databento"]["derived_dir"]) / "split_registry.csv").relative_to(ROOT)),
    ]
    surface = tuple(dict.fromkeys((*STATIC_SURFACE, *dynamic)))
    missing = [relative for relative in surface if not resolve_root_path(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"external freeze surface is incomplete: {missing}")
    focused_tests = _run_gate(
        [
            str(REPO / ".venv" / "bin" / "pytest"),
            "-q",
            str(ROOT / "tests" / "test_external_orderbook_v03.py"),
        ]
    )
    if not focused_tests["passed"]:
        raise RuntimeError(f"external focused tests failed: {focused_tests}")
    surface_hashes = {relative: sha256(resolve_root_path(relative)) for relative in surface}
    dataset_hashes: dict[str, dict[str, str]] = {}
    for source_name in ("binance", "databento"):
        manifest_path = resolve_root_path(config[source_name]["dataset_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("outcome_metrics_computed") is not False:
            raise RuntimeError(f"{source_name} dataset manifest violates pre-result boundary")
        dataset_hashes[source_name] = manifest["outputs"]
    metadata = {
        "project": "FinAuth-Audit",
        "version": "0.3.0",
        "status": "FROZEN_BEFORE_EXTERNAL_TEST",
        "surface_hashes": surface_hashes,
        "dataset_hashes": dataset_hashes,
        "structural_audit_sha256": sha256(structural_path),
        "power_report_sha256": sha256(power_path),
        "prohibitions": [
            "no prior, rule, threshold, cost, metric, endpoint, or split changes after freeze",
            "no external test rerun",
            "no community-hidden evaluation",
            "no result suppression",
            "no Databento row-level redistribution",
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
            "Content-addressed v0.3 freeze before the one-time external paper test. "
            "It does not authorize community-hidden evaluation."
        ),
    }
    write_json(freeze_path, manifest)
    os.chmod(freeze_path, 0o444)
    print(freeze_path)
    print(archive_path)
    return freeze_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the v0.3 external order-book test surface.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "external_orderbook_v03.yaml"))
    args = parser.parse_args()
    freeze(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
