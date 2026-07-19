from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import io
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from .verify_release import (
        TRUSTED_POLICY_ID,
        TRUSTED_VERSION_RE,
        audit_member_path,
        audit_member_payload,
        forbidden_field_reason,
        forbidden_member_reason,
        load_trusted_spec,
        _pinned_member_sha256,
        sensitive_value_reason,
        sha256_bytes,
        sha256_path,
        validate_release_spec,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from verify_release import (  # type: ignore[no-redef]
        TRUSTED_POLICY_ID,
        TRUSTED_VERSION_RE,
        audit_member_path,
        audit_member_payload,
        forbidden_field_reason,
        forbidden_member_reason,
        load_trusted_spec,
        _pinned_member_sha256,
        sensitive_value_reason,
        sha256_bytes,
        sha256_path,
        validate_release_spec,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path(__file__).with_name("release_spec.json")
_DROP = object()


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, object]:
    spec, _ = load_trusted_spec(path)
    return spec


def matches_any(relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def forbidden_member(relative: str, spec: Mapping[str, object]) -> str | None:
    reason = forbidden_member_reason(relative)
    if reason is not None:
        return reason
    normalized = f"/{relative.casefold().lstrip('/')}"
    for token in spec["forbidden_member_substrings"]:
        if str(token).casefold() in normalized:
            return str(token)
    return None


def collect_files(root: Path, spec: dict[str, object]) -> list[Path]:
    validate_release_spec(spec)
    selected: dict[str, Path] = {}
    include_globs = [str(value) for value in spec["include_globs"]]
    exclude_globs = [str(value) for value in spec["exclude_globs"]]
    for pattern in include_globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise RuntimeError(f"release candidate cannot contain symlinks: {path}")
            relative = path.relative_to(root).as_posix()
            if matches_any(relative, exclude_globs):
                continue
            audit_member_path(relative, spec)
            token = forbidden_member(relative, spec)
            if token is not None:
                raise RuntimeError(
                    f"forbidden release member {relative!r} matched {token!r}"
                )
            audit_member_payload(relative, path.read_bytes())
            selected[relative] = path
    missing = sorted(set(map(str, spec["required_members"])) - set(selected))
    if missing:
        raise RuntimeError(f"required release members are missing: {missing}")
    for relative, expected in _pinned_member_sha256(spec).items():
        path = selected.get(relative)
        if path is None:
            raise RuntimeError(f"pinned release member is missing: {relative}")
        if sha256_path(path) != expected:
            raise RuntimeError(f"pinned release member hash mismatch: {relative}")
    return [selected[key] for key in sorted(selected)]


def _sanitize_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            key_text = str(key)
            if forbidden_field_reason(key_text) is not None:
                continue
            cleaned = _sanitize_public_value(nested)
            if cleaned is not _DROP:
                sanitized[key_text] = cleaned
        return sanitized
    if isinstance(value, (list, tuple)):
        sanitized_items = []
        for nested in value:
            cleaned = _sanitize_public_value(nested)
            if cleaned is not _DROP:
                sanitized_items.append(cleaned)
        return sanitized_items
    if isinstance(value, str) and sensitive_value_reason(value) is not None:
        return _DROP
    return value


def sanitize_public_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Return a proposal projection without private reasoning or sealed execution fields."""

    sanitized = _sanitize_public_value(proposal)
    if not isinstance(sanitized, dict):
        raise TypeError("proposal projection must produce a JSON object")
    return sanitized


def sanitize_public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a public config projection without raw paths, keys, or hidden payload fields."""

    sanitized = _sanitize_public_value(config)
    if not isinstance(sanitized, dict):
        raise TypeError("config projection must produce a JSON object")
    return sanitized


def project_public_proposals(
    proposals: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [sanitize_public_proposal(proposal) for proposal in proposals]


def project_public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return sanitize_public_config(config)


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def write_public_proposal_projection(
    proposals: Iterable[Mapping[str, Any]],
    destination: Path,
) -> Path:
    sanitized = project_public_proposals(proposals)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.casefold()
    if suffix == ".csv":
        fieldnames = sorted({field for proposal in sanitized for field in proposal})
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for proposal in sanitized:
                writer.writerow(
                    {field: _csv_scalar(proposal.get(field)) for field in fieldnames}
                )
    elif suffix == ".jsonl":
        payload = "".join(
            json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n"
            for item in sanitized
        )
        destination.write_text(payload, encoding="utf-8")
    elif suffix == ".json":
        destination.write_text(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        raise ValueError("public proposal projections must use .csv, .json, or .jsonl")
    audit_member_payload(f"data/public/{destination.name}", destination.read_bytes())
    return destination


def write_public_config_projection(
    config: Mapping[str, Any], destination: Path
) -> Path:
    if destination.suffix.casefold() != ".json":
        raise ValueError("public config projections must use JSON")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(project_public_config(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_member_payload(f"configs/public/{destination.name}", destination.read_bytes())
    return destination


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def build_release(
    *,
    root: Path = ROOT,
    spec_path: Path = DEFAULT_SPEC,
    trusted_spec_path: Path = DEFAULT_SPEC,
    output_dir: Path,
) -> tuple[Path, Path]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    trusted_spec_path = trusted_spec_path.resolve()
    spec, spec_bytes = load_trusted_spec(spec_path)
    trusted_spec, trusted_spec_bytes = load_trusted_spec(trusted_spec_path)
    if spec != trusted_spec or sha256_bytes(spec_bytes) != sha256_bytes(
        trusted_spec_bytes
    ):
        raise RuntimeError(
            "release build spec differs from the trusted external v0.6 spec"
        )
    packaged_spec_path = root / "release" / "release_spec.json"
    if packaged_spec_path.read_bytes() != spec_bytes:
        raise RuntimeError(
            "root release/release_spec.json differs from the trusted build spec"
        )

    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if TRUSTED_VERSION_RE.fullmatch(version) is None:
        raise RuntimeError(
            f"release VERSION is outside the trusted v0.6 line: {version!r}"
        )
    release_name = str(spec["release_name"])
    prefix = f"{release_name}-{version}"
    files = collect_files(root, spec)
    entries = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    manifest = {
        "archive_root": prefix,
        "member_count": len(entries),
        "members": entries,
        "forbidden_member_substrings": spec["forbidden_member_substrings"],
        "policy_id": TRUSTED_POLICY_ID,
        "policy_version": spec["policy_version"],
        "required_members": spec["required_members"],
        "release_name": release_name,
        "spec_sha256": sha256_bytes(spec_bytes),
        "total_member_bytes": sum(int(entry["bytes"]) for entry in entries),
        "version": version,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{prefix}.tar.gz"
    with tempfile.NamedTemporaryFile(
        prefix="finauth-release-", suffix=".tar"
    ) as raw_tar:
        with tarfile.open(fileobj=raw_tar, mode="w") as tar:
            manifest_name = f"{prefix}/RELEASE_MANIFEST.json"
            tar.addfile(
                _tar_info(manifest_name, len(manifest_bytes)),
                io.BytesIO(manifest_bytes),
            )
            for entry, path in zip(entries, files, strict=True):
                arcname = f"{prefix}/{entry['path']}"
                with path.open("rb") as handle:
                    tar.addfile(_tar_info(arcname, int(entry["bytes"])), handle)
        raw_tar.flush()
        raw_tar.seek(0)
        with archive.open("wb") as output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=output, mtime=0
            ) as compressed:
                shutil.copyfileobj(raw_tar, compressed, length=1024 * 1024)
    report = {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_path(archive),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "member_count": len(entries),
        "policy_id": TRUSTED_POLICY_ID,
        "release_name": release_name,
        "spec_sha256": sha256_bytes(spec_bytes),
        "version": version,
    }
    report_path = output_dir / f"{prefix}.build.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archive, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic FinAuth-Audit v0.6 public release."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--trusted-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    archive, report = build_release(
        spec_path=args.spec,
        trusted_spec_path=args.trusted_spec,
        output_dir=args.output_dir,
    )
    print(archive)
    print(report)


if __name__ == "__main__":
    main()
