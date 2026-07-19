from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import hashlib
import io
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUSTED_SPEC = Path(__file__).with_name("release_spec.json")
TRUSTED_POLICY_ID = "finauth-audit-public-v0.6"
TRUSTED_RELEASE_NAME = "finauth-audit"
TRUSTED_VERSION_RE = re.compile(r"0\.6(?:\.\d+)?\Z")

CORE_REQUIRED_MEMBERS = frozenset(
    {
        "__init__.py",
        "VERSION",
        "README.md",
        "pyproject.toml",
        "release/__init__.py",
        "release/build_release.py",
        "release/release_spec.json",
        "release/verify_release.py",
    }
)

DATA_SUFFIXES = (
    ".csv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".arrow",
    ".feather",
    ".pkl",
    ".pickle",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".yaml",
    ".yml",
)
KEY_SUFFIXES = (".key", ".pem", ".p12", ".pfx", ".jks", ".keystore")
CIPHERTEXT_SUFFIXES = (".enc", ".cipher", ".ciphertext", ".gpg")

SECRET_MARKERS = (
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----",
    b"POLYMARKET_PRIVATE_KEY" + b"=0x",
    b"DATABENTO_API_KEY" + b"=",
)
SECRET_ASSIGNMENT_RE = re.compile(
    rb"(?i)(?:api[_-]?key|private[_-]?key|secret[_-]?key|access[_-]?token|password)"
    rb"\s*[:=]\s*['\"]?(?!example|placeholder|redacted|none|null)[A-Za-z0-9+/=_-]{12,}"
)

FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "api_key",
        "authorization_decision",
        "ciphertext",
        "credential",
        "credentials",
        "decision",
        "decision_reason",
        "decision_row",
        "decision_rows",
        "decision_timestamp",
        "decisions",
        "decrypted_payload",
        "evidence",
        "evidence_content",
        "evidence_excerpt",
        "evidence_snippet",
        "evidence_text",
        "full_utility",
        "ground_truth",
        "harm_label",
        "hidden_key_path",
        "outcome",
        "outcome_timestamp",
        "outcomes",
        "plaintext",
        "predicted_decision",
        "private_key",
        "private_key_path",
        "provider_response",
        "provider_responses",
        "rationale",
        "rationale_text",
        "rationales",
        "raw_log",
        "raw_logs",
        "raw_path",
        "raw_paths",
        "raw_payload",
        "raw_response",
        "raw_responses",
        "realized_outcome",
        "realized_pnl",
        "realized_return",
        "reduced_utility",
        "response_body",
        "risk_flag",
        "risk_flags",
        "secret",
        "secret_key",
        "selected_action",
        "tail_loss",
        "target_label",
    }
)

HIDDEN_SPLIT_VALUES = frozenset(
    {
        "community-hidden",
        "community_hidden",
        "hidden",
        "sealed",
    }
)

SUPERSEDED_PILOT_PREFIX = "results/real_agent_v05/"
CURRENT_STATUS_MEMBER = "ARTIFACT_CARD.md"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_without_duplicates(payload: bytes, *, source: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [str(key) for key, _ in pairs]
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        if duplicates:
            raise RuntimeError(f"duplicate JSON keys in {source}: {duplicates}")
        return {str(key): value for key, value in pairs}

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"non-UTF-8 JSON in {source}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {source}: {exc}") from exc


def _require_string_list(spec: Mapping[str, object], key: str) -> list[str]:
    value = spec.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RuntimeError(
            f"trusted release spec field {key!r} must be a list of strings"
        )
    duplicates = sorted(item for item, count in Counter(value).items() if count > 1)
    if duplicates:
        raise RuntimeError(
            f"trusted release spec field {key!r} has duplicates: {duplicates}"
        )
    return list(value)


def _restricted_namespaces(spec: Mapping[str, object]) -> dict[str, frozenset[str]]:
    raw = spec.get("restricted_namespaces", {})
    if not isinstance(raw, Mapping):
        raise RuntimeError("trusted release spec restricted_namespaces must be an object")
    namespaces: dict[str, frozenset[str]] = {}
    for prefix, members in raw.items():
        if not isinstance(prefix, str) or not prefix:
            raise RuntimeError("restricted namespace prefixes must be non-empty strings")
        if prefix.startswith(("/", "\\")) or ".." in PurePosixPath(prefix).parts:
            raise RuntimeError(f"unsafe restricted namespace prefix: {prefix!r}")
        if not isinstance(members, list) or any(
            not isinstance(member, str) or not member for member in members
        ):
            raise RuntimeError(
                f"restricted namespace {prefix!r} must contain a string list"
            )
        allowed = frozenset(_safe_relative_name(member) for member in members)
        if any(not member.startswith(prefix) for member in allowed):
            raise RuntimeError(
                f"restricted namespace {prefix!r} contains a path outside its prefix"
            )
        namespaces[prefix] = allowed
    return namespaces


def _pinned_member_sha256(spec: Mapping[str, object]) -> dict[str, str]:
    raw = spec.get("pinned_member_sha256", {})
    if not isinstance(raw, Mapping):
        raise RuntimeError("trusted release spec pinned_member_sha256 must be an object")
    pins: dict[str, str] = {}
    for relative, digest in raw.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("pinned member hashes must map string paths to strings")
        relative = _safe_relative_name(relative)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"invalid pinned SHA-256 for {relative!r}")
        pins[relative] = digest
    return pins


def validate_release_spec(spec: Mapping[str, object]) -> None:
    if spec.get("policy_id") != TRUSTED_POLICY_ID:
        raise RuntimeError(f"release spec must use policy_id={TRUSTED_POLICY_ID!r}")
    if spec.get("policy_version") != "0.6":
        raise RuntimeError("release spec must use policy_version='0.6'")
    if spec.get("release_name") != TRUSTED_RELEASE_NAME:
        raise RuntimeError(
            f"release spec must use release_name={TRUSTED_RELEASE_NAME!r}"
        )

    include_globs = _require_string_list(spec, "include_globs")
    _require_string_list(spec, "exclude_globs")
    _require_string_list(spec, "forbidden_member_substrings")
    required_members = set(_require_string_list(spec, "required_members"))
    optional_members = set(_require_string_list(spec, "optional_members"))
    _require_string_list(spec, "optional_member_globs")
    restricted = _restricted_namespaces(spec)
    pinned = _pinned_member_sha256(spec)

    missing_core = sorted(CORE_REQUIRED_MEMBERS - required_members)
    if missing_core:
        raise RuntimeError(
            f"trusted release spec omits core required members: {missing_core}"
        )
    overlap = sorted(required_members & optional_members)
    if overlap:
        raise RuntimeError(
            f"release members cannot be both required and optional: {overlap}"
        )

    for pattern in include_globs:
        if pattern.startswith(("/", "\\")) or ".." in PurePosixPath(pattern).parts:
            raise RuntimeError(
                f"unsafe include glob in trusted release spec: {pattern}"
            )
    for relative in sorted(required_members | optional_members):
        if not any(fnmatch.fnmatch(relative, pattern) for pattern in include_globs):
            raise RuntimeError(
                f"release member {relative!r} is not covered by include_globs"
            )
    for prefix, allowed in restricted.items():
        undeclared = sorted(allowed - required_members - optional_members)
        if undeclared:
            raise RuntimeError(
                f"restricted namespace {prefix!r} has undeclared members: {undeclared}"
            )
    unrequired_pins = sorted(set(pinned) - required_members)
    if unrequired_pins:
        raise RuntimeError(
            f"pinned release members must be required: {unrequired_pins}"
        )


def load_trusted_spec(
    path: Path = DEFAULT_TRUSTED_SPEC,
) -> tuple[dict[str, object], bytes]:
    payload = path.resolve().read_bytes()
    parsed = _json_without_duplicates(payload, source=str(path))
    if not isinstance(parsed, dict):
        raise RuntimeError("trusted release spec must be a JSON object")
    validate_release_spec(parsed)
    return parsed, payload


def _matches_any(relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def _safe_relative_name(relative: str) -> str:
    if not relative or "\\" in relative or "\x00" in relative:
        raise RuntimeError(f"unsafe release member path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe release member path: {relative!r}")
    return path.as_posix()


def forbidden_member_reason(relative: str) -> str | None:
    relative = _safe_relative_name(relative)
    lowered = relative.casefold()
    parts = lowered.split("/")
    basename = parts[-1]
    data_like = lowered.endswith(DATA_SUFFIXES)

    if relative != ".dockerignore" and any(part.startswith(".") for part in parts):
        return "hidden file or directory"
    if any(
        part in {"sealed", "raw", "logs", "credentials", "secrets"} for part in parts
    ):
        return "sealed, raw, log, or credential directory"
    if any(part in {"community_hidden", "community-hidden"} for part in parts):
        return "hidden community split"
    if any("ciphertext" in part or "plaintext" in part for part in parts):
        return "hidden ciphertext or plaintext"
    if lowered.endswith(KEY_SUFFIXES) or lowered.endswith(CIPHERTEXT_SUFFIXES):
        return "key or ciphertext file"
    if re.search(
        r"(?:^|[_-])(?:api[_-]?key|private[_-]?key|secret|credential)(?:[_-]|\.|$)",
        basename,
    ):
        return "key or credential file"
    if lowered.endswith(".log") or basename in {"stdout", "stderr"}:
        return "raw log file"
    if re.search(
        r"(?:^|[_-])(?:provider|api)?[_-]?(?:request|response)s?(?:[_-]|\.|$)", basename
    ):
        return "raw provider request or response"
    if data_like and re.search(r"(?:^|[_-])outcomes?(?:[_-]|\.|$)", basename):
        return "outcome-bearing file"
    if data_like and re.search(
        r"(?:^|[_-])(?:decision|decisions|decision_rows|row_decisions)(?:[_-]|\.|$)",
        basename,
    ):
        return "decision-row file"
    if data_like and any("snapshot" in part for part in parts):
        return "freeze snapshot"
    if data_like and re.search(r"(?:^|[_-])freeze(?:[_-]|\.|$)", basename):
        return "freeze snapshot"

    if parts[0] == "data" and (len(parts) < 2 or parts[1] != "public"):
        return "non-public row-level data"
    if parts[0] in {"configs", "manifests", "schemas"} and (
        len(parts) < 2 or parts[1] != "public"
    ):
        return f"non-public {parts[0]} surface"
    return None


def audit_member_path(relative: str, spec: Mapping[str, object]) -> None:
    relative = _safe_relative_name(relative)
    include_globs = _require_string_list(spec, "include_globs")
    exclude_globs = _require_string_list(spec, "exclude_globs")
    if not _matches_any(relative, include_globs):
        raise RuntimeError(
            f"archive member is outside the trusted allowlist: {relative}"
        )
    if _matches_any(relative, exclude_globs):
        raise RuntimeError(f"archive member matches a trusted exclusion: {relative}")
    for prefix, allowed in _restricted_namespaces(spec).items():
        if relative.startswith(prefix) and relative not in allowed:
            raise RuntimeError(
                f"archive member is outside restricted namespace allowlist: {relative}"
            )
    reason = forbidden_member_reason(relative)
    if reason is not None:
        raise RuntimeError(f"forbidden archive member {relative!r}: {reason}")


def _normalize_field(field: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(field).strip().casefold()).strip("_")


def forbidden_field_reason(field: object) -> str | None:
    normalized = _normalize_field(field)
    if normalized in FORBIDDEN_EXACT_FIELDS:
        return "forbidden raw, sealed, decision, outcome, rationale, evidence, or secret field"
    if "rationale" in normalized or normalized.startswith("risk_flag"):
        return "proposal rationale or risk flags"
    if normalized == "evidence" or (
        normalized.startswith("evidence_")
        and any(
            token in normalized
            for token in (
                "text",
                "content",
                "excerpt",
                "snippet",
                "quote",
                "raw",
                "path",
            )
        )
    ):
        return "evidence text or raw evidence path"
    if "raw" in normalized and any(
        token in normalized
        for token in (
            "path",
            "file",
            "uri",
            "location",
            "directory",
            "payload",
            "response",
            "request",
            "log",
        )
    ):
        return "raw provider material"
    if "provider" in normalized and any(
        token in normalized for token in ("response", "request", "payload", "log")
    ):
        return "raw provider material"
    if any(token in normalized for token in ("ciphertext", "plaintext")):
        return "hidden ciphertext or plaintext"
    if "key_path" in normalized and any(
        token in normalized for token in ("hidden", "private", "secret", "api")
    ):
        return "absolute or hidden key path"
    if normalized.startswith("freeze_snapshot") or normalized.endswith("snapshot_path"):
        return "freeze snapshot path"
    return None


def sensitive_value_reason(value: str) -> str | None:
    lowered = value.strip().casefold().replace("\\", "/")
    if lowered in HIDDEN_SPLIT_VALUES:
        return "hidden or sealed split value"
    path_like = "/" in lowered or re.match(r"^[a-z]:/", lowered) is not None
    if path_like:
        padded = f"/{lowered.strip('/')}"
        absolute = lowered.startswith("/") or re.match(r"^[a-z]:/", lowered) is not None
        if any(token in lowered for token in ("community_hidden", "community-hidden")):
            return "hidden community split path"
        if any(
            token in padded for token in ("/sealed/", "/raw/", "/logs/", "/snapshots/")
        ):
            return "raw, sealed, log, or snapshot path"
        if any(
            token in lowered
            for token in ("ciphertext", "plaintext", "private_key", "hidden_key")
        ):
            return "hidden plaintext, ciphertext, or key path"
        if absolute and (
            lowered.endswith(KEY_SUFFIXES)
            or ("hidden" in lowered and re.search(r"(?:^|/)key(?:[._/-]|$)", lowered))
        ):
            return "absolute hidden or private key path"
    return None


def _audit_json_value(value: Any, *, source: str, location: str = "$") -> None:
    if isinstance(value, Mapping):
        normalized_keys = {_normalize_field(key) for key in value}
        if {"choices", "model"}.issubset(normalized_keys) and (
            "usage" in normalized_keys or "object" in normalized_keys
        ):
            raise RuntimeError(
                f"raw provider response envelope in {source} at {location}"
            )
        if {"content", "model", "stop_reason", "usage"}.issubset(normalized_keys):
            raise RuntimeError(
                f"raw provider response envelope in {source} at {location}"
            )
        for key, nested in value.items():
            reason = forbidden_field_reason(key)
            if reason is not None:
                raise RuntimeError(
                    f"forbidden field {key!r} in {source} at {location}: {reason}"
                )
            _audit_json_value(nested, source=source, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _audit_json_value(nested, source=source, location=f"{location}[{index}]")
        return
    if isinstance(value, str):
        reason = sensitive_value_reason(value)
        if reason is not None:
            raise RuntimeError(f"forbidden value in {source} at {location}: {reason}")


def _audit_csv_payload(relative: str, payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"non-UTF-8 CSV in {relative}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        raise RuntimeError(f"CSV has no header: {relative}")
    normalized = [_normalize_field(field) for field in reader.fieldnames]
    duplicates = sorted(
        field for field, count in Counter(normalized).items() if count > 1
    )
    if duplicates:
        raise RuntimeError(f"duplicate CSV fields in {relative}: {duplicates}")
    for field in reader.fieldnames:
        reason = forbidden_field_reason(field)
        if reason is not None:
            raise RuntimeError(f"forbidden CSV field {field!r} in {relative}: {reason}")
    for row_number, row in enumerate(reader, start=2):
        for field, value in row.items():
            if value is None:
                continue
            reason = sensitive_value_reason(value)
            if reason is not None:
                raise RuntimeError(
                    f"forbidden CSV value in {relative} at row {row_number}, field {field!r}: {reason}"
                )


YAML_KEY_RE = re.compile(r"^\s*(?:-\s*)?([A-Za-z0-9_.-]+)\s*:\s*(.*)$")


def _audit_yaml_payload(relative: str, payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"non-UTF-8 YAML in {relative}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = YAML_KEY_RE.match(line)
        if match is None:
            continue
        key, scalar = match.groups()
        reason = forbidden_field_reason(key)
        if reason is not None:
            raise RuntimeError(
                f"forbidden YAML field {key!r} in {relative} at line {line_number}: {reason}"
            )
        scalar = scalar.strip().strip("'\"")
        if scalar:
            reason = sensitive_value_reason(scalar)
            if reason is not None:
                raise RuntimeError(
                    f"forbidden YAML value in {relative} at line {line_number}: {reason}"
                )


def audit_member_payload(relative: str, payload: bytes) -> None:
    for marker in SECRET_MARKERS:
        if marker in payload:
            raise RuntimeError(f"release member contains a secret marker: {relative}")
    if SECRET_ASSIGNMENT_RE.search(payload):
        raise RuntimeError(
            f"release member contains a likely embedded credential: {relative}"
        )
    if relative == "release/release_spec.json":
        return

    suffix = Path(relative).suffix.casefold()
    if suffix == ".json":
        parsed = _json_without_duplicates(payload, source=relative)
        _audit_json_value(parsed, source=relative)
    elif suffix in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            parsed = _json_without_duplicates(line, source=f"{relative}:{line_number}")
            _audit_json_value(parsed, source=relative, location=f"$[{line_number}]")
    elif suffix == ".csv":
        _audit_csv_payload(relative, payload)
    elif suffix in {".yaml", ".yml"}:
        _audit_yaml_payload(relative, payload)


def audit_claim_consistency(
    payloads: Mapping[str, bytes], *, version: str
) -> None:
    stale_pilot_members = sorted(
        relative
        for relative in payloads
        if relative.startswith(SUPERSEDED_PILOT_PREFIX)
    )
    if stale_pilot_members:
        raise RuntimeError(
            "v0.6 release contains superseded v0.5 pilot result members: "
            f"{stale_pilot_members}"
        )

    status_payload = payloads.get(CURRENT_STATUS_MEMBER)
    if status_payload is None:
        raise RuntimeError(f"release is missing {CURRENT_STATUS_MEMBER}")
    try:
        status = status_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{CURRENT_STATUS_MEMBER} is not UTF-8") from exc

    required_statements = (
        f"Release version: **{version}**",
        "Spearman rho: **+0.657**",
        "Exact one-sided inverse-or-more-extreme permutation p-value: **0.932**",
        "is **not supported**",
        "superseded for the inverse-transfer claim",
    )
    missing = [
        statement for statement in required_statements if statement not in status
    ]
    if missing:
        raise RuntimeError(
            f"{CURRENT_STATUS_MEMBER} is stale or claim-incomplete; missing={missing}"
        )
    if "-0.928" in status or "Release version: **0.5" in status:
        raise RuntimeError(
            f"{CURRENT_STATUS_MEMBER} promotes a superseded v0.5 pilot result"
        )


def _safe_relative(member_name: str, prefix: str) -> str:
    if not prefix or prefix in {".", ".."} or "/" in prefix or "\\" in prefix:
        raise RuntimeError(f"unsafe archive root: {prefix!r}")
    expected = f"{prefix}/"
    if not member_name.startswith(expected):
        raise RuntimeError(f"archive member is outside release root: {member_name}")
    return _safe_relative_name(member_name[len(expected) :])


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    prefix: str,
    trusted_spec_sha256: str,
    trusted_spec: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    if manifest.get("policy_id") != TRUSTED_POLICY_ID:
        raise RuntimeError("release manifest does not identify the trusted v0.6 policy")
    if manifest.get("release_name") != TRUSTED_RELEASE_NAME:
        raise RuntimeError("release manifest has an unexpected release name")
    version = manifest.get("version")
    if not isinstance(version, str) or TRUSTED_VERSION_RE.fullmatch(version) is None:
        raise RuntimeError(
            f"release manifest has an unsupported v0.6 version: {version!r}"
        )
    if prefix != f"{TRUSTED_RELEASE_NAME}-{version}":
        raise RuntimeError(
            f"archive root {prefix!r} does not match release version {version!r}"
        )
    if manifest.get("archive_root") != prefix:
        raise RuntimeError("release manifest archive_root does not match the tar root")
    if manifest.get("spec_sha256") != trusted_spec_sha256:
        raise RuntimeError(
            "release manifest was built with an untrusted or weakened release spec"
        )

    entries = manifest.get("members")
    if not isinstance(entries, list):
        raise RuntimeError("release manifest members must be a list")
    expected: dict[str, dict[str, object]] = {}
    pinned = _pinned_member_sha256(trusted_spec)
    normalized_paths: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"release manifest member {index} must be an object")
        relative = entry.get("path")
        byte_count = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise RuntimeError(f"invalid release manifest member at index {index}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"invalid SHA-256 in release manifest member {relative!r}"
            )
        relative = _safe_relative_name(relative)
        if relative == "RELEASE_MANIFEST.json":
            raise RuntimeError(
                "release manifest cannot list itself as a payload member"
            )
        if relative in expected or relative.casefold() in normalized_paths:
            raise RuntimeError(f"duplicate release manifest member path: {relative}")
        audit_member_path(relative, trusted_spec)
        if relative in pinned and digest != pinned[relative]:
            raise RuntimeError(f"pinned release member hash mismatch: {relative}")
        expected[relative] = {"bytes": byte_count, "sha256": digest}
        normalized_paths.add(relative.casefold())

    if manifest.get("member_count") != len(expected):
        raise RuntimeError("release manifest member_count is inconsistent")
    total_member_bytes = sum(int(entry["bytes"]) for entry in expected.values())
    if manifest.get("total_member_bytes") != total_member_bytes:
        raise RuntimeError("release manifest total_member_bytes is inconsistent")
    return expected


def _canonical_tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def _canonical_archive_bytes(
    prefix: str, manifest_bytes: bytes, payloads: Mapping[str, bytes]
) -> bytes:
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w") as tar:
        manifest_name = f"{prefix}/RELEASE_MANIFEST.json"
        tar.addfile(
            _canonical_tar_info(manifest_name, len(manifest_bytes)),
            io.BytesIO(manifest_bytes),
        )
        for relative in sorted(payloads):
            payload = payloads[relative]
            tar.addfile(
                _canonical_tar_info(f"{prefix}/{relative}", len(payload)),
                io.BytesIO(payload),
            )
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as handle:
        handle.write(raw_tar.getvalue())
    return compressed.getvalue()


def _find_install_python() -> str:
    candidates = [sys.executable, shutil.which("python3"), shutil.which("python")]
    seen: set[str] = set()
    probe = (
        "import re,sys,setuptools,pip,setuptools.command.bdist_wheel; "
        "m=re.match(r'(\\d+)\\.(\\d+)', setuptools.__version__); "
        "assert sys.version_info >= (3,10) and m and tuple(map(int,m.groups())) >= (68,0); "
        "print(sys.executable)"
    )
    for candidate in candidates:
        if not candidate:
            continue
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        completed = subprocess.run(
            [resolved, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.returncode == 0:
            return resolved
    raise RuntimeError(
        "no local Python >=3.10 with pip and setuptools>=68 is available for install verification"
    )


def _verify_installability(extraction_root: Path, temp_root: Path) -> dict[str, object]:
    python = _find_install_python()
    target = temp_root / "site-packages"
    env = os.environ.copy()
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    command = [
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-build-isolation",
        "--no-compile",
        "--target",
        str(target),
        str(extraction_root),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    if completed.returncode != 0:
        output = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"extracted release is not installable:\n{output[-4000:]}")

    package_root = target / "finauth_audit"
    required_installed = [
        package_root / "__init__.py",
        package_root / "release" / "__init__.py",
        package_root / "release" / "release_spec.json",
        package_root / "release" / "verify_release.py",
    ]
    missing = [
        str(path.relative_to(target))
        for path in required_installed
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"installed release is missing package files: {missing}")
    unexpected = [
        name for name in ("data", "results", "sealed") if (target / name).exists()
    ]
    if unexpected:
        raise RuntimeError(
            f"package discovery installed forbidden top-level packages: {unexpected}"
        )

    import_env = env.copy()
    import_env["PYTHONPATH"] = str(target)
    imported = subprocess.run(
        [
            python,
            "-c",
            "import finauth_audit; import finauth_audit.release.verify_release",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=temp_root,
        env=import_env,
        timeout=30,
    )
    if imported.returncode != 0:
        output = (imported.stdout + "\n" + imported.stderr).strip()
        raise RuntimeError(f"installed release cannot be imported:\n{output[-4000:]}")
    return {"install_python": python, "installed_package": "finauth_audit"}


def verify_archive(
    archive: Path,
    *,
    clean_room: bool = True,
    trusted_spec_path: Path = DEFAULT_TRUSTED_SPEC,
    check_install: bool = True,
) -> dict[str, object]:
    archive = archive.resolve()
    trusted_spec, trusted_spec_bytes = load_trusted_spec(trusted_spec_path)
    trusted_spec_sha256 = sha256_bytes(trusted_spec_bytes)
    compiled = 0
    csv_files_checked = 0
    install_report: dict[str, object] = {}

    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError("release archive is empty")
        for member in members:
            if (
                member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mode != 0o644
            ):
                raise RuntimeError(
                    f"release archive has non-canonical tar metadata: {member.name}"
                )
        duplicate_names = sorted(
            name
            for name, count in Counter(member.name for member in members).items()
            if count > 1
        )
        if duplicate_names:
            raise RuntimeError(
                f"release archive has duplicate tar members: {duplicate_names}"
            )
        folded_names = Counter(member.name.casefold() for member in members)
        case_collisions = sorted(
            name for name, count in folded_names.items() if count > 1
        )
        if case_collisions:
            raise RuntimeError(
                f"release archive has case-colliding tar members: {case_collisions}"
            )
        roots = {member.name.split("/", 1)[0] for member in members}
        if len(roots) != 1:
            raise RuntimeError(f"release archive has multiple roots: {sorted(roots)}")
        prefix = next(iter(roots))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prefix):
            raise RuntimeError(f"unsafe archive root: {prefix!r}")

        manifest_name = f"{prefix}/RELEASE_MANIFEST.json"
        manifest_members = [
            member for member in members if member.name == manifest_name
        ]
        if len(manifest_members) != 1 or not manifest_members[0].isfile():
            raise RuntimeError(
                "release archive must contain exactly one regular RELEASE_MANIFEST.json"
            )
        manifest_handle = tar.extractfile(manifest_members[0])
        if manifest_handle is None:
            raise RuntimeError("release manifest cannot be read")
        manifest_bytes = manifest_handle.read()
        manifest = _json_without_duplicates(manifest_bytes, source=manifest_name)
        if not isinstance(manifest, Mapping):
            raise RuntimeError("release manifest must be a JSON object")
        expected = _validate_manifest(
            manifest,
            prefix=prefix,
            trusted_spec_sha256=trusted_spec_sha256,
            trusted_spec=trusted_spec,
        )

        observed: dict[str, dict[str, object]] = {}
        payloads: dict[str, bytes] = {}
        for member in members:
            if not member.isfile():
                raise RuntimeError(f"unsupported archive member type: {member.name}")
            relative = _safe_relative(member.name, prefix)
            if relative == "RELEASE_MANIFEST.json":
                continue
            audit_member_path(relative, trusted_spec)
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            payload = handle.read()
            audit_member_payload(relative, payload)
            payloads[relative] = payload
            observed[relative] = {
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }

        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise RuntimeError(
                f"archive/manifest mismatch; missing={missing} extra={extra}"
            )
        for relative, entry in expected.items():
            if observed[relative] != entry:
                raise RuntimeError(f"archive member hash mismatch: {relative}")

        packaged_spec_bytes = payloads.get("release/release_spec.json")
        if packaged_spec_bytes is None:
            raise RuntimeError("packaged release spec is missing")
        if sha256_bytes(packaged_spec_bytes) != trusted_spec_sha256:
            raise RuntimeError(
                "packaged release spec differs from the trusted external spec"
            )
        packaged_spec = _json_without_duplicates(
            packaged_spec_bytes, source="release/release_spec.json"
        )
        if packaged_spec != trusted_spec:
            raise RuntimeError(
                "packaged release spec is semantically different from the trusted external spec"
            )

        missing_required = sorted(
            set(_require_string_list(trusted_spec, "required_members")) - set(observed)
        )
        if missing_required:
            raise RuntimeError(
                f"required archive members missing under trusted policy: {missing_required}"
            )
        version_payload = (
            payloads.get("VERSION", b"").decode("utf-8", errors="strict").strip()
        )
        if version_payload != manifest.get("version"):
            raise RuntimeError("VERSION file does not match the release manifest")
        audit_claim_consistency(payloads, version=version_payload)

        expected_order = [manifest_name] + [
            f"{prefix}/{relative}" for relative in sorted(payloads)
        ]
        if [member.name for member in members] != expected_order:
            raise RuntimeError("release archive member order is not canonical")
        canonical = _canonical_archive_bytes(prefix, manifest_bytes, payloads)
        if archive.read_bytes() != canonical:
            raise RuntimeError(
                "release archive bytes are not canonical or contain trailing data"
            )

        if clean_room:
            with tempfile.TemporaryDirectory(
                prefix="finauth-release-verify-"
            ) as temp_dir:
                temp_root = Path(temp_dir)
                extraction_root = temp_root / prefix
                for relative, payload in {
                    "RELEASE_MANIFEST.json": manifest_bytes,
                    **payloads,
                }.items():
                    destination = extraction_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                for path in extraction_root.rglob("*.py"):
                    py_compile.compile(str(path), doraise=True)
                    compiled += 1
                csv_files_checked = sum(
                    1 for path in extraction_root.rglob("*.csv") if path.is_file()
                )
                if check_install:
                    install_report = _verify_installability(extraction_root, temp_root)

    return {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_path(archive),
        "clean_room": clean_room,
        "compiled_python_files": compiled,
        "csv_data_files_checked": csv_files_checked,
        "csv_files_checked": csv_files_checked,
        "install_checked": bool(clean_room and check_install),
        **install_report,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "member_count": len(expected),
        "passed": True,
        "policy_id": TRUSTED_POLICY_ID,
        "trusted_spec_sha256": trusted_spec_sha256,
        "version": manifest["version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a FinAuth-Audit v0.6 public release archive."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trusted-spec", type=Path, default=DEFAULT_TRUSTED_SPEC)
    parser.add_argument("--no-clean-room", action="store_true")
    parser.add_argument("--no-install-check", action="store_true")
    args = parser.parse_args()
    report = verify_archive(
        args.archive,
        clean_room=not args.no_clean_room,
        trusted_spec_path=args.trusted_spec,
        check_install=not args.no_install_check,
    )
    report_path = args.report or args.archive.with_suffix("").with_suffix(
        ".verification.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"FinAuth-Audit release verification: PASS ({report['member_count']} members)"
    )
    print(report_path)


if __name__ == "__main__":
    main()
