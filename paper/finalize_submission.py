#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
ROOT = PAPER_DIR.parent
REPO = ROOT.parent
CANONICAL_METADATA = PAPER_DIR / "author_metadata.tex"
CANONICAL_PDF = PAPER_DIR / "FinAuth-Audit.pdf"
STRICT_REPORT = ROOT / "results" / "verification" / "paper_verification_strict.json"


def default_output_path() -> Path:
    workspace_output = REPO / "paper" / "kdd" / "FinAuth-Audit-KDD.pdf"
    if ROOT.name == "finauth_audit" and workspace_output.parent.is_dir():
        return workspace_output
    return ROOT / "dist" / "FinAuth-Audit-KDD.pdf"


def submission_paper_target() -> str:
    registry = ROOT / "manifests" / "real_agent_v06_test_registry.json"
    return "submission-paper-full" if registry.is_file() else "submission-paper"


DEFAULT_OUTPUT = default_output_path()

PLACEHOLDER_TOKENS = (
    "Author Metadata Required",
    "Replace with the submitting institution",
    "Replace before submission",
    "replace-before-submission@example.invalid",
    "Anonymous Author(s)",
    "Anonymous Institution",
    "anonymous@example.com",
)
FORBIDDEN_METADATA_COMMANDS = (
    r"\documentclass",
    r"\usepackage",
    r"\input",
    r"\include",
    r"\write18",
    r"\immediate",
    r"\openout",
    r"\read",
)
EMAIL_RE = re.compile(r"^[^\s@{}]+@[^\s@{}]+\.[^\s@{}]+$")


class MetadataValidationError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_braced_arguments(text: str, command: str) -> list[str]:
    marker = f"\\{command}"
    values: list[str] = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return values
        position = start + len(marker)
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] != "{":
            cursor = position
            continue
        depth = 1
        value_start = position + 1
        position += 1
        while position < len(text) and depth:
            if text[position] == "{" and text[position - 1] != "\\":
                depth += 1
            elif text[position] == "}" and text[position - 1] != "\\":
                depth -= 1
            position += 1
        if depth:
            raise MetadataValidationError(f"unbalanced braces in \\{command}")
        values.append(text[value_start : position - 1].strip())
        cursor = position


def _require_nonempty(values: list[str], label: str) -> None:
    if not values or any(not value.strip() for value in values):
        raise MetadataValidationError(f"author metadata requires nonempty {label}")


def validate_author_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise MetadataValidationError(f"author metadata file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MetadataValidationError("author metadata must be UTF-8") from exc

    present_placeholders = [token for token in PLACEHOLDER_TOKENS if token in text]
    if present_placeholders:
        raise MetadataValidationError(
            f"replace submission placeholders before finalization: {present_placeholders}"
        )
    present_forbidden = [
        command for command in FORBIDDEN_METADATA_COMMANDS if command in text
    ]
    if present_forbidden:
        raise MetadataValidationError(
            f"author metadata contains forbidden TeX commands: {present_forbidden}"
        )

    authors = extract_braced_arguments(text, "author")
    institutions = extract_braced_arguments(text, "institution")
    cities = extract_braced_arguments(text, "city")
    countries = extract_braced_arguments(text, "country")
    emails = extract_braced_arguments(text, "email")
    _require_nonempty(authors, "\\author entries")
    _require_nonempty(institutions, "\\institution entries")
    _require_nonempty(cities, "\\city entries")
    _require_nonempty(countries, "\\country entries")
    _require_nonempty(emails, "\\email entries")

    invalid_emails = [email for email in emails if not EMAIL_RE.fullmatch(email)]
    if invalid_emails:
        raise MetadataValidationError(f"invalid author email entries: {invalid_emails}")
    if any(email.casefold().endswith(".invalid") for email in emails):
        raise MetadataValidationError("author email may not use the .invalid domain")

    main = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    if "\\documentclass[sigconf,review]{acmart}" not in main:
        raise MetadataValidationError("main.tex is not in KDD single-blind review mode")
    if "\\documentclass[sigconf,review,anonymous]{acmart}" in main:
        raise MetadataValidationError("main.tex must not enable anonymous review mode")
    if "\\input{author_metadata}" not in main:
        raise MetadataValidationError("main.tex does not import author_metadata.tex")

    return {
        "author_count": len(authors),
        "institution_count": len(institutions),
        "city_count": len(cities),
        "country_count": len(countries),
        "email_count": len(emails),
        "metadata_sha256": sha256(path),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not read PDF page count: {path}")
    return int(match.group(1))


def build_submission_manifest(
    *,
    output_pdf: Path,
    metadata_summary: dict[str, object],
    strict_report: dict[str, object],
) -> dict[str, object]:
    return {
        "author_count": metadata_summary["author_count"],
        "email_count": metadata_summary["email_count"],
        "institution_count": metadata_summary["institution_count"],
        "metadata_sha256": metadata_summary["metadata_sha256"],
        "output_filename": output_pdf.name,
        "pdf_bytes": output_pdf.stat().st_size,
        "pdf_pages": _pdf_pages(output_pdf),
        "pdf_sha256": sha256(output_pdf),
        "strict_verification_passed": strict_report.get("passed"),
        "strict_verification_success": strict_report.get("success"),
        "strict_verification_total": strict_report.get("total"),
        "submission_ready": strict_report.get("success") is True,
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
    }


def finalize_submission(*, metadata_path: Path, output_pdf: Path) -> dict[str, object]:
    metadata_path = metadata_path.resolve()
    if metadata_path != CANONICAL_METADATA.resolve():
        raise MetadataValidationError(
            "finalization requires paper/author_metadata.tex; external files are "
            "supported only by --check-only"
        )
    metadata_summary = validate_author_metadata(metadata_path)
    _run(
        ["make", "-C", str(PAPER_DIR), submission_paper_target()],
        cwd=ROOT,
    )
    _run(["make", "-C", str(PAPER_DIR), "html"], cwd=ROOT)
    _run(
        [
            sys.executable,
            str(PAPER_DIR / "verify_paper.py"),
            "--defer-submission-output",
        ],
        cwd=ROOT,
    )

    strict_report = json.loads(STRICT_REPORT.read_text(encoding="utf-8"))
    if strict_report.get("success") is not True:
        raise RuntimeError("strict paper verification did not pass")
    if not CANONICAL_PDF.is_file():
        raise RuntimeError(f"submission PDF was not generated: {CANONICAL_PDF}")

    output_pdf = output_pdf.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = output_pdf.with_name(f".{output_pdf.name}.tmp")
    shutil.copy2(CANONICAL_PDF, temporary_pdf)
    temporary_pdf.replace(output_pdf)
    manifest = build_submission_manifest(
        output_pdf=output_pdf,
        metadata_summary=metadata_summary,
        strict_report=strict_report,
    )
    manifest_path = output_pdf.with_suffix(".submission.json")
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    checksum_path = Path(f"{output_pdf}.sha256")
    _atomic_write(
        checksum_path,
        f"{manifest['pdf_sha256']}  {output_pdf.name}\n".encode("utf-8"),
    )
    _run([sys.executable, str(PAPER_DIR / "verify_paper.py")], cwd=ROOT)
    final_strict_report = json.loads(STRICT_REPORT.read_text(encoding="utf-8"))
    if final_strict_report.get("success") is not True:
        raise RuntimeError("final output attestation did not pass strict verification")
    return {
        "manifest": str(manifest_path),
        "output_pdf": str(output_pdf),
        "sha256": manifest["pdf_sha256"],
        "sha256_file": str(checksum_path),
        "strict_verification": (
            f"{final_strict_report['passed']}/{final_strict_report['total']}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate real author metadata and build the strict KDD submission PDF."
    )
    parser.add_argument(
        "--author-metadata",
        type=Path,
        default=CANONICAL_METADATA,
        help="UTF-8 ACM author metadata file; defaults to paper/author_metadata.tex.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Final submission PDF path.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate metadata and KDD source mode without compiling.",
    )
    args = parser.parse_args()
    try:
        summary = validate_author_metadata(args.author_metadata.resolve())
        if args.check_only:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        result = finalize_submission(
            metadata_path=args.author_metadata,
            output_pdf=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (MetadataValidationError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"submission finalization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
