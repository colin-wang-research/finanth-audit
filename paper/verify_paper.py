#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
ROOT = PAPER_DIR.parent
WORKSPACE = ROOT.parent
PDF_PATH = PAPER_DIR / "FinAuth-Audit.pdf"
HTML_DIR = PAPER_DIR / "html"
HTML_PATH = HTML_DIR / "FinAuth-Audit.html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def submission_output_path() -> Path:
    workspace_output = WORKSPACE / "paper" / "kdd" / "FinAuth-Audit-KDD.pdf"
    if ROOT.name == "finauth_audit" and workspace_output.parent.is_dir():
        return workspace_output
    return ROOT / "dist" / "FinAuth-Audit-KDD.pdf"


def command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PAPER_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout


def add_check(
    checks: list[dict[str, object]], check: str, passed: bool, detail: object
) -> None:
    checks.append({"check": check, "passed": bool(passed), "detail": str(detail)})


def verify_manifest_outputs(
    manifest_path: Path, output_dir: Path, checks: list[dict[str, object]]
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs", {})
    for name, expected in outputs.items():
        path = output_dir / name
        add_check(
            checks,
            f"generated hash {name}",
            path.is_file() and sha256(path) == expected,
            f"expected={expected} actual={sha256(path) if path.is_file() else 'missing'}",
        )


def extract_page(page: int) -> str:
    return command_output(
        ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(PDF_PATH), "-"]
    )


def verify_paper(
    allow_placeholder_authors: bool,
    defer_submission_output: bool = False,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    required = [
        PDF_PATH,
        PAPER_DIR / "main.tex",
        ROOT / "ARTIFACT_CARD.md",
        PAPER_DIR / "supplement.tex",
        PAPER_DIR / "references.bib",
        PAPER_DIR / "author_metadata.tex",
        PAPER_DIR / "finalize_submission.py",
        PAPER_DIR / "figures" / "figure_manifest.json",
        PAPER_DIR / "generated" / "table_manifest.json",
        PAPER_DIR / "generated" / "claims_real_agent_v06.tex",
        PAPER_DIR / "FinAuth-Audit.log",
        PAPER_DIR / "FinAuth-Audit.blg",
        HTML_PATH,
        HTML_DIR / "html_manifest.json",
        ROOT / "docs" / "benchmark_card.md",
        ROOT / "docs" / "reproducibility.md",
        ROOT / "docs" / "maintenance.md",
        ROOT / "docs" / "external_submission_protocol.md",
        ROOT / "REVIEWER_GUIDE.md",
        ROOT / "release" / "release_spec.json",
        ROOT / "results" / "verification" / "public" / "real-agent-v06_verification.json",
        ROOT / "results" / "real_agent_v06" / "paper_test" / "rank_transfer_zero_shot.json",
    ]
    for path in required:
        add_check(checks, str(path.relative_to(ROOT)), path.is_file(), "exists" if path.is_file() else "missing")
    if not all(path.is_file() for path in required):
        return checks

    main = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    author_metadata = (PAPER_DIR / "author_metadata.tex").read_text(encoding="utf-8")
    paper_makefile = (PAPER_DIR / "Makefile").read_text(encoding="utf-8")
    root_makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    release_spec = json.loads(
        (ROOT / "release" / "release_spec.json").read_text(encoding="utf-8")
    )
    supplement = (PAPER_DIR / "supplement.tex").read_text(encoding="utf-8")
    introduction = (PAPER_DIR / "sections" / "01_introduction.tex").read_text(
        encoding="utf-8"
    )
    section_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PAPER_DIR / "sections").glob("*.tex"))
    )
    manuscript = "\n".join((main, section_sources, supplement))
    v06_public_verification = json.loads(
        (
            ROOT
            / "results"
            / "verification"
            / "public"
            / "real-agent-v06_verification.json"
        ).read_text(encoding="utf-8")
    )
    v06_public_checks = {
        item.get("check"): item
        for item in v06_public_verification.get("checks", [])
        if isinstance(item, dict)
    }
    v06_rank = json.loads(
        (
            ROOT
            / "results"
            / "real_agent_v06"
            / "paper_test"
            / "rank_transfer_zero_shot.json"
        ).read_text(encoding="utf-8")
    )
    single_blind_source = (
        "\\documentclass[sigconf,review,screen]{acmart}" in main
        and "anonymous" not in main.partition("\\documentclass")[2].partition("}")[0]
        and "\\input{author_metadata}" in main
    )
    placeholder_tokens = (
        "Author Metadata Required",
        "Replace with the submitting institution",
        "Replace before submission",
        "replace-before-submission@example.invalid",
    )
    present_placeholders = [token for token in placeholder_tokens if token in author_metadata]
    metadata_shape = all(
        token in author_metadata
        for token in ("\\author{", "\\institution{", "\\city{", "\\country{", "\\email{")
    )
    add_check(
        checks,
        "KDD single-blind source mode",
        single_blind_source,
        "main.tex uses sigconf,review,screen and inputs author_metadata.tex",
    )
    add_check(
        checks,
        "single-blind author metadata",
        metadata_shape and (allow_placeholder_authors or not present_placeholders),
        (
            f"placeholder_waiver={allow_placeholder_authors} "
            f"placeholders={present_placeholders}"
        ),
    )
    submission_required = set(release_spec.get("required_members", []))
    finalizer_wiring = all(
        token in root_makefile and token in paper_makefile
        for token in ("submission-preflight", "submission-finalize")
    ) and {
        "paper/finalize_submission.py",
        "ARTIFACT_CARD.md",
    }.issubset(submission_required)
    add_check(
        checks,
        "strict submission finalizer wiring",
        finalizer_wiring,
        "Make targets and required public-release members",
    )
    preflight = subprocess.run(
        [sys.executable, str(PAPER_DIR / "finalize_submission.py"), "--check-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    expected_preflight_code = 2 if present_placeholders else 0
    add_check(
        checks,
        "author-metadata submission preflight",
        preflight.returncode == expected_preflight_code,
        (
            f"expected_exit={expected_preflight_code} actual_exit={preflight.returncode} "
            f"placeholders={present_placeholders}"
        ),
    )

    add_check(
        checks,
        "KDD 2027 conference metadata",
        all(
            token in main
            for token in (
                "KDD '27",
                "August 1--5, 2027",
                "San Jose, CA, USA",
                "\\documentclass[sigconf,review,screen]{acmart}",
            )
        ),
        "single-blind screen review-mode acmart; KDD 2027 dates and location",
    )
    add_check(
        checks,
        "combined supplement",
        "\\appendix" in main and "\\input{supplement}" in main,
        "main and supplement compile into one PDF",
    )
    add_check(
        checks,
        "binding v0.6 paper-test registry",
        v06_public_verification.get("success") is True
        and v06_public_checks.get("exactly-once paper-test registry", {}).get("passed")
        is True
        and "status=COMPLETED clusters=200 proposal_rows=1200"
        in v06_public_checks.get("exactly-once paper-test registry", {}).get(
            "detail", ""
        )
        and v06_public_checks.get("community-hidden split remains sealed", {}).get(
            "passed"
        )
        is True
        and "outcomes_evaluated=false proposals_decrypted=false"
        in v06_public_checks.get("community-hidden split remains sealed", {}).get(
            "detail", ""
        ),
        "public aggregate verification proves the completed registry and sealed hidden split",
    )
    add_check(
        checks,
        "v0.6 inverse-rank result retained",
        v06_rank.get("primary_support") is False
        and "not supported" in manuscript
        and "larger prospective" in manuscript,
        (
            f"primary_support={v06_rank.get('primary_support')} "
            f"rho={v06_rank.get('point', {}).get('spearman_rho')}"
        ),
    )
    stale_claims = [
        token
        for token in (
            "100-date actual-model proposal layer",
            "actual-model test whose rule ranking reverses",
            "actual-proposal test reverses the controlled FAR ranking",
            "The v0.5 extension evaluates actual cached proposal outputs",
            "controlled-to-agent FAR rank correlation is $-0.928$",
        )
        if token in manuscript
    ]
    add_check(
        checks,
        "no superseded v0.5 headline claims",
        not stale_claims,
        f"stale_claims={stale_claims}",
    )
    add_check(
        checks,
        "negative-utility terminology contract",
        "negative-utility authorization rate (NUAR)" in manuscript
        and "\\mathrm{NUAR}" in manuscript
        and "historical field name \\texttt{FAR}" in manuscript
        and "false authorization rate" not in manuscript.casefold(),
        "NUAR displayed; legacy far field documented; generic false-authorization-rate wording absent",
    )
    add_check(
        checks,
        "provenance partial-identification bound",
        "\\label{eq:provenance-partial-id}" in manuscript
        and "Derivation of Equation~\\eqref{eq:provenance-partial-id}" in supplement,
        "bound stated in main paper and derived in supplement",
    )
    add_check(
        checks,
        "benchmark overview in main paper",
        "\\label{fig:overview}" in introduction
        and "\\label{fig:certification}" in supplement
        and "\\label{fig:certification}" not in introduction,
        "overview is main Figure 1; certification surface is supplementary",
    )

    html_manifest = json.loads(
        (HTML_DIR / "html_manifest.json").read_text(encoding="utf-8")
    )
    add_check(
        checks,
        "combined page-faithful HTML bundle",
        bool(html_manifest.get("success"))
        and html_manifest.get("rendered_pages") == html_manifest.get("expected_pages")
        and not html_manifest.get("missing_local_assets")
        and not html_manifest.get("unresolved_reference_tokens"),
        (
            f"pages={html_manifest.get('rendered_pages')}/"
            f"{html_manifest.get('expected_pages')} "
            f"missing_assets={html_manifest.get('missing_local_assets')}"
        ),
    )
    add_check(
        checks,
        "HTML source matches current PDF",
        html_manifest.get("source_pdf_sha256") == sha256(PDF_PATH),
        (
            f"manifest={html_manifest.get('source_pdf_sha256')} "
            f"current={sha256(PDF_PATH)}"
        ),
    )
    verify_manifest_outputs(HTML_DIR / "html_manifest.json", HTML_DIR, checks)

    info = command_output(["pdfinfo", str(PDF_PATH)])
    page_match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    pages = int(page_match.group(1)) if page_match else 0
    add_check(checks, "PDF page count", 10 <= pages <= 30, f"pages={pages}")
    page_text = {page: extract_page(page) for page in range(1, pages + 1)}
    actual_model_pages = "\n".join(
        page_text.get(page, "") for page in range(6, min(pages, 8) + 1)
    )
    rendered_v06_values = (
        "1,200 cached proposals",
        "200 independent UTC",
        "not supported",
        "0.657",
        "0.600",
        "0.932",
        "0.173",
    )
    missing_v06_values = [
        value for value in rendered_v06_values if value not in actual_model_pages
    ]
    add_check(
        checks,
        "rendered v0.6 claim macros",
        not missing_v06_values,
        f"missing={missing_v06_values}",
    )
    author_names = [
        value.strip()
        for value in re.findall(r"\\author\{([^{}]+)\}", author_metadata)
        if value.strip()
    ]
    page_one = page_text.get(1, "")
    named_authors_displayed = (
        bool(author_names)
        and "Anonymous Author(s)" not in page_one
        and all(name in page_one for name in author_names)
    )
    displayed_authors_ok = (
        "Anonymous Author(s)" in page_one or named_authors_displayed
        if allow_placeholder_authors
        else named_authors_displayed
    )
    add_check(
        checks,
        "displayed author policy",
        displayed_authors_ok,
        (
            "internal anonymous wrapper or single-blind author page allowed"
            if allow_placeholder_authors
            else f"single-blind authors expected={author_names}"
        ),
    )
    submission_pdf = submission_output_path()
    submission_manifest = submission_pdf.with_suffix(".submission.json")
    submission_checksum = Path(f"{submission_pdf}.sha256")
    submission_detail = "strict submission path is absent until finalization"
    submission_output_safe = not submission_pdf.exists()
    if defer_submission_output:
        submission_output_safe = True
        submission_detail = "output attestation deferred until finalizer writes the new artifact"
    elif submission_pdf.exists():
        try:
            submission_payload = json.loads(
                submission_manifest.read_text(encoding="utf-8")
            )
            checksum_text = submission_checksum.read_text(encoding="utf-8")
            submission_hash = sha256(submission_pdf)
            submission_output_safe = (
                submission_payload.get("submission_ready") is True
                and submission_payload.get("pdf_sha256") == submission_hash
                and checksum_text.strip()
                == f"{submission_hash}  {submission_pdf.name}"
            )
            submission_detail = (
                f"ready={submission_payload.get('submission_ready')} "
                f"sha256={submission_hash}"
            )
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            submission_output_safe = False
            submission_detail = f"unverified submission artifact: {exc}"
    add_check(
        checks,
        "strict submission output safety",
        submission_output_safe,
        submission_detail,
    )
    add_check(
        checks,
        "eight-page main-paper boundary",
        any("CONCLUSION" in page_text.get(page, "") for page in range(1, 9))
        and "REFERENCES" not in page_text.get(8, "")
        and "REFERENCES" in page_text.get(9, "")
        and "WebArena: A Realistic Web Environment for Building Autonomous Agents"
        in re.sub(r"\s+", " ", page_text.get(9, "")),
        (
            "conclusion is within the first eight pages; references start on "
            "page 9 and include the final cited work"
        ),
    )
    add_check(
        checks,
        "supplement starts after references",
        "ARTIFACT AND REPRODUCTION" in page_text.get(10, ""),
        "supplement marker expected on page 10",
    )
    blank_pages = [page for page, text in page_text.items() if len(text.strip()) < 40]
    add_check(checks, "no blank PDF pages", not blank_pages, f"blank_pages={blank_pages}")

    log_text = (PAPER_DIR / "FinAuth-Audit.log").read_text(encoding="utf-8", errors="replace")
    blg_text = (PAPER_DIR / "FinAuth-Audit.blg").read_text(encoding="utf-8", errors="replace")
    blocking_patterns = {
        "undefined references": "There were undefined references",
        "undefined citations": "Citation `",
        "overfull horizontal boxes": "Overfull \\hbox",
    }
    for name, pattern in blocking_patterns.items():
        add_check(checks, name, pattern not in log_text, f"pattern={pattern!r}")
    vertical_overfull = [
        float(value)
        for value in re.findall(r"Overfull \\vbox \((\d+(?:\.\d+)?)pt too high\)", log_text)
    ]
    maximum_vertical_overfull = max(vertical_overfull, default=0.0)
    add_check(
        checks,
        "vertical page-box tolerance",
        maximum_vertical_overfull <= 5.0,
        f"max_overfull_pt={maximum_vertical_overfull:.3f}; visually audited threshold=5.0",
    )
    add_check(checks, "BibTeX metadata warnings", "Warning--" not in blg_text, "no BibTeX warnings")

    fonts = command_output(["pdffonts", str(PDF_PATH)])
    font_rows = [line.split() for line in fonts.splitlines()[2:] if line.strip()]
    unembedded = [row[0] for row in font_rows if len(row) >= 6 and row[5].lower() != "yes"]
    add_check(checks, "embedded PDF fonts", not unembedded, f"unembedded={unembedded}")

    figure_manifest = json.loads(
        (PAPER_DIR / "figures" / "figure_manifest.json").read_text(encoding="utf-8")
    )
    figures = figure_manifest.get("figures", [])
    figure_clean = len(figures) == 5 and all(
        figure.get("text_qa", {}).get("minimum_font_points", 0) >= 8.0
        and figure.get("text_qa", {}).get("all_text_overlap_count") == 0
        and figure.get("text_qa", {}).get("qa_label_overlap_count") == 0
        and figure.get("text_qa", {}).get("qa_label_point_overlap_count") == 0
        and figure.get("text_qa", {}).get("qa_label_boundary_violation_count") == 0
        and figure.get("text_qa", {}).get("legend_axes_overlap_count") == 0
        for figure in figures
    )
    add_check(checks, "five standardized paper figures", figure_clean, f"figures={len(figures)}")
    for figure in figures:
        for suffix, expected in figure.get("hashes", {}).items():
            path = PAPER_DIR / "figures" / f"{figure['name']}.{suffix}"
            add_check(
                checks,
                f"figure hash {path.name}",
                path.is_file() and sha256(path) == expected,
                f"expected={expected} actual={sha256(path) if path.is_file() else 'missing'}",
            )

    table_manifest_path = PAPER_DIR / "generated" / "table_manifest.json"
    table_manifest = json.loads(table_manifest_path.read_text(encoding="utf-8"))
    generated_table_outputs = {
        name: digest
        for name, digest in table_manifest.get("outputs", {}).items()
        if name.startswith("table_") and name.endswith(".tex")
    }
    generated_table_count = len(generated_table_outputs)
    add_check(
        checks,
        "script-generated table set",
        generated_table_count >= 17,
        f"tables={generated_table_count}",
    )
    verify_manifest_outputs(table_manifest_path, PAPER_DIR / "generated", checks)
    table_count = manuscript.count("\\begin{table}") + manuscript.count("\\begin{table*}")
    format_count = manuscript.count("\\benchmarktableformat") - 1
    generated_input_count = len(
        re.findall(r"\\input\{generated/table_[^}]+\}", manuscript)
    )
    has_resizebox = "\\resizebox" in manuscript
    add_check(
        checks,
        "uniform table typography",
        generated_input_count == generated_table_count
        and format_count == table_count
        and not has_resizebox,
        (
            f"table_environments={table_count} generated_inputs={generated_input_count} "
            f"format_uses={format_count} resizebox={'yes' if has_resizebox else 'no'}"
        ),
    )

    references = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    reference_count = len(re.findall(r"^\s*@", references, flags=re.MULTILINE))
    bibliography_safe = (
        "\\balance" not in main
        and "\\nobalance" not in main
        and "\\documentclass[sigconf,review,screen]{acmart}" in main
        and all(
            f"{setting}=FinAuthLinkBlue" in main
            for setting in ("linkcolor", "citecolor", "urlcolor", "filecolor")
        )
        and main.index("\\begin{document}") < main.index("\\hypersetup")
    )
    add_check(
        checks,
        "reference count",
        reference_count >= 35 and bibliography_safe,
        f"references={reference_count}; complete-layout-and-link-style={bibliography_safe}",
    )
    banned_submission_tokens = [
        token
        for token in ("Conference Placeholder", "anonymous@example.com", "Anonymous Institution")
        if token in manuscript
    ]
    add_check(
        checks,
        "no submission placeholders outside author metadata",
        not banned_submission_tokens,
        f"tokens={banned_submission_tokens}",
    )
    return checks


def write_report(
    checks: list[dict[str, object]],
    allow_placeholder_authors: bool,
    defer_submission_output: bool = False,
) -> tuple[Path, Path]:
    passed = sum(bool(check["passed"]) for check in checks)
    report = {
        "allow_placeholder_authors": allow_placeholder_authors,
        "defer_submission_output": defer_submission_output,
        "passed": passed,
        "total": len(checks),
        "success": passed == len(checks),
        "checks": checks,
    }
    output_dir = ROOT / "results" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "internal" if allow_placeholder_authors else "strict"
    json_path = output_dir / f"paper_verification_{suffix}.json"
    md_path = output_dir / f"paper_verification_{suffix}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# FinAuth-Audit Paper Verification ({suffix})",
        "",
        f"- Passed: {passed}/{len(checks)}",
        f"- Success: {report['success']}",
        f"- Placeholder-author waiver: {allow_placeholder_authors}",
        "",
        "| # | Status | Check | Detail |",
        "|---:|:---:|---|---|",
    ]
    for index, check in enumerate(checks, start=1):
        status = "PASS" if check["passed"] else "FAIL"
        name = str(check["check"]).replace("|", "\\|").replace("\n", "<br>")
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {index} | {status} | {name} | {detail} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the combined FinAuth-Audit paper package.")
    parser.add_argument(
        "--allow-placeholder-authors",
        action="store_true",
        help="Waive only the author-metadata check for internal builds.",
    )
    parser.add_argument(
        "--defer-submission-output",
        action="store_true",
        help="Defer only the output-attestation check while the finalizer replaces it.",
    )
    args = parser.parse_args()
    checks = verify_paper(
        args.allow_placeholder_authors,
        defer_submission_output=args.defer_submission_output,
    )
    json_path, md_path = write_report(
        checks,
        args.allow_placeholder_authors,
        defer_submission_output=args.defer_submission_output,
    )
    passed = sum(bool(check["passed"]) for check in checks)
    print(f"FinAuth-Audit paper verification: {passed}/{len(checks)}")
    for check in checks:
        if not check["passed"]:
            print(f"FAIL: {check['check']}: {check['detail']}", file=sys.stderr)
    print(json_path)
    print(md_path)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
