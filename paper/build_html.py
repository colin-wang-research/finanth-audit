#!/usr/bin/env python3
"""Build a page-faithful HTML bundle from the verified combined paper PDF."""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
PDF_PATH = PAPER_DIR / "FinAuth-Audit.pdf"
DEFAULT_OUTPUT = PAPER_DIR / "html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return result.stdout + result.stderr


def local_asset_refs(document: str, output_dir: Path) -> list[str]:
    refs = re.findall(r"(?:src|href)=\"([^\"]+)\"", document)
    missing: list[str] = []
    for ref in refs:
        if ref.startswith(("#", "http://", "https://", "mailto:", "data:")):
            continue
        relative = ref.split("#", 1)[0].split("?", 1)[0]
        if relative and not (output_dir / relative).is_file():
            missing.append(ref)
    return sorted(set(missing))


def patch_document(document: str) -> str:
    document = document.replace(
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="" xml:lang="">',
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">',
    )
    if 'name="viewport"' not in document:
        document = document.replace(
            "<head>",
            '<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">',
            1,
        )
    publication_css = """
<style type="text/css">
html { scroll-behavior: smooth; }
body {
  margin: 0 !important;
  padding: 20px 0 40px;
  min-width: 918px;
  background: #e9edf0 !important;
}
div[id^="page"] {
  margin: 0 auto 22px;
  background: #ffffff;
  box-shadow: 0 2px 12px rgba(31, 41, 51, 0.14);
}
@media print {
  body { padding: 0; background: #ffffff !important; }
  div[id^="page"] { margin: 0; box-shadow: none; page-break-after: always; }
}
</style>
"""
    return document.replace("</head>", publication_css + "</head>", 1)


def visible_text(document: str) -> str:
    text = re.sub(r"<style.*?</style>", " ", document, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html_module.unescape(text)).strip()


def pdf_page_count() -> int:
    info = command_output(["pdfinfo", str(PDF_PATH)])
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo did not report a page count")
    return int(match.group(1))


def build(output_dir: Path) -> dict[str, object]:
    pdftohtml = shutil.which("pdftohtml")
    if not pdftohtml:
        raise RuntimeError("missing required HTML build tool: pdftohtml")
    if not PDF_PATH.is_file():
        raise RuntimeError(f"missing combined paper PDF: {PDF_PATH}")

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_pages = pdf_page_count()
    with tempfile.TemporaryDirectory(prefix="finauth-audit-pdftohtml-") as temp_name:
        temp_dir = Path(temp_name)
        temporary_html = temp_dir / "FinAuth-Audit.html"
        build_output = command_output(
            [
                pdftohtml,
                "-c",
                "-hidden",
                "-noframes",
                "-enc",
                "UTF-8",
                "-zoom",
                "1.5",
                str(PDF_PATH),
                str(temporary_html),
            ]
        )
        document = patch_document(temporary_html.read_text(encoding="utf-8"))
        final_html = output_dir / "FinAuth-Audit.html"
        final_html.write_text(document, encoding="utf-8")
        page_images = sorted(temp_dir.glob("FinAuth-Audit[0-9][0-9][0-9].png"))
        for image in page_images:
            shutil.copy2(image, output_dir / image.name)

    rendered_pages = len(re.findall(r'<div id="page\d+-div"', document))
    missing_assets = local_asset_refs(document, output_dir)
    text = visible_text(document)
    required_markers = {
        "title": (
            "FinAuth-Audit: Validity-Aware Benchmarking of Financial Agent Authorization"
            in text
        ),
        "references": "REFERENCES" in text,
        "supplement": "ARTIFACT AND REPRODUCTION" in text,
    }
    unresolved = bool(re.search(r"\[\?\]|undefined reference|undefined citation", text, re.I))
    success = (
        rendered_pages == expected_pages
        and len(page_images) == expected_pages
        and not missing_assets
        and all(required_markers.values())
        and not unresolved
    )
    manifest = {
        "artifact": "FinAuth-Audit page-faithful HTML",
        "source_pdf_sha256": sha256(PDF_PATH),
        "expected_pages": expected_pages,
        "rendered_pages": rendered_pages,
        "page_images": len(page_images),
        "missing_local_assets": missing_assets,
        "required_markers": required_markers,
        "unresolved_reference_tokens": unresolved,
        "success": success,
        "outputs": {
            str(path.relative_to(output_dir)): sha256(path)
            for path in sorted(output_dir.glob("FinAuth-Audit*"))
            if path.is_file()
        },
    }
    (output_dir / "html_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "build.log").write_text(build_output + "\n", encoding="utf-8")
    if not success:
        raise RuntimeError(f"HTML verification failed: {json.dumps(manifest, sort_keys=True)}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output_dir)
    print(f"HTML written to {args.output_dir / 'FinAuth-Audit.html'}")
    print(
        f"pages={manifest['rendered_pages']} "
        f"missing_assets={manifest['missing_local_assets']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
