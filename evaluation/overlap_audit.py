#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)
SEED_LITERAL_PATTERNS = (
    re.compile(r"(?i)\b[a-z0-9_]*seed\s*[:=]\s*(\d{1,12})\b"),
    re.compile(r"(?i)\brandom_state\s*=\s*(\d{1,12})\b"),
    re.compile(r"(?i)\bdefault_rng\(\s*(\d{1,12})\s*\)"),
    re.compile(r"(?i)\bseed\s*=\s*[^,)]*?default\s*=\s*(\d{1,12})\b"),
)
LATEX_COMMAND_RE = re.compile(
    r"\\(?:cite\w*|ref|label|input|includegraphics|begin|end|section\*?|"
    r"subsection\*?|paragraph|caption|textbf|emph|texttt)"
    r"(?:\[[^]]*\])?\{[^{}]*\}"
)

AAAI_EXCLUSIVE_HEADLINE_TERMS = (
    "executable prior validation",
    "prior-to-action validation",
    "causal analogue replay",
    "split-conformal false-authorization",
    "miniwo b",
    "miniwob",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_source(source: Path, destination: Path, attempts: int = 3) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        before = source.stat()
        shutil.copyfile(source, destination)
        after = source.stat()
        source_hash = sha256(source)
        snapshot_hash = sha256(destination)
        if (
            before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
            and source_hash == snapshot_hash
        ):
            return {
                "live_path": str(source),
                "snapshot_path": str(destination),
                "sha256": snapshot_hash,
                "size_bytes": int(after.st_size),
                "live_mtime_ns": int(after.st_mtime_ns),
            }
    raise RuntimeError(f"source changed while snapshotting: {source}")


def read_text(paths: Iterable[Path]) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in paths if path.exists()
    )


def normalize(text: str) -> str:
    text = re.sub(r"(?m)^\s*%.*$", " ", text)
    previous = None
    while previous != text:
        previous = text
        text = LATEX_COMMAND_RE.sub(" ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[`*_{}$&^~#|>]", " ", text)
    return " ".join(TOKEN_RE.findall(text.lower()))


def ngrams(text: str, n: int) -> Counter[str]:
    tokens = text.split()
    return Counter(" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def overlap(left: str, right: str, n: int, limit: int = 30) -> dict[str, object]:
    a = ngrams(left, n)
    b = ngrams(right, n)
    shared = set(a).intersection(b)
    examples = sorted(shared, key=lambda value: (min(a[value], b[value]), value), reverse=True)
    return {
        "n": n,
        "aaai_unique": len(a),
        "audit_unique": len(b),
        "shared_unique": len(shared),
        "aaai_rate": len(shared) / max(1, len(a)),
        "audit_rate": len(shared) / max(1, len(b)),
        "examples": examples[:limit],
    }


def extract_seed_tokens(paths: Iterable[Path]) -> list[int]:
    values: set[int] = set()
    for path in paths:
        if not path.exists() or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("seeds"), dict):
                values.update(
                    int(value)
                    for value in payload["seeds"].values()
                    if isinstance(value, int) and not isinstance(value, bool)
                )
        for pattern in SEED_LITERAL_PATTERNS:
            values.update(int(match) for match in pattern.findall(text))
        for line in text.splitlines():
            if re.search(r"(?i)\bseeds\s*:", line):
                values.update(int(value) for value in re.findall(r"\b\d{1,12}\b", line))
    return sorted(values)


def file_hash_inventory(paths: Iterable[Path], max_bytes: int = 100_000_000) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in paths:
        if path.is_file() and path.stat().st_size <= max_bytes:
            inventory[str(path)] = sha256(path)
    return inventory


def render_average_hash(pdf: Path) -> str | None:
    if shutil.which("pdftoppm") is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-gray", "-r", "72", "-png", str(pdf), str(prefix)],
            check=True,
            capture_output=True,
        )
        with Image.open(prefix.with_suffix(".png")) as image:
            pixels = list(image.resize((32, 32)).convert("L").get_flattened_data())
    mean = sum(pixels) / len(pixels)
    return "".join("1" if value >= mean else "0" for value in pixels)


def similarity(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return 1.0 - sum(a != b for a, b in zip(left, right)) / len(left)


def figure_audit(aaai_dirs: list[Path], audit_dirs: list[Path]) -> dict[str, object]:
    aaai_paths = sorted({path for directory in aaai_dirs if directory.exists() for path in directory.glob("*.pdf")})
    audit_paths = sorted({path for directory in audit_dirs if directory.exists() for path in directory.glob("*.pdf")})
    aaai_exact = {str(path): sha256(path) for path in aaai_paths}
    audit_exact = {str(path): sha256(path) for path in audit_paths}
    exact_overlap = sorted(set(aaai_exact.values()).intersection(audit_exact.values()))

    perceptual_matches: list[dict[str, object]] = []
    if aaai_paths and audit_paths:
        aaai_hashes = {str(path): render_average_hash(path) for path in aaai_paths}
        audit_hashes = {str(path): render_average_hash(path) for path in audit_paths}
        for a_path, a_hash in aaai_hashes.items():
            if a_hash is None:
                continue
            for b_path, b_hash in audit_hashes.items():
                if b_hash is None:
                    continue
                score = similarity(a_hash, b_hash)
                if score >= 0.92:
                    perceptual_matches.append({"aaai": a_path, "audit": b_path, "similarity": score})
    return {
        "aaai_count": len(aaai_paths),
        "audit_count": len(audit_paths),
        "exact_hash_overlap": exact_overlap,
        "perceptual_matches_at_0_92": perceptual_matches,
    }


def manuscript_headline_audit(paper_path: Path) -> dict[str, object]:
    if not paper_path.exists():
        return {"paper_exists": False, "violations": []}
    text = paper_path.read_text(encoding="utf-8", errors="ignore").lower()
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    title_match = re.search(r"\\title\{(.*?)\}", text, re.S)
    contributions_match = re.search(
        r"(?:our contributions are|contributions)(.*?)(?:\\section|\\subsection)", text, re.S
    )
    surfaces = {
        "title": title_match.group(1) if title_match else "",
        "abstract": abstract_match.group(1) if abstract_match else "",
        "contributions": contributions_match.group(1) if contributions_match else "",
    }
    violations: list[dict[str, str]] = []
    for surface, value in surfaces.items():
        for term in AAAI_EXCLUSIVE_HEADLINE_TERMS:
            if term in value:
                violations.append({"surface": surface, "term": term})
        if "epv" in value:
            violations.append({"surface": surface, "term": "epv"})
    return {"paper_exists": True, "violations": violations}
def markdown(report: dict[str, object]) -> str:
    lines = [
        "# FinAuth-Audit AAAI Overlap Audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "This audit reads the actual AAAI source and screens the current FinAuth-Audit scope/manuscript.",
        "The audited AAAI text is copied to immutable round-local snapshots before comparison, so later upstream edits do not rewrite this audit's evidence.",
        "It is a mechanical guard, not a substitute for claim-level human review.",
        "",
        "## Source snapshots",
        "",
    ]
    for item in report["source_observations"]:
        lines.append(
            f"- `{item['live_path']}` -> `{item['snapshot_path']}` "
            f"SHA-256 `{item['sha256']}`"
        )
    lines.extend(
        [
            "",
        "## Text overlap",
        "",
        ]
    )
    for item in report["text_overlap"]:
        lines.append(
            f"- {item['n']}-gram: shared={item['shared_unique']}, "
            f"AAAI rate={item['aaai_rate']:.6f}, audit rate={item['audit_rate']:.6f}"
        )
        for example in item["examples"][:8]:
            lines.append(f"  - `{example}`")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- AAAI PDFs: {report['figures']['aaai_count']}",
            f"- FinAuth-Audit PDFs: {report['figures']['audit_count']}",
            f"- Exact overlaps: {len(report['figures']['exact_hash_overlap'])}",
            f"- Perceptual matches >= 0.92: {len(report['figures']['perceptual_matches_at_0_92'])}",
            "",
            "## Seeds and data hashes",
            "",
            f"- AAAI seed tokens: {report['seeds']['aaai']}",
            f"- FinAuth-Audit seed tokens: {report['seeds']['audit']}",
            f"- Shared seed tokens: {report['seeds']['shared']}",
            f"- Exact data-file hash overlaps: {len(report['data_hash_overlap'])}",
            "",
            "## Headline firewall",
            "",
            f"- Manuscript exists: {report['headline_firewall']['paper_exists']}",
            f"- Forbidden headline occurrences: {len(report['headline_firewall']['violations'])}",
            "",
            "## Interpretation",
            "",
            "Scope documents intentionally name AAAI-exclusive concepts in the firewall ledger, so their exact-term overlap is expected. Before submission, rerun this audit against the manuscript and figures; any long passage, exact/perceptual figure match, shared seed, exact data hash, or forbidden headline occurrence is a blocking issue.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FinAuth-Audit separation from the AAAI EPV project.")
    parser.add_argument("--aaai-project", default="/opt/projects/research/epv_aaai")
    parser.add_argument("--audit-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    aaai_project = Path(args.aaai_project).resolve()
    audit_root = Path(args.audit_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else audit_root / "results" / "overlap_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    live_aaai_text_paths = [
        aaai_project / "paper" / "aaai" / "main.tex",
        aaai_project / "paper" / "aaai" / "supplement.tex",
    ]
    snapshot_dir = out_dir / "source_snapshot"
    source_observations = [
        snapshot_source(path, snapshot_dir / f"{index:02d}_{path.name}")
        for index, path in enumerate(live_aaai_text_paths, start=1)
        if path.exists()
    ]
    aaai_text_paths = [Path(item["snapshot_path"]) for item in source_observations]
    audit_text_paths = sorted((audit_root / "docs").glob("*.md")) + sorted((audit_root / "paper").glob("*.tex"))
    aaai_raw = read_text(aaai_text_paths)
    audit_raw = read_text(audit_text_paths)

    aaai_config_paths = sorted((aaai_project / "configs").rglob("*.yaml"))
    aaai_config_paths += sorted((aaai_project / "scripts").rglob("*.py"))
    aaai_config_paths += sorted((aaai_project / "epv").rglob("*.py"))
    aaai_config_paths += sorted((aaai_project / "epv_eval").rglob("*.py"))
    audit_config_paths = sorted((audit_root / "configs").rglob("*.yaml"))
    audit_config_paths += sorted((audit_root / "manifests").glob("*seed*.json"))

    aaai_data_paths = sorted((aaai_project / "data").rglob("*.csv")) + sorted((aaai_project / "data").rglob("*.json"))
    audit_data_paths = sorted((audit_root / "data").rglob("*.csv")) + sorted((audit_root / "data").rglob("*.json"))
    aaai_data_hashes = file_hash_inventory(aaai_data_paths)
    audit_data_hashes = file_hash_inventory(audit_data_paths)

    aaai_seeds = extract_seed_tokens(aaai_config_paths)
    audit_seeds = extract_seed_tokens(audit_config_paths)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aaai_project": str(aaai_project),
        "audit_root": str(audit_root),
        "source_observations": source_observations,
        "source_hashes": {
            str(item["snapshot_path"]): str(item["sha256"])
            for item in source_observations
        },
        "text_overlap": [overlap(normalize(aaai_raw), normalize(audit_raw), n) for n in (8, 12)],
        "numeric_overlap": sorted(set(NUMBER_RE.findall(aaai_raw)).intersection(NUMBER_RE.findall(audit_raw))),
        "figures": figure_audit(
            [aaai_project / "paper" / "aaai" / "figures", aaai_project / "paper" / "figures"],
            [audit_root / "figures", audit_root / "paper" / "figures"],
        ),
        "seeds": {
            "aaai": aaai_seeds,
            "audit": audit_seeds,
            "shared": sorted(set(aaai_seeds).intersection(audit_seeds)),
        },
        "data_hash_overlap": sorted(set(aaai_data_hashes.values()).intersection(audit_data_hashes.values())),
        "headline_firewall": manuscript_headline_audit(audit_root / "paper" / "main.tex"),
    }
    report["blocking"] = bool(
        report["figures"]["exact_hash_overlap"]
        or report["figures"]["perceptual_matches_at_0_92"]
        or report["seeds"]["shared"]
        or report["data_hash_overlap"]
        or report["headline_firewall"]["violations"]
    )

    json_path = out_dir / "aaai_overlap_audit.json"
    md_path = out_dir / "aaai_overlap_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 1 if report["blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
