from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from finauth_audit.release.build_release import build_release, collect_files
from finauth_audit.release.verify_release import (
    audit_claim_consistency,
    audit_member_payload,
    verify_archive,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "release" / "release_spec.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec() -> dict[str, object]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_public_release_identity_and_core_outputs() -> None:
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "0.6.0"
    paper_pdf = ROOT / "paper" / "FinAuth-Audit.pdf"
    assert paper_pdf.stat().st_size > 100_000
    assert _sha256(paper_pdf) == (
        "153834ba3fda929f34f322a17c1b14f1d76c3eb7b9e3454a02aac6ab04b61cb7"
    )
    author_metadata = (ROOT / "paper" / "author_metadata.tex").read_text(
        encoding="utf-8"
    )
    for token in (
        "Ke Wang",
        "Xiaorui Tang",
        "Chanjin (Guangzhou) Technology Development",
        "colinwang@gatech.edu",
        "alicetang0618@gmail.com",
    ):
        assert token in author_metadata
    assert "Anonymous Author(s)" not in author_metadata
    assert (ROOT / "paper" / "supplement.tex").stat().st_size > 1_000
    assert (ROOT / "paper" / "html" / "FinAuth-Audit.html").stat().st_size > 10_000
    figure_manifest = json.loads(
        (ROOT / "paper" / "figures" / "figure_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(figure_manifest["figures"]) == 5
    html_dir = ROOT / "paper" / "html"
    page_images = sorted(html_dir.glob("FinAuth-Audit[0-9][0-9][0-9].png"))
    assert len(page_images) == 20
    assert not (html_dir / "FinAuth-Audit021.png").exists()


def test_pinned_v06_aggregates_are_exact_and_payload_safe() -> None:
    spec = _spec()
    pinned = {
        str(path): str(digest)
        for path, digest in dict(spec["pinned_member_sha256"]).items()
    }
    assert len(pinned) == 12
    for relative, expected in pinned.items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative
        audit_member_payload(relative, path.read_bytes())


def test_public_v06_verification_is_aggregate_only() -> None:
    report_path = (
        ROOT
        / "results"
        / "verification"
        / "public"
        / "real-agent-v06_verification.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["passed"] == report["total"] == 8
    assert report["internal_passed"] == report["internal_total"] == 771
    serialized = json.dumps(report, sort_keys=True).lower()
    for forbidden in ("community_hidden", "private_key", "ciphertext", "decisions.csv"):
        assert forbidden not in serialized


def test_release_contains_no_row_level_or_hidden_material() -> None:
    relative_files = {
        path.relative_to(ROOT).as_posix()
        for path in collect_files(ROOT, _spec())
    }
    assert not {
        path
        for path in relative_files
        if "community_hidden" in path
        or "community-hidden" in path
        or path.endswith("/decisions.csv")
        or path.endswith(".key")
        or path.endswith(".enc")
    }


def test_release_excludes_superseded_pilot_and_artifact_card_is_current() -> None:
    relative_files = {
        path.relative_to(ROOT).as_posix()
        for path in collect_files(ROOT, _spec())
    }
    assert not {
        path for path in relative_files if path.startswith("results/real_agent_v05/")
    }
    status = (ROOT / "ARTIFACT_CARD.md").read_bytes()
    audit_claim_consistency({"ARTIFACT_CARD.md": status}, version="0.6.0")

    with pytest.raises(RuntimeError, match="superseded v0.5 pilot"):
        audit_claim_consistency(
            {
                "ARTIFACT_CARD.md": status,
                "results/real_agent_v05/paper_test/ranking_transfer.json": b"{}",
            },
            version="0.6.0",
        )

    stale_status = status.replace(b"Spearman rho: **+0.657**", b"-0.928")
    with pytest.raises(RuntimeError, match="stale or claim-incomplete"):
        audit_claim_consistency(
            {"ARTIFACT_CARD.md": stale_status}, version="0.6.0"
        )


def test_reviewer_tree_excludes_internal_workspace_files() -> None:
    for relative in (
        "FINAL_STATUS_REPORT.md",
        "paper/main_internal.tex",
        "docs/submission_handoff.md",
        "docs/aaai_kdd_overlap_ledger.md",
        "docs/current_artifact_audit.md",
        "docs/risk_register.md",
    ):
        assert not (ROOT / relative).exists(), relative
    assert (ROOT / "REVIEWER_GUIDE.md").is_file()
    assert (ROOT / "ARTIFACT_CARD.md").is_file()
    public_verifier = ROOT / "verify_artifact.py"
    assert public_verifier.is_file()
    assert public_verifier.stat().st_size < 10_000
    verifier_text = public_verifier.read_text(encoding="utf-8")
    assert "manifests/real_agent_v06" not in verifier_text
    assert "community_hidden" not in verifier_text


def test_extracted_release_rebuilds_and_verifies(tmp_path: Path) -> None:
    archive, _ = build_release(
        root=ROOT,
        spec_path=SPEC_PATH,
        trusted_spec_path=SPEC_PATH,
        output_dir=tmp_path,
    )
    checksum = Path(f"{archive}.sha256").read_text(encoding="utf-8").strip()
    assert checksum == f"{_sha256(archive)}  {archive.name}"
    report = verify_archive(
        archive,
        clean_room=True,
        trusted_spec_path=SPEC_PATH,
    )
    assert report["passed"] is True
    assert report["install_checked"] is True
    assert report["installed_package"] == "finauth_audit"
