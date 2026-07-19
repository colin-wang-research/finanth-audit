#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = (
    ROOT
    / "results"
    / "verification"
    / "public"
    / "real-agent-v06_verification.json"
)


def verify_public_report() -> dict[str, object]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    valid = (
        payload.get("success") is True
        and payload.get("passed") == payload.get("total") == 8
        and payload.get("internal_passed") == payload.get("internal_total") == 771
    )
    if not valid:
        raise RuntimeError("public aggregate verification report is incomplete")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the released FinAuth-Audit aggregate attestation."
    )
    parser.add_argument(
        "--phase",
        choices=("public", "real-agent-v06"),
        default="public",
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    try:
        payload = verify_public_report()
        if args.run_tests:
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "release_tests"],
                cwd=ROOT,
                check=True,
            )
        print(
            "FinAuth-Audit public verification: "
            f"{payload['passed']}/{payload['total']} summary checks; "
            f"{payload['internal_passed']}/{payload['internal_total']} attested internal checks"
        )
        return 0
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"public verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
