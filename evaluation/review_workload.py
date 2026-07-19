from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAPACITY_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
REVIEW_COST_BPS = (0.5, 1.0, 2.0, 5.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workload_metrics(frame: pd.DataFrame) -> dict[str, object]:
    review = frame[frame["decision"] == "review"]
    authorized = frame[frame["authorized"]]
    review_count = len(review)
    return {
        "rows": len(frame),
        "review_count": review_count,
        "review_rate": review_count / len(frame),
        "execute_rate": float(frame["decision"].eq("execute").mean()),
        "reduce_rate": float(frame["decision"].eq("reduce").mean()),
        "abstain_rate": float(frame["decision"].eq("abstain").mean()),
        "authorized_utility": float(authorized["selected_utility"].sum()),
        "missed_opportunity": float(frame["missed_opportunity"].sum()),
        "review_if_executed_harm_rate": float((review["full_utility"] < 0).mean()) if review_count else np.nan,
        "review_safe_opportunity_rate": float(
            ((review["best_safe_utility"] > 0) & review["original_source_eligible"]).mean()
        )
        if review_count
        else np.nan,
        "review_deflected_harm_count": int((review["full_utility"] < 0).sum()),
        "reviewed_opportunity_value": float(review["best_safe_utility"].sum()),
    }


def cost_adjusted_utility(authorized_utility: float, review_count: int, cost_bps: float) -> float:
    return float(authorized_utility - review_count * cost_bps / 10_000.0)


def _render_report(summary: pd.DataFrame, capacity: pd.DataFrame, costs: pd.DataFrame) -> str:
    lines = [
        "# Review Workload and Cost Diagnostics",
        "",
        "This validation-only diagnostic treats review as a deferred decision, not as a human study. It reports workload, simulated capacity overload, and review-cost sensitivity without assuming reviewer accuracy.",
        "",
        "## Routing workload",
        "",
        "| Layer | Rule | Review rate | Harm if review rows executed | Safe opportunity in review | Deflected harmful rows |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["layer", "review_rate", "rule"], ascending=[True, False, True]).to_dict(orient="records"):
        harm = "N/A" if pd.isna(row["review_if_executed_harm_rate"]) else f"{row['review_if_executed_harm_rate']:.3f}"
        safe = "N/A" if pd.isna(row["review_safe_opportunity_rate"]) else f"{row['review_safe_opportunity_rate']:.3f}"
        lines.append(
            f"| {row['layer']} | {row['rule']} | {row['review_rate']:.3f} | {harm} | {safe} | "
            f"{int(row['review_deflected_harm_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Capacity overload",
            "",
            "| Layer | Rule | Capacity | Excess review fraction | Queue rows |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in capacity[capacity["excess_review_fraction"] > 0].sort_values(
        ["capacity_fraction", "excess_review_fraction"], ascending=[True, False]
    ).to_dict(orient="records"):
        lines.append(
            f"| {row['layer']} | {row['rule']} | {row['capacity_fraction']:.2f} | "
            f"{row['excess_review_fraction']:.3f} | {int(row['queue_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## Cost sensitivity",
            "",
            "| Layer | Rule | Review cost (bps) | Cost-adjusted authorized utility |",
            "|---|---|---:|---:|",
        ]
    )
    for row in costs.sort_values(["review_cost_bps", "layer", "rule"]).to_dict(orient="records"):
        lines.append(
            f"| {row['layer']} | {row['rule']} | {row['review_cost_bps']:.1f} | "
            f"{row['cost_adjusted_authorized_utility']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Review-cost scenarios are normalized basis-point diagnostics. They are not observed staffing costs, reviewer times, human accuracy estimates, or institutional utility. A future human workflow study must measure those quantities directly.",
            "",
        ]
    )
    return "\n".join(lines)


def run(controlled_decisions: Path, provenance_decisions: Path) -> Path:
    sources = {
        "controlled": controlled_decisions.resolve(),
        "provenance": provenance_decisions.resolve(),
    }
    summary_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for layer, path in sources.items():
        frame = pd.read_csv(path, low_memory=False)
        if set(frame["split"].unique()) != {"validation"}:
            raise RuntimeError(f"{layer} review workload must remain validation-only")
        for rule, group in frame.groupby("rule", sort=True):
            metrics = workload_metrics(group)
            summary_rows.append({"layer": layer, "rule": rule, **metrics})
            for capacity_fraction in CAPACITY_FRACTIONS:
                excess = max(float(metrics["review_rate"]) - capacity_fraction, 0.0)
                capacity_rows.append(
                    {
                        "layer": layer,
                        "rule": rule,
                        "capacity_fraction": capacity_fraction,
                        "excess_review_fraction": excess,
                        "queue_rows": int(np.ceil(excess * int(metrics["rows"]))),
                    }
                )
            for cost_bps in REVIEW_COST_BPS:
                cost_rows.append(
                    {
                        "layer": layer,
                        "rule": rule,
                        "review_cost_bps": cost_bps,
                        "review_count": int(metrics["review_count"]),
                        "authorized_utility": float(metrics["authorized_utility"]),
                        "cost_adjusted_authorized_utility": cost_adjusted_utility(
                            float(metrics["authorized_utility"]),
                            int(metrics["review_count"]),
                            cost_bps,
                        ),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    capacity = pd.DataFrame(capacity_rows)
    costs = pd.DataFrame(cost_rows)
    output_dir = ROOT / "results" / "review_workload"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary.csv": summary,
        "capacity_sensitivity.csv": capacity,
        "cost_sensitivity.csv": costs,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    report_path = output_dir / "report.md"
    report_path.write_text(_render_report(summary, capacity, costs), encoding="utf-8")
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0-round75-review-diagnostic",
        "evaluation_split": "validation",
        "test_outcomes_evaluated": False,
        "capacity_fractions": list(CAPACITY_FRACTIONS),
        "review_cost_bps": list(REVIEW_COST_BPS),
        "inputs": {layer: sha256(path) for layer, path in sources.items()},
        "outputs": {
            **{name: sha256(output_dir / name) for name in outputs},
            "report.md": sha256(report_path),
        },
        "claim_boundary": "Simulated validation-only review workload; not a human study, staffing estimate, or deployment claim.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit validation-only review workload and cost sensitivity.")
    parser.add_argument(
        "--controlled-decisions",
        default=str(ROOT / "results" / "main" / "decisions.csv"),
    )
    parser.add_argument(
        "--provenance-decisions",
        default=str(ROOT / "results" / "provenance_main" / "decisions.csv"),
    )
    args = parser.parse_args()
    print(run(Path(args.controlled_decisions), Path(args.provenance_decisions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

