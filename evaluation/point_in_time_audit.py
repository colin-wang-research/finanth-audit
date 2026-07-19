from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for record in frame.fillna("N/A").astype(str).to_dict(orient="records"):
        rows.append(
            "| "
            + " | ".join(str(record[column]).replace("|", "\\|") for column in columns)
            + " |"
        )
    return "\n".join(rows)


def _legacy_rows(name: str, config: dict[str, Any]) -> dict[str, object]:
    derived_path = _path(str(config["derived_path"]))
    frame = pd.read_csv(derived_path, low_memory=False)
    builder_text = _path(str(config["builder_path"])).read_text(encoding="utf-8")
    event_column = next(
        (column for column in ("external_event_id", "gdelt_event_id", "fred_observation_date") if column in frame),
        None,
    )
    raw_records = 0
    raw_path = config.get("raw_path")
    if raw_path:
        target = _path(str(raw_path))
        if target.suffix == ".json":
            raw_records = len(json.loads(target.read_text(encoding="utf-8")))
        else:
            raw_records = len(pd.read_csv(target, low_memory=False))
    else:
        raw_dir = _path(str(config["raw_dir"]))
        raw_records = sum(len(pd.read_csv(path)) for path in raw_dir.glob("*.csv"))
    clusters = int(frame[event_column].nunique()) if event_column else 0
    split_leakage = 0
    if event_column and "split" in frame:
        split_leakage = int((frame.groupby(event_column)["split"].nunique() > 1).sum())
    code_flags = {
        "synthetic_hash_probability": "predecision_probability" in builder_text and "hashlib.sha256" in builder_text,
        "repeated_snapshots": "snap{repeat}" in builder_text or "expanded.append" in builder_text,
        "synthetic_realized_outcome": "realized_base =" in builder_text or "realized_return =" in builder_text,
        "non_vintage_fred": "public_graph_csv_not_vintage_aware" in builder_text,
        "index_forced_stress": "index %" in builder_text or "i %" in builder_text,
    }
    status = "descriptive_public_derived"
    if name == "polymarket_repeated":
        status = "excluded_from_inference"
    elif name == "gdelt":
        status = "controlled_information_shock_extension"
    elif name == "fred":
        status = "exploratory_non_vintage"
    return {
        "source": name,
        "raw_records": raw_records,
        "derived_rows": len(frame),
        "event_clusters": clusters,
        "derived_rows_per_raw": len(frame) / raw_records if raw_records else None,
        "derived_rows_per_cluster": len(frame) / clusters if clusters else None,
        "clusters_crossing_splits": split_leakage,
        **code_flags,
        "classification": status,
        "confirmatory_eligible": False,
    }


def audit_point_in_time_frame(
    frame: pd.DataFrame, lifecycle_fraction: float | None = None
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "detail": detail})

    source = pd.to_datetime(frame["source_timestamp"], utc=True, errors="coerce", format="mixed")
    decision = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce", format="mixed")
    action = pd.to_datetime(frame["action_timestamp"], utc=True, errors="coerce", format="mixed")
    outcome = pd.to_datetime(frame["outcome_timestamp"], utc=True, errors="coerce", format="mixed")
    add(
        "all required timestamps parse",
        not pd.concat([source, decision, action, outcome], axis=1).isna().any().any(),
        f"rows={len(frame)}",
    )
    ordered = (source < decision) & (decision < action) & (action < outcome)
    add("source < decision < action < outcome", ordered.all(), f"violations={int((~ordered).sum())}")
    if lifecycle_fraction is not None:
        created = pd.to_datetime(
            frame["market_created_timestamp"], utc=True, errors="coerce", format="mixed"
        )
        scheduled_end = pd.to_datetime(
            frame["scheduled_end_timestamp"], utc=True, errors="coerce", format="mixed"
        )
        expected = created + lifecycle_fraction * (scheduled_end - created)
        deviation_seconds = (decision - expected).abs().dt.total_seconds()
        add(
            "decision follows frozen lifecycle fraction",
            bool((deviation_seconds <= 1.0).all()),
            f"max_deviation_seconds={float(deviation_seconds.max()):.6f}",
        )
    split_counts = frame.groupby("event_cluster_id")["split"].nunique()
    add("no event cluster crosses splits", bool((split_counts == 1).all()), f"violations={int((split_counts > 1).sum())}")
    rows_per_cluster = frame.groupby("event_cluster_id").size()
    add("one row per public event cluster", int(rows_per_cluster.max()) == 1, f"max={int(rows_per_cluster.max())}")
    add(
        "historical probability observed",
        bool(frame["historical_probability_observed"].astype(bool).all()),
        f"observed={int(frame['historical_probability_observed'].astype(bool).sum())}",
    )
    add(
        "historical spread not claimed",
        not bool(frame["historical_spread_observed"].astype(bool).any()) and "spread" not in frame.columns,
        "spread values absent",
    )
    add(
        "historical depth not claimed",
        not bool(frame["historical_depth_observed"].astype(bool).any()) and "depth" not in frame.columns,
        "depth values absent",
    )
    add(
        "resolution text excluded from rules",
        not bool(frame["resolution_text_used_by_rule"].astype(bool).any()),
        "resolution_text_used_by_rule=false",
    )
    add(
        "outcome excluded from rules",
        not bool(frame["outcome_used_by_rule"].astype(bool).any()),
        "outcome_used_by_rule=false",
    )
    allowed_history_sources = {"clob_prices_history", "data_api_trade"}
    observed_history_sources = set(frame["historical_probability_source"].astype(str))
    add(
        "historical source provenance is explicit",
        observed_history_sources <= allowed_history_sources,
        f"sources={sorted(observed_history_sources)}",
    )
    add(
        "series provenance is present",
        bool(frame["series_key"].notna().all())
        and not bool((frame["series_key"].astype(str).str.len() == 0).any()),
        f"distinct_series={frame['series_key'].nunique()}",
    )
    add(
        "test cluster threshold structurally met",
        int(frame.loc[frame["split"] == "test", "event_cluster_id"].nunique()) >= 500,
        f"test_clusters={frame.loc[frame['split'] == 'test', 'event_cluster_id'].nunique()}",
    )
    return pd.DataFrame(checks)


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    legacy = pd.DataFrame(
        [_legacy_rows(name, payload) for name, payload in config["legacy_sources"].items()]
    )
    legacy.to_csv(results_dir / "legacy_source_audit.csv", index=False)

    dataset_path = ROOT / config["polymarket"]["dataset_path"]
    frame = pd.read_csv(dataset_path, low_memory=False)
    checks = audit_point_in_time_frame(
        frame,
        lifecycle_fraction=float(config["polymarket"]["decision_lifecycle_fraction"]),
    )
    checks.to_csv(results_dir / "point_in_time_checks.csv", index=False)
    pit_pass = bool(checks["passed"].all())
    classifications = legacy[
        ["source", "classification", "confirmatory_eligible"]
    ].copy()
    classifications = pd.concat(
        [
            classifications,
            pd.DataFrame(
                [
                    {
                        "source": "polymarket_point_in_time",
                        "classification": "confirmatory_candidate" if pit_pass else "excluded_from_inference",
                        "confirmatory_eligible": pit_pass,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    classifications.to_csv(results_dir / "source_classification.csv", index=False)
    fetch_manifest = json.loads(
        (ROOT / config["polymarket"]["fetch_manifest"]).read_text(encoding="utf-8")
    )
    truncated_months = fetch_manifest.get("truncated_window_names", [])
    report = [
        "# Public Point-in-Time Audit",
        "",
        "The inferential unit is the raw event cluster. Derived rows are not treated as independent.",
        "",
        "## Classification",
        "",
        _markdown_table(classifications),
        "",
        "## Point-in-time checks",
        "",
        _markdown_table(checks),
        "",
        "## Sampling truncation",
        "",
        (
            f"The 20-page monthly cap was reached in {len(truncated_months)} windows: "
            + ", ".join(truncated_months)
            + ". These windows are sampled, not exhaustive."
        ),
        "",
        "## Boundary",
        "",
        "The legacy Polymarket, GDELT, and FRED layers remain descriptive or exploratory. "
        "Only the new historical-probability Polymarket layer can become confirmatory, and "
        "only after the independent power gate passes. This is offline replay, not deployment evidence.",
        "",
    ]
    (results_dir / "point_in_time_report.md").write_text("\n".join(report), encoding="utf-8")
    print(results_dir)
    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit public point-in-time identifiability.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "public_audit.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
