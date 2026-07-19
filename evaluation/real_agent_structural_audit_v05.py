from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)
from finauth_audit.generators.generate_real_agent_proposals_v05 import PROMPT_FIELDS


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    source = config["binance"]
    derived = resolve_root_path(source["derived_dir"])
    dataset_manifest_path = resolve_root_path(source["dataset_manifest"])
    proposal_manifest_path = resolve_root_path(source["proposal_manifest"])
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    proposals_meta = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    contexts = pd.concat(
        [
            pd.read_csv(derived / "development_contexts.csv"),
            pd.read_csv(derived / "paper_test_contexts.csv"),
            pd.read_csv(derived / "community_hidden_contexts.csv"),
        ],
        ignore_index=True,
    )
    proposals = pd.read_csv(ROOT / proposals_meta["proposal_file"])
    feature_access = json.loads(
        resolve_root_path(config["freeze"]["feature_access"]).read_text(encoding="utf-8")
    )
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("dataset pre-result boundary", dataset.get("outcome_metrics_computed") is False, str(dataset.get("outcome_metrics_computed")))
    check("proposal pre-result boundary", proposals_meta.get("outcome_fields_read") is False, str(proposals_meta.get("outcome_fields_read")))
    check("nonoverlap with v0.3", dataset.get("calendar_overlap_with_v03") is False, str(dataset.get("calendar_overlap_with_v03")))
    check("100 independent contexts", len(contexts) == contexts["event_cluster_id"].nunique() == 100, f"rows={len(contexts)} clusters={contexts['event_cluster_id'].nunique()}")
    expected_splits = {"development": 20, "paper_test": 60, "community_hidden": 20}
    observed_splits = contexts["split"].value_counts().to_dict()
    check("chronological split counts", observed_splits == expected_splits, str(observed_splits))
    ordered = contexts.sort_values("event_cluster_id").reset_index(drop=True)
    expected_order = ["development"] * 20 + ["paper_test"] * 60 + ["community_hidden"] * 20
    check("chronological split membership", ordered["split"].tolist() == expected_order, "positions 1-20/21-80/81-100")
    source_time = pd.to_datetime(contexts["source_timestamp"], utc=True)
    decision_time = pd.to_datetime(contexts["decision_timestamp"], utc=True)
    action_time = pd.to_datetime(contexts["action_timestamp"], utc=True)
    check("source before decision", bool((source_time < decision_time).all()), "source_timestamp < decision_timestamp")
    check("decision before action", bool((decision_time < action_time).all()), "decision_timestamp < action_timestamp")
    forbidden = set(feature_access["global_forbidden"])
    check("forbidden fields absent from contexts", not bool(forbidden & set(contexts.columns)), str(sorted(forbidden & set(contexts.columns))))
    check("prompt fields present", set(PROMPT_FIELDS).issubset(contexts.columns), str(sorted(set(PROMPT_FIELDS)-set(contexts.columns))))
    model_ids = [str(value["model_id"]) for value in config["models"]]
    check("registered model set", sorted(proposals["model_id"].unique()) == sorted(model_ids), str(sorted(proposals["model_id"].unique())))
    expected_proposal_rows = (20 + 60) * len(model_ids)
    check("proposal row count", len(proposals) == expected_proposal_rows, f"{len(proposals)} expected={expected_proposal_rows}")
    check("no hidden proposal generation", not proposals["split"].eq("community_hidden").any(), str(proposals["split"].value_counts().to_dict()))
    expected_ids = set(contexts.loc[contexts["split"].isin(["development", "paper_test"]), "context_id"])
    exact = all(set(group["context_id"]) == expected_ids for _, group in proposals.groupby("model_id"))
    check("one proposal per model and context", exact and not proposals.duplicated(["model_id", "context_id"]).any(), "exact context set per model")
    check("valid actions", set(proposals["action"]).issubset({"long", "short", "abstain"}), str(sorted(proposals["action"].unique())))
    check("bounded confidence", bool(proposals["confidence"].between(0, 1).all()), "[0,1]")
    check("bounded uncertainty", bool(proposals["uncertainty"].between(0, 1).all()), "[0,1]")
    check("structured output validity", bool(proposals["raw_schema_valid"].all()), f"validity={proposals['raw_schema_valid'].mean():.3f}")
    check("no rationale execution evidence", bool(proposals["rationale_not_execution_evidence"].all()), "metadata only")
    check("uniform reasoning effort", proposals["reasoning_effort"].nunique() == 1 and proposals["reasoning_effort"].iloc[0] == "low", str(proposals["reasoning_effort"].value_counts().to_dict()))
    check("proposal hashes present", bool(proposals[["prompt_hash", "context_hash", "output_hash"]].notna().all().all()), "prompt/context/output")
    check("sealed outcomes exist", all((derived / f"{split}_outcomes.parquet").is_file() for split in expected_splits), "content not read")
    passed = all(bool(record["passed"]) for record in checks)
    output = resolve_root_path(config["freeze"]["structural_audit"])
    write_json(
        output,
        {
            "project": "FinAuth-Audit",
            "version": "0.5.0",
            "passed": passed,
            "checks": checks,
            "outcome_metrics_computed": False,
            "outcome_files_read": False,
            "inputs": {
                str(config_path.relative_to(ROOT)): sha256(config_path),
                str(dataset_manifest_path.relative_to(ROOT)): sha256(dataset_manifest_path),
                str(proposal_manifest_path.relative_to(ROOT)): sha256(proposal_manifest_path),
            },
            "claim_boundary": "Pre-result structural audit only; sealed outcome file contents were not read.",
        },
    )
    if not passed:
        failed = [record for record in checks if not record["passed"]]
        raise RuntimeError(f"real-agent structural audit failed: {failed}")
    print(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the real-agent v0.5 pre-result surface.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_agent_v05.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

