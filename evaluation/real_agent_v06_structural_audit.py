from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)
from finauth_audit.generators.generate_real_agent_proposals_v06 import (
    PROMPT_FIELDS,
    TASK_ACTIONS,
)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    source = config["binance"]
    derived = resolve_root_path(source["derived_dir"])
    dataset_manifest_path = resolve_root_path(source["dataset_manifest"])
    proposal_manifest_path = resolve_root_path(source["proposal_manifest"])
    design_freeze_path = resolve_root_path(config["freeze"]["design_freeze_manifest"])
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    proposal_meta = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    design_freeze = json.loads(design_freeze_path.read_text(encoding="utf-8"))
    feature_access = json.loads(
        resolve_root_path(config["freeze"]["feature_access"]).read_text(encoding="utf-8")
    )
    registry = pd.read_csv(derived / "split_registry.csv")
    context_frames = []
    prompt_context_columns: set[str] = set()
    for split in ("development", "paper_test", "community_hidden"):
        current = pd.read_csv(derived / f"{split}_contexts.csv")
        prompt_context_columns.update(str(column) for column in current.columns)
        current = current.merge(
            registry.loc[
                registry["split"] == split,
                ["context_id", "event_cluster_id", "split", "assigned_source_role"],
            ],
            on=["context_id", "assigned_source_role"],
            how="left",
            validate="one_to_one",
        )
        context_frames.append(current)
    contexts = pd.concat(context_frames, ignore_index=True)
    risk_task = config["tasks"]["risk_limit_increase"]
    contexts["original_source_eligible"] = np.where(
        contexts["task_id"].eq("directional_execution"),
        True,
        contexts["assigned_source_role"].eq(risk_task["eligible_source_role"]),
    )
    proposals = pd.read_csv(ROOT / proposal_meta["proposal_file"])
    hidden_manifest_path = resolve_root_path(config["generation"]["hidden_manifest"])
    hidden = json.loads(hidden_manifest_path.read_text(encoding="utf-8"))
    ciphertext_path = resolve_root_path(config["generation"]["hidden_ciphertext"])
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check(
        "design freeze active",
        design_freeze.get("status") == "FROZEN_BEFORE_SOURCE_ACQUISITION",
        str(design_freeze.get("status")),
    )
    surface_ok = all(
        resolve_root_path(relative).is_file()
        and sha256(resolve_root_path(relative)) == expected
        for relative, expected in design_freeze.get("surface_hashes", {}).items()
    )
    check("design surface unchanged", surface_ok, f"files={len(design_freeze.get('surface_hashes', {}))}")
    check(
        "dataset pre-result boundary",
        dataset.get("outcome_metrics_computed") is False,
        str(dataset.get("outcome_metrics_computed")),
    )
    check(
        "proposal pre-result boundary",
        proposal_meta.get("outcome_fields_read") is False,
        str(proposal_meta.get("outcome_fields_read")),
    )
    check(
        "calendar nonoverlap",
        dataset.get("calendar_overlap_with_v05") is False,
        str(dataset.get("calendar_overlap_with_v05")),
    )
    cluster_count = contexts["event_cluster_id"].nunique()
    task_counts = contexts.groupby("event_cluster_id")["task_id"].nunique()
    check(
        "300 independent dates and two tasks per date",
        cluster_count == 300 and len(contexts) == 600 and bool(task_counts.eq(2).all()),
        f"rows={len(contexts)} clusters={cluster_count} tasks_per_cluster={task_counts.value_counts().to_dict()}",
    )
    expected_context_splits = {"development": 100, "paper_test": 400, "community_hidden": 100}
    observed_context_splits = contexts["split"].value_counts().to_dict()
    check(
        "chronological context split counts",
        observed_context_splits == expected_context_splits,
        str(observed_context_splits),
    )
    cluster_splits = (
        contexts[["event_cluster_id", "split"]]
        .drop_duplicates()
        .sort_values("event_cluster_id")["split"]
        .tolist()
    )
    expected_cluster_splits = ["development"] * 50 + ["paper_test"] * 200 + ["community_hidden"] * 50
    check(
        "chronological cluster membership",
        cluster_splits == expected_cluster_splits,
        "positions 1-50/51-250/251-300",
    )
    symbol_counts = contexts.groupby("event_cluster_id")["symbol"].nunique()
    check("one assigned symbol per date", bool(symbol_counts.eq(1).all()), str(symbol_counts.value_counts().to_dict()))
    source_time = pd.to_datetime(contexts["source_timestamp"], utc=True)
    decision_time = pd.to_datetime(contexts["decision_timestamp"], utc=True)
    action_time = pd.to_datetime(contexts["action_timestamp"], utc=True)
    check("source before decision", bool((source_time < decision_time).all()), "source_timestamp < decision_timestamp")
    check("decision before action", bool((decision_time < action_time).all()), "decision_timestamp < action_timestamp")
    forbidden = set(feature_access["global_forbidden"])
    overlap = forbidden.intersection(prompt_context_columns)
    check("forbidden fields absent from contexts", not overlap, str(sorted(overlap)))
    check(
        "prompt fields present",
        set(PROMPT_FIELDS).issubset(contexts.columns),
        str(sorted(set(PROMPT_FIELDS).difference(contexts.columns))),
    )
    role_pairs = set(
        contexts.loc[contexts["task_id"] == "risk_limit_increase", ["assigned_source_role", "original_source_eligible"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    normalized_role_pairs = {(str(role), str(eligible).lower()) for role, eligible in role_pairs}
    check(
        "risk-limit declared roles and eligibility",
        normalized_role_pairs
        == {("risk_limit_proposer", "true"), ("risk_critic", "false")},
        str(sorted(normalized_role_pairs)),
    )

    model_ids = sorted(str(value["model_id"]) for value in config["models"])
    check("registered model set", sorted(proposals["model_id"].unique()) == model_ids, str(sorted(proposals["model_id"].unique())))
    check("plaintext proposal row count", len(proposals) == 1500, f"{len(proposals)} expected=1500")
    check(
        "plaintext splits exclude hidden",
        proposals["split"].value_counts().to_dict() == {"paper_test": 1200, "development": 300},
        str(proposals["split"].value_counts().to_dict()),
    )
    expected_ids = set(contexts.loc[contexts["split"].isin(["development", "paper_test"]), "context_id"])
    exact = all(set(group["context_id"]) == expected_ids for _, group in proposals.groupby("model_id"))
    check(
        "one proposal per model and plaintext context",
        exact and not proposals.duplicated(["model_id", "context_id"]).any(),
        "exact development and paper-test context set per model",
    )
    valid_actions = all(
        set(group["action"]).issubset(TASK_ACTIONS[str(task_id)])
        for task_id, group in proposals.groupby("task_id")
    )
    check("task-valid actions", valid_actions, str(proposals.groupby("task_id")["action"].unique().to_dict()))
    check("bounded confidence", bool(proposals["confidence"].between(0, 1).all()), "[0,1]")
    check("bounded uncertainty", bool(proposals["uncertainty"].between(0, 1).all()), "[0,1]")
    check(
        "raw validity and malformed columns present",
        {
            "raw_schema_valid",
            "malformed_placeholder",
            "malformed_reason",
            "parse_status",
        }.issubset(proposals.columns),
        f"validity={proposals['raw_schema_valid'].mean():.3f}",
    )
    malformed = proposals["malformed_placeholder"].astype(bool)
    malformed_safe = (
        proposals.loc[malformed, "action"].eq("abstain").all()
        and proposals.loc[malformed, "candidate_action"].eq(0).all()
        and proposals.loc[malformed, "confidence"].eq(0.0).all()
        and proposals.loc[malformed, "uncertainty"].eq(1.0).all()
        and (~proposals.loc[malformed, "raw_schema_valid"].astype(bool)).all()
    )
    check(
        "malformed proposals are deterministic non-actions",
        bool(malformed_safe),
        f"malformed={int(malformed.sum())}",
    )
    check(
        "manifest malformed rates agree",
        abs(
            float(proposal_meta.get("malformed_placeholder_rate", -1.0))
            - float(malformed.mean())
        )
        < 1e-12,
        str(proposal_meta.get("malformed_placeholder_rate")),
    )
    check("no rationale execution evidence", bool(proposals["rationale_not_execution_evidence"].all()), "metadata only")
    check(
        "proposal hashes present",
        bool(proposals[["prompt_hash", "context_hash", "output_hash"]].notna().all().all()),
        "prompt/context/output",
    )
    check(
        "encrypted hidden proposal count",
        hidden.get("proposal_rows") == 300 and hidden.get("clusters") == 50,
        f"rows={hidden.get('proposal_rows')} clusters={hidden.get('clusters')}",
    )
    check(
        "hidden ciphertext hash",
        ciphertext_path.is_file() and sha256(ciphertext_path) == hidden.get("ciphertext_sha256"),
        str(hidden.get("ciphertext_sha256")),
    )
    hidden_dir = ciphertext_path.parent
    hidden_plaintext = [
        path.name for path in hidden_dir.iterdir() if path.is_file() and path != ciphertext_path
    ]
    check("no repository hidden plaintext", not hidden_plaintext, str(hidden_plaintext))
    key_path = Path(str(config["generation"]["hidden_key_path"])).resolve()
    check(
        "hidden key outside repository",
        not key_path.is_relative_to(ROOT.parent.resolve()),
        str(key_path),
    )
    expected_outcomes = [derived / f"{split}_outcomes.parquet" for split in expected_context_splits]
    check("sealed outcome files exist", all(path.is_file() for path in expected_outcomes), "contents not read")

    passed = all(bool(record["passed"]) for record in checks)
    output = resolve_root_path(config["freeze"]["structural_audit"])
    write_json(
        output,
        {
            "project": "FinAuth-Audit",
            "version": "0.6.0",
            "passed": passed,
            "checks": checks,
            "outcome_metrics_computed": False,
            "outcome_files_read": False,
            "hidden_plaintext_read": False,
            "inputs": {
                str(config_path.relative_to(ROOT)): sha256(config_path),
                str(design_freeze_path.relative_to(ROOT)): sha256(design_freeze_path),
                str(dataset_manifest_path.relative_to(ROOT)): sha256(dataset_manifest_path),
                str(proposal_manifest_path.relative_to(ROOT)): sha256(proposal_manifest_path),
                str(hidden_manifest_path.relative_to(ROOT)): sha256(hidden_manifest_path),
            },
            "claim_boundary": (
                "Pre-result structural audit only. Sealed outcome contents and hidden "
                "proposal plaintext were not read."
            ),
        },
    )
    if not passed:
        failed = [record for record in checks if not record["passed"]]
        raise RuntimeError(f"v0.6 structural audit failed: {failed}")
    print(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the prospective v0.6 pre-result surface.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_agent_v06.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
