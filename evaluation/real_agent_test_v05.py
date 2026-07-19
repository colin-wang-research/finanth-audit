from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from finauth_audit.baselines.rules import phase1_rules
from finauth_audit.evaluation.cluster_bootstrap import cluster_bootstrap_bounds
from finauth_audit.evaluation.metrics import profile_metrics, with_decision_outcomes
from finauth_audit.evaluation.seeds import derive_seed
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    impact_bps,
    load_config,
    resolve_root_path,
    sha256,
    write_json,
)


def _timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _verify_freeze(freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_REAL_AGENT_TEST":
        raise RuntimeError("real-agent v0.5 freeze manifest is not active")
    if freeze.get("paper_test_outcomes_evaluated") is not False:
        raise RuntimeError("real-agent freeze records prior paper-test execution")
    if freeze.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("real-agent freeze records hidden-set access")
    for relative, expected in freeze.get("surface_hashes", {}).items():
        path = resolve_root_path(relative)
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"frozen real-agent surface changed: {relative}")
    for relative, expected in freeze.get("dataset_hashes", {}).items():
        path = resolve_root_path(relative)
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"frozen real-agent dataset changed: {relative}")
    return freeze


def materialize_agent_outcomes(
    proposals: pd.DataFrame,
    sealed_outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = proposals.merge(
        sealed_outcomes.drop(columns=["split"], errors="ignore"),
        on=["context_id", "event_cluster_id"],
        how="left",
        validate="many_to_one",
    )
    required = [
        "entry_price",
        "exit_price",
        "entry_bid_depth_1pct",
        "entry_ask_depth_1pct",
        "exit_bid_depth_1pct",
        "exit_ask_depth_1pct",
    ]
    if frame[required].isna().any().any():
        raise RuntimeError("sealed real-agent outcome inputs are incomplete")
    direction = frame["candidate_action"].to_numpy(dtype=float)
    entry = frame["entry_price"].to_numpy(dtype=float)
    exit_price = frame["exit_price"].to_numpy(dtype=float)
    gross = direction * (exit_price / entry - 1.0) * 10000.0
    source = config["binance"]
    notional = float(source["action_notional_usd"])
    slope = float(source["impact_slope_bps"])
    cap = float(source["impact_cap_bps"])
    entry_depth = np.where(
        direction > 0,
        frame["entry_ask_depth_1pct"],
        frame["entry_bid_depth_1pct"],
    ).astype(float)
    exit_depth = np.where(
        direction > 0,
        frame["exit_bid_depth_1pct"],
        frame["exit_ask_depth_1pct"],
    ).astype(float)
    entry_impact = np.asarray(
        [impact_bps(notional, value, slope, cap) for value in entry_depth],
        dtype=float,
    )
    exit_impact = np.asarray(
        [impact_bps(notional, value, slope, cap) for value in exit_depth],
        dtype=float,
    )
    reduced_entry = np.asarray(
        [impact_bps(notional * 0.5, value, slope, cap) for value in entry_depth],
        dtype=float,
    )
    reduced_exit = np.asarray(
        [impact_bps(notional * 0.5, value, slope, cap) for value in exit_depth],
        dtype=float,
    )
    fee = frame["fee_bps"].to_numpy(dtype=float)
    active = direction != 0
    full_utility = np.where(active, gross - fee - entry_impact - exit_impact, 0.0)
    reduced_utility = np.where(
        active,
        0.5 * (gross - fee - reduced_entry - reduced_exit),
        0.0,
    )
    frame["row_id"] = frame["model_id"].astype(str) + ":" + frame["context_id"].astype(str)
    frame["prior_family"] = frame["model_id"].astype(str)
    frame["gross_return_bps"] = gross
    frame["realized_return"] = gross
    frame["realized_entry_impact_bps"] = np.where(active, entry_impact, 0.0)
    frame["realized_exit_impact_bps"] = np.where(active, exit_impact, 0.0)
    frame["full_utility"] = full_utility
    frame["reduced_utility"] = reduced_utility
    frame["harm_label"] = active & (full_utility < 0.0)
    frame["tail_loss"] = np.minimum(full_utility, 0.0)
    high_cost = frame["liquidity_cost_bps"] > float(config["thresholds"]["risk_cost_bps"])
    high_volatility = frame["volatility_proxy"] > float(
        config["thresholds"]["risk_volatility"]
    )
    frame["stress_tag"] = np.select(
        [high_cost, high_volatility],
        ["high_execution_friction", "high_volatility"],
        default="ordinary",
    )
    return frame


def _registered_rules(config: dict[str, Any]):
    selected = set(str(value) for value in config["evaluation"]["rules"])
    rules = [rule for rule in phase1_rules() if rule.name in selected]
    observed = {rule.name for rule in rules}
    if observed != selected:
        raise RuntimeError(f"registered real-agent rules are unavailable: {sorted(selected-observed)}")
    return rules


def _profiles(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall": frame,
        "ordinary": frame[frame["stress_tag"] == "ordinary"],
        "stress": frame[frame["stress_tag"] != "ordinary"],
    }


def evaluate_agent_frame(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    decisions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    bounds: list[dict[str, object]] = []
    model_metrics: list[dict[str, object]] = []
    model_bounds: list[dict[str, object]] = []
    abstain_mask = frame["candidate_action"].eq(0).to_numpy()
    for rule_index, rule in enumerate(_registered_rules(config)):
        decision = rule.decide(frame, config["thresholds"])
        decision = np.asarray(decision, dtype=object)
        decision[abstain_mask] = "abstain"
        ruled = with_decision_outcomes(frame, decision)
        ruled.insert(0, "rule", rule.name)
        decisions.append(ruled)
        for profile_name, profile in _profiles(ruled).items():
            if profile.empty:
                continue
            point = profile_metrics(profile)
            point.update({"rule": rule.name, "profile": profile_name})
            metrics.append(point)
            interval = cluster_bootstrap_bounds(
                profile,
                replicates=int(config["bootstrap_replicates"]),
                seed=derive_seed(
                    int(config["bootstrap_seed"]),
                    f"real-agent/{rule_index}/{rule.name}/{profile_name}",
                ),
                chunk_size=int(config["bootstrap_chunk_size"]),
            )
            interval.update({"rule": rule.name, "profile": profile_name})
            bounds.append(interval)
        for model_index, (model_id, model_frame) in enumerate(ruled.groupby("model_id")):
            point = profile_metrics(model_frame)
            point.update({"rule": rule.name, "model_id": model_id})
            model_metrics.append(point)
            interval = cluster_bootstrap_bounds(
                model_frame,
                replicates=int(config["bootstrap_replicates"]),
                seed=derive_seed(
                    int(config["bootstrap_seed"]),
                    f"real-agent/model/{rule_index}/{model_index}/{rule.name}/{model_id}",
                ),
                chunk_size=int(config["bootstrap_chunk_size"]),
            )
            interval.update({"rule": rule.name, "model_id": model_id})
            model_bounds.append(interval)
    return {
        "decisions": pd.concat(decisions, ignore_index=True),
        "rule_metrics": pd.DataFrame(metrics),
        "bootstrap_bounds": pd.DataFrame(bounds),
        "model_rule_metrics": pd.DataFrame(model_metrics),
        "model_bootstrap_bounds": pd.DataFrame(model_bounds),
    }


def _model_source_quality(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    threshold = float(config["evaluation"]["overconfidence_confidence_threshold"])
    rows: list[dict[str, object]] = []
    for model_id, current in frame.groupby("model_id"):
        proposed = current["candidate_action"].ne(0)
        high_confidence = proposed & current["confidence"].ge(threshold)
        harmful_high_confidence = high_confidence & current["full_utility"].lt(0)
        rows.append(
            {
                "model_id": model_id,
                "contexts": len(current),
                "raw_schema_validity": float(current["raw_schema_valid"].mean()),
                "repairs_applied": 0,
                "model_abstain_rate": float((~proposed).mean()),
                "mean_confidence": float(current["confidence"].mean()),
                "mean_uncertainty": float(current["uncertainty"].mean()),
                "review_recommended_rate": float(current["review_recommended"].mean()),
                "overconfidence_threshold": threshold,
                "high_confidence_nonabstain_count": int(high_confidence.sum()),
                "harmful_high_confidence_count": int(harmful_high_confidence.sum()),
                "overconfidence_rate": (
                    float(harmful_high_confidence.sum() / high_confidence.sum())
                    if high_confidence.any()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)


def _rank_statistic(left: pd.Series, right: pd.Series) -> float | None:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    result = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return float(result.statistic) if np.isfinite(result.statistic) else None


def _ranking_transfer(agent_metrics: pd.DataFrame) -> dict[str, object]:
    controlled_path = ROOT / "results" / "paper_test" / "controlled" / "metrics.csv"
    controlled = pd.read_csv(controlled_path)
    controlled = controlled[controlled["profile"] == "overall"].set_index("rule")
    agent = agent_metrics[agent_metrics["profile"] == "overall"].set_index("rule")
    common = sorted(set(controlled.index) & set(agent.index))
    result: dict[str, object] = {
        "rules_compared": common,
        "classification": "descriptive prospective transfer",
    }
    for metric in ("far", "coverage", "cau", "moc"):
        result[f"{metric}_spearman_rho"] = _rank_statistic(
            controlled.loc[common, metric], agent.loc[common, metric]
        )
    result["claim_boundary"] = (
        "Rule-rank transfer across one controlled paper test and one bounded real-agent task; "
        "it is not a model ranking or evidence of institutional deployment validity."
    )
    return result


def _write_results(
    output_dir: Path,
    result: dict[str, pd.DataFrame],
    source_quality: pd.DataFrame,
    ranking: dict[str, object],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    frames = {
        "rule_metrics.csv": result["rule_metrics"],
        "bootstrap_bounds.csv": result["bootstrap_bounds"],
        "model_rule_metrics.csv": result["model_rule_metrics"],
        "model_bootstrap_bounds.csv": result["model_bootstrap_bounds"],
        "model_source_quality.csv": source_quality,
        "decisions.csv": result["decisions"][
            [
                "rule",
                "row_id",
                "event_cluster_id",
                "model_id",
                "action",
                "decision",
                "authorized",
                "selected_harm",
                "selected_utility",
                "confidence",
                "uncertainty",
                "stress_tag",
            ]
        ],
    }
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs[name] = sha256(path)
    ranking_path = output_dir / "ranking_transfer.json"
    write_json(ranking_path, ranking)
    outputs[ranking_path.name] = sha256(ranking_path)
    overall = result["rule_metrics"]
    overall = overall[overall["profile"] == "overall"].copy()
    summary_path = output_dir / "summary.md"
    lines = [
        "# Real Agent Proposal Paper Test v0.5",
        "",
        "This one-time result evaluates cached model proposals as weak financial priors. ",
        "It is not a model leaderboard, trading-profitability claim, or institutional deployment study.",
        "",
        "| Rule | Coverage | FAR | CAU | MOC | Review rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        far = "N/A" if pd.isna(row.far) else f"{row.far:.3f}"
        lines.append(
            f"| {row.rule} | {row.coverage:.3f} | {far} | {row.cau:.3f} | "
            f"{row.moc:.3f} | {row.review_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "Community-hidden contexts were not sent to a model and their outcomes remain unevaluated.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs[summary_path.name] = sha256(summary_path)
    return outputs


def execute_once(config_path: Path, freeze_path: Path) -> Path:
    config_path = config_path.resolve()
    freeze_path = freeze_path.resolve()
    config = load_config(config_path)
    _verify_freeze(freeze_path)
    registry_path = resolve_root_path(config["freeze"]["test_registry"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    started = {
        "project": "FinAuth-Audit",
        "version": "0.5.0",
        "status": "STARTED",
        "started_at": _timestamp(),
        "freeze_manifest": str(freeze_path.relative_to(ROOT)),
        "freeze_manifest_sha256": sha256(freeze_path),
        "community_hidden_outcomes_evaluated": False,
    }
    try:
        with registry_path.open("x", encoding="utf-8") as handle:
            json.dump(started, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError("real-agent v0.5 test registry exists; rerun prohibited") from exc

    try:
        source = config["binance"]
        proposal_manifest = json.loads(
            resolve_root_path(source["proposal_manifest"]).read_text(encoding="utf-8")
        )
        proposals = pd.read_csv(ROOT / proposal_manifest["proposal_file"])
        proposals = proposals[proposals["split"] == "paper_test"].copy()
        sealed = pd.read_parquet(
            resolve_root_path(source["derived_dir"]) / "paper_test_outcomes.parquet"
        )
        frame = materialize_agent_outcomes(proposals, sealed, config)
        result = evaluate_agent_frame(frame, config)
        source_quality = _model_source_quality(frame, config)
        ranking = _ranking_transfer(result["rule_metrics"])
        output_dir = resolve_root_path(config["results_dir"]) / "paper_test"
        output_hashes = _write_results(output_dir, result, source_quality, ranking)
        manifest = {
            **started,
            "status": "COMPLETED",
            "completed_at": _timestamp(),
            "paper_test_outcomes_evaluated": True,
            "community_hidden_outcomes_evaluated": False,
            "models": sorted(frame["model_id"].unique().tolist()),
            "paper_test_clusters": int(frame["event_cluster_id"].nunique()),
            "proposal_rows": len(frame),
            "outputs": output_hashes,
            "ranking_transfer": ranking,
            "claim_boundary": config["claim_boundary"],
        }
        write_json(registry_path, manifest)
        os.chmod(registry_path, 0o444)
        print(registry_path)
        return registry_path
    except Exception as exc:
        failure = {
            **started,
            "status": "FAILED_RETAINED_NO_RERUN",
            "failed_at": _timestamp(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "paper_test_outcomes_may_have_been_read": True,
            "community_hidden_outcomes_evaluated": False,
        }
        write_json(registry_path, failure)
        os.chmod(registry_path, 0o444)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute the frozen real-agent v0.5 paper test exactly once."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_agent_v05.yaml"))
    parser.add_argument("--execute-frozen-test", action="store_true")
    parser.add_argument("--freeze-manifest")
    args = parser.parse_args()
    if not args.execute_frozen_test or not args.freeze_manifest:
        raise SystemExit("--execute-frozen-test and --freeze-manifest are required")
    execute_once(Path(args.config), Path(args.freeze_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
