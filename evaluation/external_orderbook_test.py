from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from finauth_audit.baselines.rules import phase1_rules
from finauth_audit.evaluation.certification_robustness import (
    continuous_hypervolume,
    pareto_flags,
)
from finauth_audit.evaluation.cluster_bootstrap import cluster_bootstrap_bounds
from finauth_audit.evaluation.holm import holm_adjust
from finauth_audit.evaluation.metrics import profile_metrics, with_decision_outcomes
from finauth_audit.evaluation.seeds import derive_seed
from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    load_config,
    impact_bps,
    resolve_root_path,
    sha256,
    write_json,
)


PAIR = ("Cost-Aware Gate", "Lifecycle Checklist")


def _timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _profiles(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall": frame,
        "stress": frame[frame["stress_tag"] != "ordinary"],
    }


def materialize_outcomes(
    features: pd.DataFrame,
    sealed_outcomes: pd.DataFrame,
    config: dict[str, Any],
    source_name: str,
) -> pd.DataFrame:
    """Materialize labels only after the frozen test gate has been verified."""

    frame = features.merge(
        sealed_outcomes.drop(columns=["split"], errors="ignore"),
        on=["row_id", "event_cluster_id"],
        how="left",
        validate="one_to_one",
    )
    if frame[["entry_price", "exit_price"]].isna().any().any():
        raise RuntimeError(f"sealed outcome inputs are incomplete for {source_name}")
    direction = frame["candidate_action"].to_numpy(dtype=float)
    entry = frame["entry_price"].to_numpy(dtype=float)
    exit_price = frame["exit_price"].to_numpy(dtype=float)
    if source_name == "binance":
        source = config["binance"]
        gross = direction * (exit_price / entry - 1.0) * 10000.0
        action_notional = float(source["action_notional_usd"])
        slope = float(source["impact_slope_bps"])
        cap = float(source["impact_cap_bps"])
        entry_impact = np.asarray(
            [impact_bps(action_notional, value, slope, cap) for value in frame["entry_side_depth_1pct"]],
            dtype=float,
        )
        exit_impact = np.asarray(
            [impact_bps(action_notional, value, slope, cap) for value in frame["exit_side_depth_1pct"]],
            dtype=float,
        )
        reduced_entry_impact = np.asarray(
            [impact_bps(action_notional * 0.5, value, slope, cap) for value in frame["entry_side_depth_1pct"]],
            dtype=float,
        )
        reduced_exit_impact = np.asarray(
            [impact_bps(action_notional * 0.5, value, slope, cap) for value in frame["exit_side_depth_1pct"]],
            dtype=float,
        )
    elif source_name == "databento":
        gross = np.where(
            direction > 0,
            (exit_price / entry - 1.0) * 10000.0,
            (entry / exit_price - 1.0) * 10000.0,
        )
        entry_impact = np.zeros(len(frame), dtype=float)
        exit_impact = np.zeros(len(frame), dtype=float)
        reduced_entry_impact = entry_impact
        reduced_exit_impact = exit_impact
    else:
        raise KeyError(f"unknown external source {source_name}")
    fee = frame["fee_bps"].to_numpy(dtype=float)
    full_utility = gross - fee - entry_impact - exit_impact
    reduced_utility = 0.5 * (
        gross - fee - reduced_entry_impact - reduced_exit_impact
    )
    frame["gross_return_bps"] = gross
    frame["realized_return"] = gross
    frame["realized_entry_impact_bps"] = entry_impact
    frame["realized_exit_impact_bps"] = exit_impact
    frame["full_utility"] = full_utility
    frame["reduced_utility"] = reduced_utility
    frame["harm_label"] = full_utility < 0.0
    frame["tail_loss"] = np.minimum(full_utility, 0.0)
    return frame


def paired_burden_table(decisions: pd.DataFrame) -> pd.DataFrame:
    selected = decisions[decisions["rule"].isin(PAIR)].copy()
    burden = (
        selected.groupby(["rule", "event_cluster_id"], as_index=False)
        .agg(
            rows=("row_id", "size"),
            harmful=("selected_harm", "sum"),
            laundering=("selected_laundering", "sum"),
        )
        .assign(
            false_authorization_burden=lambda value: value["harmful"] / value["rows"],
            laundering_burden=lambda value: value["laundering"] / value["rows"],
        )
    )
    pivot = burden.pivot(
        index="event_cluster_id",
        columns="rule",
        values=["false_authorization_burden", "laundering_burden"],
    )
    pivot.columns = [f"{metric}__{rule}" for metric, rule in pivot.columns]
    pivot = pivot.reset_index()
    pivot["false_authorization_difference"] = (
        pivot[f"false_authorization_burden__{PAIR[0]}"]
        - pivot[f"false_authorization_burden__{PAIR[1]}"]
    )
    pivot["laundering_difference"] = (
        pivot[f"laundering_burden__{PAIR[0]}"]
        - pivot[f"laundering_burden__{PAIR[1]}"]
    )
    return pivot.sort_values("event_cluster_id").reset_index(drop=True)


def paired_bootstrap_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    block_length: int | None = None,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires at least two finite cluster values")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(replicates, dtype=float)
    if block_length is None:
        for index in range(replicates):
            sample = rng.integers(0, n, size=n)
            means[index] = values[sample].mean()
    else:
        block_length = max(1, min(int(block_length), n))
        blocks = int(math.ceil(n / block_length))
        offsets = np.arange(block_length)
        for index in range(replicates):
            starts = rng.integers(0, n, size=blocks)
            sample = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
            means[index] = values[sample].mean()
    return {
        "clusters": n,
        "replicates": replicates,
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
    }


def _primary_endpoints(
    burdens: pd.DataFrame,
    config: dict[str, Any],
    namespace: str,
) -> dict[str, object]:
    replicates = int(config["bootstrap_replicates"])
    base_seed = int(config["bootstrap_seed"])
    false = paired_bootstrap_interval(
        burdens["false_authorization_difference"].to_numpy(),
        replicates=replicates,
        seed=derive_seed(base_seed, f"external/{namespace}/false-burden"),
    )
    laundering = paired_bootstrap_interval(
        burdens["laundering_difference"].to_numpy(),
        replicates=replicates,
        seed=derive_seed(base_seed, f"external/{namespace}/laundering-burden"),
    )
    false_block = paired_bootstrap_interval(
        burdens["false_authorization_difference"].to_numpy(),
        replicates=replicates,
        seed=derive_seed(base_seed, f"external/{namespace}/false-burden-block7"),
        block_length=7,
    )
    laundering_block = paired_bootstrap_interval(
        burdens["laundering_difference"].to_numpy(),
        replicates=replicates,
        seed=derive_seed(base_seed, f"external/{namespace}/laundering-burden-block7"),
        block_length=7,
    )
    false_sesoi = float(
        config["primary_endpoints"]["false_authorization_burden"]["sesoi"]
    )
    laundering_sesoi = float(config["primary_endpoints"]["laundering_burden"]["sesoi"])
    false_pass = bool(false["mean"] <= false_sesoi and false["ci95_upper"] < 0.0)
    laundering_pass = bool(
        laundering["mean"] >= laundering_sesoi and laundering["ci95_lower"] > 0.0
    )
    return {
        "comparison": f"{PAIR[0]} minus {PAIR[1]}",
        "false_authorization_burden": {
            **false,
            "sesoi": false_sesoi,
            "passed": false_pass,
        },
        "laundering_burden": {
            **laundering,
            "sesoi": laundering_sesoi,
            "passed": laundering_pass,
        },
        "moving_block_7day_sensitivity": {
            "false_authorization_burden": false_block,
            "laundering_burden": laundering_block,
        },
        "intersection_union_passed": false_pass and laundering_pass,
    }


def _secondary_prior_family(decisions: pd.DataFrame) -> pd.DataFrame:
    pvalues: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for family in sorted(decisions["prior_family"].unique()):
        family_decisions = decisions[decisions["prior_family"] == family]
        burdens = paired_burden_table(family_decisions)
        for metric in ("false_authorization_difference", "laundering_difference"):
            values = burdens[metric].to_numpy(dtype=float)
            key = f"{family}:{metric}"
            standard_deviation = float(values.std(ddof=1))
            if np.isclose(standard_deviation, 0.0):
                raw_p = 1.0 if np.isclose(values.mean(), 0.0) else 0.0
            else:
                result = ttest_1samp(values, popmean=0.0, nan_policy="raise")
                raw_p = float(result.pvalue) if np.isfinite(result.pvalue) else 1.0
            pvalues[key] = raw_p
            rows.append(
                {
                    "prior_family": family,
                    "metric": metric,
                    "clusters": len(values),
                    "mean_difference": float(values.mean()),
                    "standard_deviation": standard_deviation,
                    "raw_p": raw_p,
                    "holm_adjusted_p": np.nan,
                    "holm_reject": False,
                }
            )
    adjusted = holm_adjust(pvalues)
    for row in rows:
        key = f"{row['prior_family']}:{row['metric']}"
        row["holm_adjusted_p"] = adjusted[key]["holm_adjusted_p"]
        row["holm_reject"] = adjusted[key]["reject"]
    return pd.DataFrame(rows)


def evaluate_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    namespace: str,
) -> dict[str, object]:
    rules = phase1_rules()
    decision_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    bound_rows: list[dict[str, object]] = []
    for rule_index, rule in enumerate(rules):
        ruled = with_decision_outcomes(frame, rule.decide(frame, config["thresholds"]))
        ruled.insert(0, "rule", rule.name)
        decision_frames.append(ruled)
        for profile_name, profile in _profiles(ruled).items():
            if profile.empty:
                continue
            point = profile_metrics(profile)
            point.update({"rule": rule.name, "profile": profile_name})
            metric_rows.append(point)
            bounds = cluster_bootstrap_bounds(
                profile,
                replicates=int(config["bootstrap_replicates"]),
                seed=derive_seed(
                    int(config["bootstrap_seed"]),
                    f"external/{namespace}/{rule_index}/{rule.name}/{profile_name}",
                ),
                chunk_size=int(config["bootstrap_chunk_size"]),
            )
            bounds.update({"rule": rule.name, "profile": profile_name})
            bound_rows.append(bounds)
    decisions = pd.concat(decision_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    bounds = pd.DataFrame(bound_rows)
    overall_bounds = bounds[bounds["profile"] == "overall"].copy()
    overall_bounds["continuous_hypervolume"] = overall_bounds.apply(
        lambda row: continuous_hypervolume(
            float(row["far_ucb95"]) if pd.notna(row["far_ucb95"]) else np.nan,
            float(row["alr_ucb95"]) if pd.notna(row["alr_ucb95"]) else np.nan,
            float(row["coverage_lcb95"]),
        ),
        axis=1,
    )
    pareto = metrics[metrics["profile"] == "overall"][
        ["rule", "coverage", "far", "alr", "cau", "moc", "review_rate"]
    ].copy()
    pareto["pareto_efficient"] = pareto_flags(pareto)
    burdens = paired_burden_table(decisions)
    primary = _primary_endpoints(burdens, config, namespace)
    secondary = _secondary_prior_family(decisions)
    minimal_decisions = decisions[
        [
            "rule",
            "row_id",
            "event_cluster_id",
            "prior_family",
            "source_role",
            "decision",
            "authorized",
            "selected_harm",
            "selected_laundering",
            "selected_utility",
        ]
    ].copy()
    return {
        "decisions": minimal_decisions,
        "metrics": metrics,
        "bounds": bounds,
        "hypervolume": overall_bounds[
            ["rule", "coverage_lcb95", "far_ucb95", "alr_ucb95", "continuous_hypervolume"]
        ],
        "pareto": pareto,
        "burdens": burdens,
        "primary": primary,
        "secondary": secondary,
    }


def _fee_sensitivity(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    original_fee = frame["fee_bps"].astype(float)
    for fee in config["binance"]["fee_sensitivity_bps"]:
        adjusted = frame.copy()
        fee = float(fee)
        delta = original_fee - fee
        adjusted["fee_bps"] = fee
        adjusted["full_utility"] = adjusted["full_utility"] + delta
        adjusted["reduced_utility"] = adjusted["reduced_utility"] + 0.5 * delta
        # The fee grid is secondary and uses 1,000 replicates; co-primary
        # endpoints always use the frozen 10,000-replicate configuration.
        result = evaluate_frame(
            adjusted,
            {**config, "bootstrap_replicates": 1000},
            f"binance-fee-{fee:g}",
        )
        primary = result["primary"]
        rows.append(
            {
                "roundtrip_fee_bps": fee,
                "false_authorization_difference": primary["false_authorization_burden"]["mean"],
                "false_authorization_ci95_lower": primary["false_authorization_burden"]["ci95_lower"],
                "false_authorization_ci95_upper": primary["false_authorization_burden"]["ci95_upper"],
                "laundering_difference": primary["laundering_burden"]["mean"],
                "laundering_ci95_lower": primary["laundering_burden"]["ci95_lower"],
                "laundering_ci95_upper": primary["laundering_burden"]["ci95_upper"],
                "intersection_union_passed": primary["intersection_union_passed"],
            }
        )
    return pd.DataFrame(rows)


def _verify_freeze(freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_EXTERNAL_TEST":
        raise RuntimeError("external freeze manifest is not active")
    if freeze.get("paper_test_outcomes_evaluated") is not False:
        raise RuntimeError("freeze manifest shows prior external-test execution")
    if freeze.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("freeze manifest shows community-hidden access")
    for relative, expected in freeze["surface_hashes"].items():
        path = resolve_root_path(relative)
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"frozen surface changed: {relative}")
    for source_name, outputs in freeze.get("dataset_hashes", {}).items():
        for relative, expected in outputs.items():
            path = resolve_root_path(relative)
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(
                    f"frozen {source_name} dataset changed: {relative}"
                )
    return freeze


def _write_outputs(
    output_dir: Path,
    result: dict[str, object],
    *,
    save_row_level: bool,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "metrics.csv": result["metrics"],
        "bootstrap_bounds.csv": result["bounds"],
        "continuous_hypervolume.csv": result["hypervolume"],
        "pareto_frontier.csv": result["pareto"],
        "prior_family_secondary.csv": result["secondary"],
    }
    if save_row_level:
        frames["decisions.csv"] = result["decisions"]
        frames["paired_cluster_burdens.csv"] = result["burdens"]
    outputs: dict[str, str] = {}
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs[name] = sha256(path)
    primary_path = output_dir / "primary_endpoints.json"
    write_json(primary_path, result["primary"])
    outputs[primary_path.name] = sha256(primary_path)
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
        "version": "0.3.0",
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
        raise RuntimeError("external test registry already exists; rerun is prohibited") from exc
    try:
        results_dir = resolve_root_path(config["results_dir"]) / "paper_test"
        binance_path = resolve_root_path(config["binance"]["derived_dir"]) / "paper_test.csv"
        databento_path = resolve_root_path(config["databento"]["derived_dir"]) / "paper_test.parquet"
        binance_features = pd.read_csv(binance_path, low_memory=False)
        databento_features = pd.read_parquet(databento_path)
        binance_sealed = pd.read_parquet(
            resolve_root_path(config["binance"]["derived_dir"])
            / "paper_test_outcomes.parquet"
        )
        databento_sealed = pd.read_parquet(
            resolve_root_path(config["databento"]["derived_dir"])
            / "paper_test_outcomes.parquet"
        )
        binance = materialize_outcomes(
            binance_features, binance_sealed, config, "binance"
        )
        databento = materialize_outcomes(
            databento_features, databento_sealed, config, "databento"
        )
        binance_result = evaluate_frame(binance, config, "binance")
        databento_result = evaluate_frame(databento, config, "databento")
        output_hashes = {
            "binance": _write_outputs(
                results_dir / "binance", binance_result, save_row_level=True
            ),
            "databento": _write_outputs(
                results_dir / "databento", databento_result, save_row_level=False
            ),
        }
        fee_frame = _fee_sensitivity(binance, config)
        fee_path = results_dir / "binance" / "fee_sensitivity.csv"
        fee_frame.to_csv(fee_path, index=False)
        output_hashes["binance"][fee_path.name] = sha256(fee_path)
        left = binance_result["metrics"]
        right = databento_result["metrics"]
        rank_input = left[left["profile"] == "overall"][["rule", "far"]].merge(
            right[right["profile"] == "overall"][["rule", "far"]],
            on="rule",
            suffixes=("_binance", "_databento"),
        ).dropna()
        rho = (
            float(spearmanr(rank_input["far_binance"], rank_input["far_databento"]).statistic)
            if len(rank_input) >= 3 and rank_input["far_binance"].nunique() > 1 and rank_input["far_databento"].nunique() > 1
            else None
        )
        cross_source = {
            "metric": "aggregate FAR rule-rank correlation",
            "rules_compared": len(rank_input),
            "spearman_rho": rho,
            "classification": "descriptive",
        }
        cross_path = results_dir / "cross_source_rank_correlation.json"
        write_json(cross_path, cross_source)
        output_hashes["cross_source_rank_correlation.json"] = sha256(cross_path)
        manifest = {
            "project": "FinAuth-Audit",
            "version": "0.3.0",
            "status": "COMPLETED",
            "started_at": started["started_at"],
            "completed_at": _timestamp(),
            "freeze_manifest": started["freeze_manifest"],
            "freeze_manifest_sha256": started["freeze_manifest_sha256"],
            "paper_test_outcomes_evaluated": True,
            "community_hidden_outcomes_evaluated": False,
            "binance_classification": "confirmatory",
            "databento_classification": "descriptive_underpowered",
            "binance_primary_result": binance_result["primary"],
            "databento_primary_result": databento_result["primary"],
            "outputs": output_hashes,
            "claim_boundary": (
                "One-time preregistered external test. Binance is the powered public "
                "replication; Databento is descriptive. All null, adverse, or mixed "
                "results are retained. Community-hidden outcomes remain unevaluated."
            ),
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
    parser = argparse.ArgumentParser(description="Execute the frozen v0.3 external test exactly once.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "external_orderbook_v03.yaml"))
    parser.add_argument("--execute-frozen-test", action="store_true")
    parser.add_argument("--freeze-manifest")
    args = parser.parse_args()
    if not args.execute_frozen_test or not args.freeze_manifest:
        raise SystemExit("--execute-frozen-test and --freeze-manifest are required")
    execute_once(Path(args.config), Path(args.freeze_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
