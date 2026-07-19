from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from finauth_audit.evaluation.real_agent_v06_common import apply_decision
from finauth_audit.evaluation.seeds import derive_seed


AUTHORIZED_DECISIONS = {"execute", "reduce"}
HIDDEN_SPLITS = {"community_hidden", "community-hidden", "hidden"}
RATE_METRICS = {
    "economic_loss_authorization_rate",
    "far",
    "material_harm_authorization_rate",
    "tail_harm_authorization_rate",
    "authority_violation_rate",
}


def assert_no_community_hidden(frame: pd.DataFrame, split_col: str = "split") -> None:
    """Reject any inference frame containing community-hidden rows or fields."""

    forbidden_columns = [
        column for column in frame.columns if "community_hidden_outcome" in column.lower()
    ]
    if forbidden_columns:
        raise ValueError(
            "community-hidden outcome fields are prohibited: "
            f"{sorted(forbidden_columns)}"
        )
    if split_col not in frame.columns:
        return
    splits = {
        str(value).strip().lower().replace("-", "_")
        for value in frame[split_col].dropna().unique()
    }
    if splits & {value.replace("-", "_") for value in HIDDEN_SPLITS}:
        raise ValueError("community-hidden outcomes must remain unevaluated")


def _as_rule_series(
    values: pd.Series | pd.DataFrame | Mapping[str, float],
    *,
    rule_col: str = "rule",
    score_col: str = "score",
) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        missing = {rule_col, score_col}.difference(values.columns)
        if missing:
            raise ValueError(f"rank input is missing columns: {sorted(missing)}")
        if values[rule_col].duplicated().any():
            duplicates = values.loc[values[rule_col].duplicated(), rule_col].tolist()
            raise ValueError(f"rank input has duplicate rules: {duplicates}")
        series = values.set_index(rule_col)[score_col]
    elif isinstance(values, pd.Series):
        if values.index.has_duplicates:
            raise ValueError("rank input has duplicate rule index values")
        series = values
    elif isinstance(values, Mapping):
        series = pd.Series(dict(values), dtype=float)
    else:
        raise TypeError("rank input must be a Series, DataFrame, or mapping")
    series = pd.to_numeric(series, errors="coerce")
    series.index = series.index.map(str)
    return series.astype(float)


def _aligned_rule_scores(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    observed: pd.Series | pd.DataFrame | Mapping[str, float],
    *,
    rules: Iterable[str] | None = None,
    controlled_rule_col: str = "rule",
    controlled_score_col: str = "score",
    observed_rule_col: str = "rule",
    observed_score_col: str = "score",
) -> pd.DataFrame:
    left = _as_rule_series(
        controlled,
        rule_col=controlled_rule_col,
        score_col=controlled_score_col,
    ).rename("controlled")
    right = _as_rule_series(
        observed,
        rule_col=observed_rule_col,
        score_col=observed_score_col,
    ).rename("observed")
    if rules is None:
        ordered = sorted(set(left.index) & set(right.index))
        aligned = pd.concat([left.reindex(ordered), right.reindex(ordered)], axis=1)
        return aligned.dropna()
    else:
        ordered = [str(rule) for rule in rules]
        if len(ordered) != len(set(ordered)):
            raise ValueError("rules contains duplicates")
        return pd.concat([left.reindex(ordered), right.reindex(ordered)], axis=1)


def rank_statistics(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    observed: pd.Series | pd.DataFrame | Mapping[str, float],
    *,
    rules: Iterable[str] | None = None,
    controlled_rule_col: str = "rule",
    controlled_score_col: str = "score",
    observed_rule_col: str = "rule",
    observed_score_col: str = "score",
) -> dict[str, object]:
    """Return Spearman and Kendall tau-b with structural N/A semantics.

    Partial ties are valid and are handled by average ranks and tau-b. A fully
    tied/constant ranking is structurally undefined and remains N/A.
    """

    aligned = _aligned_rule_scores(
        controlled,
        observed,
        rules=rules,
        controlled_rule_col=controlled_rule_col,
        controlled_score_col=controlled_score_col,
        observed_rule_col=observed_rule_col,
        observed_score_col=observed_score_col,
    )
    result: dict[str, object] = {
        "valid": False,
        "reason": None,
        "n_rules": int(len(aligned)),
        "rules": aligned.index.tolist(),
        "spearman_rho": None,
        "kendall_tau_b": None,
    }
    if len(aligned) < 3:
        result["reason"] = "fewer_than_three_aligned_rules"
        return result
    if aligned.isna().any().any():
        result["reason"] = "missing_or_na_rule_scores"
        return result
    if aligned["controlled"].nunique() < 2:
        result["reason"] = "controlled_ranking_constant_or_fully_tied"
        return result
    if aligned["observed"].nunique() < 2:
        result["reason"] = "observed_ranking_constant_or_fully_tied"
        return result
    spearman = spearmanr(aligned["controlled"], aligned["observed"])
    kendall = kendalltau(
        aligned["controlled"],
        aligned["observed"],
        variant="b",
        nan_policy="omit",
    )
    rho = float(spearman.statistic)
    tau = float(kendall.statistic)
    if not np.isfinite(rho) or not np.isfinite(tau):
        result["reason"] = "rank_statistic_undefined"
        return result
    result.update(
        {
            "valid": True,
            "reason": None,
            "spearman_rho": rho,
            "kendall_tau_b": tau,
        }
    )
    return result


def exact_one_sided_permutation_test(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    observed: pd.Series | pd.DataFrame | Mapping[str, float],
    *,
    rules: Iterable[str] | None = None,
    controlled_rule_col: str = "rule",
    controlled_score_col: str = "score",
    observed_rule_col: str = "rule",
    observed_score_col: str = "score",
    alternative: str = "inverse_or_more_extreme",
    max_rules: int = 9,
) -> dict[str, object]:
    """Enumerate the exact randomization distribution for inverse transfer."""

    if alternative != "inverse_or_more_extreme":
        raise ValueError(f"unsupported exact-permutation alternative: {alternative}")
    aligned = _aligned_rule_scores(
        controlled,
        observed,
        rules=rules,
        controlled_rule_col=controlled_rule_col,
        controlled_score_col=controlled_score_col,
        observed_rule_col=observed_rule_col,
        observed_score_col=observed_score_col,
    )
    point = rank_statistics(
        aligned["controlled"],
        aligned["observed"],
        rules=aligned.index,
    )
    result: dict[str, object] = {
        "valid": False,
        "reason": point["reason"],
        "alternative": alternative,
        "n_rules": int(len(aligned)),
        "observed_spearman_rho": point["spearman_rho"],
        "permutations": 0,
        "valid_permutations": 0,
        "exact_p_value": None,
    }
    if not point["valid"]:
        return result
    if len(aligned) > max_rules:
        raise ValueError(
            f"exact enumeration is limited to {max_rules} rules, observed {len(aligned)}"
        )
    left = aligned["controlled"].to_numpy(dtype=float)
    right = aligned["observed"].to_numpy(dtype=float)
    observed_rho = float(point["spearman_rho"])
    total = 0
    valid = 0
    inverse_or_more = 0
    for permutation in itertools.permutations(right.tolist()):
        total += 1
        statistic = float(spearmanr(left, permutation).statistic)
        if not np.isfinite(statistic):
            continue
        valid += 1
        if statistic <= observed_rho + 1e-12:
            inverse_or_more += 1
    if valid == 0:
        result["reason"] = "all_permutation_statistics_undefined"
        result["permutations"] = total
        return result
    result.update(
        {
            "valid": True,
            "reason": None,
            "permutations": total,
            "valid_permutations": valid,
            "exact_p_value": float(inverse_or_more / valid),
        }
    )
    return result


def validate_utc_date_clusters(
    frame: pd.DataFrame,
    *,
    cluster_col: str = "event_cluster_id",
    timestamp_col: str = "decision_timestamp",
) -> pd.DataFrame:
    required = {cluster_col, timestamp_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"UTC-date cluster bootstrap is missing fields: {missing}")
    timestamps = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{timestamp_col} contains invalid timestamps")
    mapping = pd.DataFrame(
        {
            cluster_col: frame[cluster_col].astype(str),
            "utc_date": timestamps.dt.normalize(),
        }
    ).drop_duplicates()
    if mapping.groupby(cluster_col)["utc_date"].nunique().ne(1).any():
        raise ValueError("an event cluster spans more than one UTC date")
    if mapping.groupby("utc_date")[cluster_col].nunique().ne(1).any():
        raise ValueError("more than one independent cluster is assigned to a UTC date")
    return mapping.sort_values(["utc_date", cluster_col], kind="mergesort").reset_index(
        drop=True
    )


def _authorized(frame: pd.DataFrame) -> np.ndarray:
    if "authorized" in frame.columns:
        return frame["authorized"].fillna(False).astype(bool).to_numpy()
    if "decision" not in frame.columns:
        raise ValueError("rank metric aggregation requires authorized or decision")
    return frame["decision"].astype(str).isin(AUTHORIZED_DECISIONS).to_numpy()


def _selected_route_label(
    frame: pd.DataFrame,
    *,
    common: str,
    execute: str,
    reduce: str,
) -> np.ndarray:
    if execute in frame.columns and reduce in frame.columns and "decision" in frame.columns:
        decision = frame["decision"].astype(str).to_numpy()
        return np.select(
            [decision == "execute", decision == "reduce"],
            [
                frame[execute].fillna(False).astype(bool).to_numpy(),
                frame[reduce].fillna(False).astype(bool).to_numpy(),
            ],
            default=False,
        ).astype(bool)
    if common not in frame.columns:
        raise ValueError(f"rank metric aggregation is missing {common!r}")
    return frame[common].fillna(False).astype(bool).to_numpy()


def _selected_utility(frame: pd.DataFrame) -> np.ndarray:
    if "selected_normalized_task_utility" in frame.columns:
        return pd.to_numeric(
            frame["selected_normalized_task_utility"], errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
    if "selected_utility" in frame.columns:
        return pd.to_numeric(frame["selected_utility"], errors="coerce").fillna(0).to_numpy()
    required = {
        "decision",
        "full_task_utility",
        "reduced_task_utility",
        "normalized_full_task_utility",
        "normalized_reduced_task_utility",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"rank utility aggregation is missing fields: {missing}")
    decision = frame["decision"].astype(str).to_numpy(dtype=object)
    return apply_decision(frame, decision)["selected_normalized_task_utility"].to_numpy(
        dtype=float
    )


def _metric_components(frame: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    authorized = _authorized(frame)
    if metric == "coverage":
        return authorized.astype(float), np.ones(len(frame), dtype=float)
    if metric in {"economic_loss_authorization_rate", "far"}:
        if "selected_economic_loss" in frame.columns:
            label = frame["selected_economic_loss"].fillna(False).astype(bool).to_numpy()
        elif "selected_harm" in frame.columns:
            label = frame["selected_harm"].fillna(False).astype(bool).to_numpy()
        elif "economic_loss_label" in frame.columns:
            label = frame["economic_loss_label"].fillna(False).astype(bool).to_numpy()
        else:
            label = _selected_utility(frame) < 0
        return (authorized & label).astype(float), authorized.astype(float)
    if metric == "material_harm_authorization_rate":
        if "selected_material_harm" in frame.columns:
            label = frame["selected_material_harm"].fillna(False).astype(bool).to_numpy()
        else:
            label = _selected_route_label(
                frame,
                common="material_harm_label",
                execute="material_harm_full",
                reduce="material_harm_reduced",
            )
        return (authorized & label).astype(float), authorized.astype(float)
    if metric == "tail_harm_authorization_rate":
        if "selected_tail_harm" in frame.columns:
            label = frame["selected_tail_harm"].fillna(False).astype(bool).to_numpy()
        else:
            label = _selected_route_label(
                frame,
                common="tail_harm_label",
                execute="tail_harm_full",
                reduce="tail_harm_reduced",
            )
        return (authorized & label).astype(float), authorized.astype(float)
    if metric == "authority_violation_rate":
        if "authority_violation" in frame.columns:
            label = frame["authority_violation"].fillna(False).astype(bool).to_numpy()
        elif "authority_violation_label" in frame.columns:
            label = frame["authority_violation_label"].fillna(False).astype(bool).to_numpy()
        elif "original_source_eligible" in frame.columns:
            label = ~frame["original_source_eligible"].astype(bool).to_numpy()
        else:
            raise ValueError("authority violation requires a label or source eligibility")
        return (authorized & label).astype(float), authorized.astype(float)
    if metric == "normalized_task_utility":
        return _selected_utility(frame), np.ones(len(frame), dtype=float)
    if metric in frame.columns:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        return np.where(valid, values, 0.0), valid.astype(float)
    raise ValueError(f"unsupported rank-transfer metric: {metric}")


def aggregate_rule_scores(
    frame: pd.DataFrame,
    *,
    metric: str,
    rule_col: str = "rule",
    rules: Iterable[str] | None = None,
) -> pd.Series:
    assert_no_community_hidden(frame)
    if rule_col not in frame.columns:
        raise ValueError(f"rank metric aggregation is missing {rule_col!r}")
    numerator, denominator = _metric_components(frame, metric)
    components = pd.DataFrame(
        {
            rule_col: frame[rule_col].astype(str),
            "numerator": numerator,
            "denominator": denominator,
        }
    ).groupby(rule_col, sort=True)[["numerator", "denominator"]].sum()
    score = components["numerator"].div(components["denominator"].replace(0, np.nan))
    if rules is not None:
        score = score.reindex([str(rule) for rule in rules])
    score.name = metric
    return score


def _cluster_component_arrays(
    frame: pd.DataFrame,
    *,
    metric: str,
    rules: list[str],
    cluster_col: str,
    rule_col: str,
    timestamp_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mapping = validate_utc_date_clusters(
        frame,
        cluster_col=cluster_col,
        timestamp_col=timestamp_col,
    )
    numerator, denominator = _metric_components(frame, metric)
    components = pd.DataFrame(
        {
            cluster_col: frame[cluster_col].astype(str),
            rule_col: frame[rule_col].astype(str),
            "numerator": numerator,
            "denominator": denominator,
        }
    )
    grouped = components.groupby([cluster_col, rule_col], sort=False)[
        ["numerator", "denominator"]
    ].sum()
    cluster_order = mapping[cluster_col].astype(str).tolist()
    index = pd.MultiIndex.from_product(
        [cluster_order, rules], names=[cluster_col, rule_col]
    )
    grouped = grouped.reindex(index, fill_value=0.0)
    n_clusters = len(cluster_order)
    n_rules = len(rules)
    numerators = grouped["numerator"].to_numpy(dtype=float).reshape(n_clusters, n_rules)
    denominators = grouped["denominator"].to_numpy(dtype=float).reshape(n_clusters, n_rules)
    return numerators, denominators, cluster_order


def date_cluster_bootstrap(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    agent_frame: pd.DataFrame,
    *,
    metric: str = "material_harm_authorization_rate",
    rules: Iterable[str] | None = None,
    controlled_rule_col: str = "rule",
    controlled_score_col: str = "score",
    rule_col: str = "rule",
    cluster_col: str = "event_cluster_id",
    timestamp_col: str = "decision_timestamp",
    replicates: int = 10_000,
    seed: int = 0,
    chunk_size: int = 64,
) -> dict[str, object]:
    """Bootstrap whole UTC dates, retaining every within-date repeated measure."""

    assert_no_community_hidden(agent_frame)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    controlled_series = _as_rule_series(
        controlled,
        rule_col=controlled_rule_col,
        score_col=controlled_score_col,
    )
    if rules is None:
        rules_list = sorted(
            set(controlled_series.index) & set(agent_frame[rule_col].astype(str).unique())
        )
    else:
        rules_list = [str(rule) for rule in rules]
    numerators, denominators, clusters = _cluster_component_arrays(
        agent_frame,
        metric=metric,
        rules=rules_list,
        cluster_col=cluster_col,
        rule_col=rule_col,
        timestamp_col=timestamp_col,
    )
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for start in range(0, replicates, chunk_size):
        batch = min(chunk_size, replicates - start)
        sampled = rng.integers(0, len(clusters), size=(batch, len(clusters)))
        batch_num = numerators[sampled].sum(axis=1)
        batch_den = denominators[sampled].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            batch_scores = batch_num / batch_den
        for scores in batch_scores:
            observed = pd.Series(scores, index=rules_list, dtype=float)
            statistic = rank_statistics(
                controlled_series,
                observed,
                rules=rules_list,
            )
            if statistic["valid"]:
                rhos.append(float(statistic["spearman_rho"]))
    finite = np.asarray(rhos, dtype=float)
    result: dict[str, object] = {
        "valid": bool(len(finite)),
        "reason": None if len(finite) else "all_bootstrap_rankings_undefined",
        "clusters": int(len(clusters)),
        "replicates": int(replicates),
        "valid_replicates": int(len(finite)),
        "valid_replicate_fraction": float(len(finite) / replicates),
        "spearman_median": None,
        "spearman_lcb95": None,
        "spearman_ucb95": None,
        "probability_below_zero": None,
    }
    if len(finite):
        result.update(
            {
                "spearman_median": float(np.quantile(finite, 0.50)),
                "spearman_lcb95": float(np.quantile(finite, 0.025)),
                "spearman_ucb95": float(np.quantile(finite, 0.975)),
                "probability_below_zero": float(np.mean(finite < 0.0)),
            }
        )
    return result


def leave_one_rule_out(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    observed: pd.Series | pd.DataFrame | Mapping[str, float],
    *,
    rules: Iterable[str] | None = None,
) -> pd.DataFrame:
    aligned = _aligned_rule_scores(controlled, observed, rules=rules)
    rows: list[dict[str, object]] = []
    for omitted in aligned.index:
        retained = [rule for rule in aligned.index if rule != omitted]
        stats = rank_statistics(
            aligned["controlled"], aligned["observed"], rules=retained
        )
        exact = exact_one_sided_permutation_test(
            aligned["controlled"], aligned["observed"], rules=retained
        )
        rows.append(
            {
                "omitted_rule": omitted,
                "n_rules": stats["n_rules"],
                "valid": stats["valid"],
                "reason": stats["reason"],
                "spearman_rho": stats["spearman_rho"],
                "kendall_tau_b": stats["kendall_tau_b"],
                "exact_permutation_p": exact["exact_p_value"],
            }
        )
    return pd.DataFrame(rows)


def pairwise_reversal(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    observed: pd.Series | pd.DataFrame | Mapping[str, float],
    *,
    rules: Iterable[str] | None = None,
) -> pd.DataFrame:
    aligned = _aligned_rule_scores(controlled, observed, rules=rules)
    rows: list[dict[str, object]] = []
    for rule_a, rule_b in itertools.combinations(aligned.index, 2):
        controlled_delta = float(
            aligned.loc[rule_a, "controlled"] - aligned.loc[rule_b, "controlled"]
        )
        observed_delta = float(
            aligned.loc[rule_a, "observed"] - aligned.loc[rule_b, "observed"]
        )
        missing = not np.isfinite(controlled_delta) or not np.isfinite(observed_delta)
        tied = bool(
            not missing
            and (np.isclose(controlled_delta, 0.0) or np.isclose(observed_delta, 0.0))
        )
        rows.append(
            {
                "rule_a": rule_a,
                "rule_b": rule_b,
                "controlled_delta": controlled_delta,
                "observed_delta": observed_delta,
                "valid": not tied and not missing,
                "reason": (
                    "pair_missing_n_a" if missing else "pair_tied_n_a" if tied else None
                ),
                "reversal": (
                    None
                    if tied or missing
                    else bool(controlled_delta * observed_delta < 0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def subgroup_rank_transfer(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    agent_frame: pd.DataFrame,
    *,
    group_col: str,
    metric: str,
    rules: Iterable[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_chunk_size: int = 64,
) -> pd.DataFrame:
    if group_col not in agent_frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for group_index, (group_value, group) in enumerate(
        agent_frame.groupby(group_col, sort=True, dropna=False)
    ):
        scores = aggregate_rule_scores(group, metric=metric, rules=rules)
        stats = rank_statistics(controlled, scores, rules=rules)
        exact = exact_one_sided_permutation_test(controlled, scores, rules=rules)
        bootstrap = date_cluster_bootstrap(
            controlled,
            group,
            metric=metric,
            rules=rules,
            replicates=bootstrap_replicates,
            seed=derive_seed(
                bootstrap_seed,
                f"real-agent-v06/rank-transfer/{group_col}/{group_index}/{group_value}",
            ),
            chunk_size=bootstrap_chunk_size,
        )
        rows.append(
            {
                group_col: group_value,
                "n_rules": stats["n_rules"],
                "valid": stats["valid"],
                "reason": stats["reason"],
                "spearman_rho": stats["spearman_rho"],
                "kendall_tau_b": stats["kendall_tau_b"],
                "exact_permutation_p": exact["exact_p_value"],
                "bootstrap_spearman_median": bootstrap["spearman_median"],
                "bootstrap_spearman_lcb95": bootstrap["spearman_lcb95"],
                "bootstrap_spearman_ucb95": bootstrap["spearman_ucb95"],
                "probability_below_zero": bootstrap["probability_below_zero"],
                "valid_replicate_fraction": bootstrap["valid_replicate_fraction"],
                "clusters": bootstrap["clusters"],
            }
        )
    return pd.DataFrame(rows)


def robust_rank_transfer(
    controlled: pd.Series | pd.DataFrame | Mapping[str, float],
    agent_frame: pd.DataFrame,
    *,
    metric: str = "material_harm_authorization_rate",
    rules: Iterable[str] | None = None,
    controlled_rule_col: str = "rule",
    controlled_score_col: str = "score",
    task_id: str | None = "directional_execution",
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_chunk_size: int = 64,
) -> dict[str, object]:
    """Run the full preregistered rank-transfer robustness surface."""

    assert_no_community_hidden(agent_frame)
    current = agent_frame.copy()
    if task_id is not None and "task_id" in current.columns:
        current = current[current["task_id"].astype(str) == str(task_id)].copy()
    if current.empty:
        raise ValueError("rank-transfer frame is empty after task filtering")
    controlled_series = _as_rule_series(
        controlled,
        rule_col=controlled_rule_col,
        score_col=controlled_score_col,
    )
    if rules is None:
        rules_list = sorted(
            set(controlled_series.index) & set(current["rule"].astype(str).unique())
        )
    else:
        rules_list = [str(rule) for rule in rules]
    observed = aggregate_rule_scores(current, metric=metric, rules=rules_list)
    point = rank_statistics(controlled_series, observed, rules=rules_list)
    exact = exact_one_sided_permutation_test(
        controlled_series,
        observed,
        rules=rules_list,
    )
    bootstrap = date_cluster_bootstrap(
        controlled_series,
        current,
        metric=metric,
        rules=rules_list,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        chunk_size=bootstrap_chunk_size,
    )
    exact_p = exact["exact_p_value"]
    probability = bootstrap["probability_below_zero"]
    primary_support = (
        None
        if exact_p is None or probability is None
        else bool(float(probability) >= 0.95 and float(exact_p) <= 0.05)
    )
    subgroup_kwargs = {
        "metric": metric,
        "rules": rules_list,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_chunk_size": bootstrap_chunk_size,
    }
    return {
        "metric": metric,
        "rules": rules_list,
        "point": point,
        "exact_permutation": exact,
        "date_cluster_bootstrap": bootstrap,
        "primary_support": primary_support,
        "leave_one_rule_out": leave_one_rule_out(
            controlled_series, observed, rules=rules_list
        ),
        "per_model": subgroup_rank_transfer(
            controlled_series, current, group_col="model_id", **subgroup_kwargs
        ),
        "per_asset": subgroup_rank_transfer(
            controlled_series, current, group_col="symbol", **subgroup_kwargs
        ),
        "per_volatility_regime": subgroup_rank_transfer(
            controlled_series,
            current,
            group_col="volatility_regime",
            **subgroup_kwargs,
        ),
        "pairwise_reversal": pairwise_reversal(
            controlled_series, observed, rules=rules_list
        ),
        "community_hidden_outcomes_evaluated": False,
    }
