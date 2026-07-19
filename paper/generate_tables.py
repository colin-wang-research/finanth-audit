from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "generated"

V06_VERSION = "0.6.0"
V06_TRACKS = ("zero_shot", "recalibrated")
V06_TASK_PROFILES = ("directional_execution", "risk_limit_increase")
V06_TASK_METRICS = {
    "directional_execution": "economic_loss_authorization_rate",
    "risk_limit_increase": "material_risk_event_authorization_rate",
}
V06_RULE_ORDER = (
    "Direct Prior",
    "Confidence Gate",
    "Uncertainty Gate",
    "Risk Filter",
    "Cost-Aware Gate",
    "Hard Role Gate",
    "Lifecycle Checklist",
)
V06_REGISTRY_RELATIVE = Path("manifests/real_agent_v06_test_registry.json")
V06_MANIFEST_RELATIVES = {
    "source": Path("manifests/real_agent_v06/source_manifest.json"),
    "dataset": Path("manifests/real_agent_v06/dataset_manifest.json"),
    "proposal": Path("manifests/real_agent_v06/proposal_manifest.json"),
}
V06_PAPER_TEST_RELATIVE = Path("results/real_agent_v06/paper_test")
V06_HISTORICAL_MEMORY_RELATIVE = Path("results/real_agent_v06/historical_memory_audit")
V06_HISTORICAL_OUTPUTS = (
    "temporal_summary.csv",
    "temporal_action_distribution.csv",
    "temporal_drift_summary.csv",
    "lexical_summary.csv",
)
V06_TRACK_OUTPUTS = ("metrics.csv", "bootstrap_bounds.csv", "subgroup_metrics.csv")


@dataclass(frozen=True)
class V06PaperInputs:
    registry: dict[str, Any]
    source_manifest: dict[str, Any]
    dataset_manifest: dict[str, Any]
    proposal_manifest: dict[str, Any]
    historical_memory_manifest: dict[str, Any]
    metrics: dict[str, pd.DataFrame]
    bootstrap_bounds: dict[str, pd.DataFrame]
    subgroup_metrics: dict[str, pd.DataFrame]
    rank_transfer: dict[str, Any]
    source_quality: pd.DataFrame
    historical_aggregates: dict[str, pd.DataFrame]
    input_paths: tuple[Path, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v06_allowed_relative_paths() -> frozenset[Path]:
    paths = {
        V06_REGISTRY_RELATIVE,
        *V06_MANIFEST_RELATIVES.values(),
        V06_HISTORICAL_MEMORY_RELATIVE / "manifest.json",
        *(V06_HISTORICAL_MEMORY_RELATIVE / name for name in V06_HISTORICAL_OUTPUTS),
        V06_PAPER_TEST_RELATIVE / "rank_transfer_zero_shot.json",
        V06_PAPER_TEST_RELATIVE / "model_source_quality.csv",
    }
    for track in V06_TRACKS:
        paths.update(
            V06_PAPER_TEST_RELATIVE / track / name for name in V06_TRACK_OUTPUTS
        )
    return frozenset(paths)


def _v06_input_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _assert_v06_paper_input(path: Path | str) -> Path:
    candidate = _v06_input_path(path)
    try:
        relative = candidate.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"v0.6 paper input is outside the repository: {candidate}"
        ) from exc
    if relative not in _v06_allowed_relative_paths():
        raise RuntimeError(f"v0.6 paper input is not aggregate-allowlisted: {relative}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _read_v06_json(path: Path | str) -> dict[str, Any]:
    candidate = _assert_v06_paper_input(path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"v0.6 JSON input must be an object: {candidate}")
    return payload


def _read_v06_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(_assert_v06_paper_input(path))


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _require_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"{label} contains duplicate keys: {columns}")


def _validate_completed_v06_registry(registry: dict[str, Any]) -> None:
    if registry.get("status") != "COMPLETED":
        raise RuntimeError("v0.6 paper generation requires a COMPLETED test registry")
    if registry.get("version") != V06_VERSION:
        raise RuntimeError("v0.6 paper-test registry has the wrong version")
    if registry.get("paper_test_outcomes_evaluated") is not True:
        raise RuntimeError(
            "v0.6 registry does not record completed paper-test evaluation"
        )
    if registry.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("community-hidden outcomes must remain unevaluated")
    if registry.get("hidden_proposals_decrypted") is not False:
        raise RuntimeError("community-hidden proposals must remain encrypted")
    if registry.get("paper_test_clusters") != 200:
        raise RuntimeError("v0.6 paper test must use 200 independent UTC-date clusters")
    if (
        not isinstance(registry.get("proposal_rows"), int)
        or registry["proposal_rows"] <= 0
    ):
        raise ValueError("v0.6 registry proposal_rows must be a positive integer")
    if set(registry.get("tasks", [])) != set(V06_TASK_PROFILES):
        raise ValueError("v0.6 registry tasks do not match the two preregistered tasks")
    if not registry.get("models"):
        raise ValueError("v0.6 registry must list evaluated models")
    if not isinstance(registry.get("outputs"), dict):
        raise ValueError("v0.6 registry outputs must be a hash mapping")


def _verify_v06_hash(path: Path, expected: object, label: str) -> None:
    candidate = _assert_v06_paper_input(path)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} is missing a SHA-256 hash")
    if sha256(candidate) != expected:
        raise RuntimeError(f"v0.6 aggregate input hash mismatch: {label}")


def _validate_v06_manifests(
    registry: dict[str, Any],
    source: dict[str, Any],
    dataset: dict[str, Any],
    proposal: dict[str, Any],
    historical: dict[str, Any],
) -> None:
    for label, payload in (
        ("source", source),
        ("dataset", dataset),
        ("proposal", proposal),
        ("historical-memory", historical),
    ):
        if payload.get("version") != V06_VERSION:
            raise ValueError(f"v0.6 {label} manifest has the wrong version")
    if source.get("complete_registered_request") is not True:
        raise RuntimeError("v0.6 source manifest is not complete")
    if source.get("model_proposals_generated") is not False:
        raise RuntimeError("v0.6 source manifest is not source-only")
    if source.get("outcome_metrics_computed") is not False:
        raise RuntimeError("v0.6 source manifest contains outcome computation")
    if dataset.get("independent_cluster") != "utc_date":
        raise ValueError("v0.6 dataset independent unit must be utc_date")
    if dataset.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("v0.6 dataset manifest records hidden-outcome evaluation")
    if dataset.get("outcome_metrics_computed") is not False:
        raise RuntimeError("v0.6 dataset manifest contains outcome metrics")
    if dataset.get("splits", {}).get("paper_test") != registry["paper_test_clusters"]:
        raise ValueError("v0.6 dataset and registry paper-test date counts disagree")
    if proposal.get("outcome_fields_read") is not False:
        raise RuntimeError("v0.6 proposal manifest records outcome-field access")
    if proposal.get("community_hidden_plaintext_in_repository") is not False:
        raise RuntimeError(
            "v0.6 hidden proposal plaintext must not be in the repository"
        )
    if proposal.get("repairs_applied") != 0:
        raise RuntimeError("v0.6 proposal outputs must retain malformed placeholders")
    if set(proposal.get("models", [])) != set(registry["models"]):
        raise ValueError("v0.6 proposal manifest and registry model sets disagree")
    for field in ("raw_schema_validity", "malformed_placeholder_rate"):
        if field not in proposal:
            raise ValueError(f"v0.6 proposal manifest is missing {field}")
    if historical.get("status") != "COMPLETED_BEFORE_ANY_OUTCOME_EVALUATION":
        raise RuntimeError("v0.6 historical-memory aggregate audit is incomplete")
    for field in (
        "development_outcomes_read",
        "paper_test_outcomes_read",
        "community_hidden_outcomes_read",
        "community_hidden_proposals_decrypted",
    ):
        if historical.get(field) is not False:
            raise RuntimeError(f"v0.6 historical-memory audit violates {field}")
    if historical.get("rationale_text_persisted_in_outputs") is not False:
        raise RuntimeError("historical-memory outputs must remain aggregate-only")


def _validate_v06_track_frames(
    track: str,
    metrics: pd.DataFrame,
    bounds: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> None:
    metric_columns = {
        "rule",
        "profile",
        "rows",
        "clusters",
        "coverage",
        "review_rate",
        "economic_loss_authorization_rate",
        "material_harm_authorization_rate",
        "material_risk_event_authorization_rate",
        "authority_violation_rate",
        "normalized_task_utility",
        "track",
    }
    _require_columns(metrics, metric_columns, f"{track} metrics")
    _require_unique(metrics, ["rule", "profile"], f"{track} metrics")
    if set(metrics["track"].astype(str)) != {track}:
        raise ValueError(f"{track} metrics contain a mismatched track label")
    expected_profiles = {"overall", *V06_TASK_PROFILES}
    for rule, current in metrics.groupby("rule", sort=False):
        if set(current["profile"].astype(str)) != expected_profiles:
            raise ValueError(f"{track} metrics are incomplete for rule {rule}")

    bound_columns = {
        "track",
        "rule",
        "profile",
        "metric",
        "point",
        "lcb95",
        "ucb95",
        "valid_replicate_fraction",
        "replicates",
        "clusters",
    }
    _require_columns(bounds, bound_columns, f"{track} bootstrap bounds")
    _require_unique(
        bounds,
        ["rule", "profile", "metric"],
        f"{track} bootstrap bounds",
    )
    if set(bounds["track"].astype(str)) != {track}:
        raise ValueError(f"{track} bootstrap bounds contain a mismatched track label")
    bound_keys = set(
        zip(
            bounds["rule"].astype(str),
            bounds["profile"].astype(str),
            bounds["metric"].astype(str),
            strict=True,
        )
    )
    for row in metrics.itertuples(index=False):
        for metric in ("coverage", V06_TASK_METRICS.get(str(row.profile), "coverage")):
            if (str(row.rule), str(row.profile), metric) not in bound_keys:
                raise ValueError(
                    f"{track} bootstrap bounds are missing {row.rule}/{row.profile}/{metric}"
                )

    subgroup_columns = {
        "rule",
        "dimension",
        "value",
        "rows",
        "clusters",
        "coverage",
        "material_harm_authorization_rate",
        "track",
    }
    _require_columns(subgroups, subgroup_columns, f"{track} subgroup metrics")
    _require_unique(
        subgroups,
        ["rule", "dimension", "value"],
        f"{track} subgroup metrics",
    )
    if set(subgroups["track"].astype(str)) != {track}:
        raise ValueError(f"{track} subgroup metrics contain a mismatched track label")
    expected_dimensions = {"model_id", "symbol", "volatility_regime", "task_id"}
    if not expected_dimensions.issubset(set(subgroups["dimension"].astype(str))):
        raise ValueError(f"{track} subgroup metrics are missing a registered dimension")


def _validate_v06_rank(rank: dict[str, Any], registry: dict[str, Any]) -> None:
    required = {
        "metric",
        "rules",
        "point",
        "exact_permutation",
        "date_cluster_bootstrap",
        "primary_support",
        "community_hidden_outcomes_evaluated",
    }
    missing = required.difference(rank)
    if missing:
        raise ValueError(f"v0.6 rank transfer is missing fields: {sorted(missing)}")
    support = rank["primary_support"]
    if support is not True and support is not False and support is not None:
        raise ValueError("v0.6 primary_support must be true, false, or null")
    if rank.get("community_hidden_outcomes_evaluated") is not False:
        raise RuntimeError("rank transfer must not use community-hidden outcomes")
    point = rank["point"]
    exact = rank["exact_permutation"]
    bootstrap = rank["date_cluster_bootstrap"]
    if not all(isinstance(value, dict) for value in (point, exact, bootstrap)):
        raise ValueError("v0.6 rank-transfer statistics must be JSON objects")
    for field in ("n_rules", "spearman_rho", "kendall_tau_b"):
        if field not in point:
            raise ValueError(f"v0.6 rank point estimate is missing {field}")
    if "exact_p_value" not in exact:
        raise ValueError("v0.6 rank exact-permutation output is incomplete")
    for field in (
        "clusters",
        "spearman_median",
        "spearman_lcb95",
        "spearman_ucb95",
        "probability_below_zero",
        "valid_replicate_fraction",
    ):
        if field not in bootstrap:
            raise ValueError(f"v0.6 rank bootstrap is missing {field}")
    if bootstrap["clusters"] != registry["paper_test_clusters"]:
        raise ValueError("rank-transfer bootstrap does not use all 200 UTC dates")
    if point["n_rules"] != len(rank["rules"]):
        raise ValueError("rank-transfer n_rules does not match its rule list")
    exact_p = exact["exact_p_value"]
    probability = bootstrap["probability_below_zero"]
    derived = (
        None
        if exact_p is None or probability is None
        else bool(float(probability) >= 0.95 and float(exact_p) <= 0.05)
    )
    if support is not derived:
        raise ValueError(
            "rank-transfer primary_support disagrees with registered statistics"
        )


def _validate_v06_source_quality(frame: pd.DataFrame, registry: dict[str, Any]) -> None:
    required = {
        "model_id",
        "task_id",
        "rows",
        "clusters",
        "raw_schema_validity",
        "malformed_placeholder_rate",
        "repairs_applied",
        "proposal_abstain_rate",
        "mean_confidence",
        "review_recommended_rate",
        "material_harm_overconfidence_rate",
    }
    _require_columns(frame, required, "v0.6 model source quality")
    _require_unique(frame, ["model_id", "task_id"], "v0.6 model source quality")
    expected = {
        (str(model), task) for model in registry["models"] for task in V06_TASK_PROFILES
    }
    observed = set(
        zip(
            frame["model_id"].astype(str),
            frame["task_id"].astype(str),
            strict=True,
        )
    )
    if observed != expected:
        raise ValueError("v0.6 model source quality is incomplete")
    if (frame["repairs_applied"] != 0).any():
        raise ValueError("v0.6 source quality must retain malformed placeholders")


def _validate_historical_aggregates(frames: dict[str, pd.DataFrame]) -> None:
    required = {
        "temporal_summary.csv": {
            "calendar_quarter",
            "model_id",
            "task_id",
            "proposals",
            "independent_dates",
            "explicit_retrospective_signal_rate",
            "unsupported_external_source_signal_rate",
        },
        "temporal_action_distribution.csv": {
            "calendar_quarter",
            "model_id",
            "task_id",
            "action",
            "proposals",
            "action_rate",
        },
        "temporal_drift_summary.csv": {
            "model_id",
            "task_id",
            "first_quarter",
            "last_quarter",
            "action_distribution_jensen_shannon_bits",
        },
        "lexical_summary.csv": {
            "scope",
            "proposals",
            "independent_dates",
            "explicit_retrospective_signal_rate",
            "unsupported_external_source_signal_rate",
        },
    }
    for name, columns in required.items():
        _require_columns(frames[name], columns, f"historical-memory {name}")


def load_v06_paper_inputs() -> V06PaperInputs:
    input_paths: list[Path] = []
    registry_path = ROOT / V06_REGISTRY_RELATIVE
    registry = _read_v06_json(registry_path)
    input_paths.append(registry_path)
    _validate_completed_v06_registry(registry)

    manifests = {
        name: _read_v06_json(ROOT / relative)
        for name, relative in V06_MANIFEST_RELATIVES.items()
    }
    input_paths.extend(ROOT / relative for relative in V06_MANIFEST_RELATIVES.values())
    historical_manifest_path = ROOT / V06_HISTORICAL_MEMORY_RELATIVE / "manifest.json"
    historical_manifest = _read_v06_json(historical_manifest_path)
    input_paths.append(historical_manifest_path)
    _validate_v06_manifests(
        registry,
        manifests["source"],
        manifests["dataset"],
        manifests["proposal"],
        historical_manifest,
    )
    metrics: dict[str, pd.DataFrame] = {}
    bounds: dict[str, pd.DataFrame] = {}
    subgroups: dict[str, pd.DataFrame] = {}
    declared_outputs = registry["outputs"]
    for track in V06_TRACKS:
        paths = {
            name: ROOT / V06_PAPER_TEST_RELATIVE / track / name
            for name in V06_TRACK_OUTPUTS
        }
        for name, path in paths.items():
            key = f"{track}/{name}"
            _verify_v06_hash(path, declared_outputs.get(key), key)
            input_paths.append(path)
        metrics[track] = _read_v06_csv(paths["metrics.csv"])
        bounds[track] = _read_v06_csv(paths["bootstrap_bounds.csv"])
        subgroups[track] = _read_v06_csv(paths["subgroup_metrics.csv"])
        _validate_v06_track_frames(
            track, metrics[track], bounds[track], subgroups[track]
        )
    rule_sets = {track: set(metrics[track]["rule"].astype(str)) for track in V06_TRACKS}
    if rule_sets["zero_shot"] != rule_sets["recalibrated"]:
        raise ValueError("v0.6 zero-shot and recalibrated rule sets disagree")

    rank_path = ROOT / V06_PAPER_TEST_RELATIVE / "rank_transfer_zero_shot.json"
    source_quality_path = ROOT / V06_PAPER_TEST_RELATIVE / "model_source_quality.csv"
    _verify_v06_hash(
        rank_path,
        declared_outputs.get("rank_transfer_zero_shot.json"),
        "rank_transfer_zero_shot.json",
    )
    _verify_v06_hash(
        source_quality_path,
        declared_outputs.get("model_source_quality.csv"),
        "model_source_quality.csv",
    )
    input_paths.extend([rank_path, source_quality_path])
    rank = _read_v06_json(rank_path)
    source_quality = _read_v06_csv(source_quality_path)
    _validate_v06_rank(rank, registry)
    if not set(str(rule) for rule in rank["rules"]).issubset(rule_sets["zero_shot"]):
        raise ValueError("v0.6 rank-transfer rules are missing from zero-shot metrics")
    _validate_v06_source_quality(source_quality, registry)

    historical_outputs = historical_manifest.get("outputs")
    if not isinstance(historical_outputs, dict):
        raise ValueError("historical-memory manifest outputs must be a hash mapping")
    historical_aggregates: dict[str, pd.DataFrame] = {}
    for name in V06_HISTORICAL_OUTPUTS:
        path = ROOT / V06_HISTORICAL_MEMORY_RELATIVE / name
        _verify_v06_hash(
            path, historical_outputs.get(name), f"historical-memory/{name}"
        )
        input_paths.append(path)
        historical_aggregates[name] = _read_v06_csv(path)
    _validate_historical_aggregates(historical_aggregates)

    return V06PaperInputs(
        registry=registry,
        source_manifest=manifests["source"],
        dataset_manifest=manifests["dataset"],
        proposal_manifest=manifests["proposal"],
        historical_memory_manifest=historical_manifest,
        metrics=metrics,
        bootstrap_bounds=bounds,
        subgroup_metrics=subgroups,
        rank_transfer=rank,
        source_quality=source_quality,
        historical_aggregates=historical_aggregates,
        input_paths=tuple(input_paths),
    )


def _resolve_v06_inputs(v06: V06PaperInputs | None) -> V06PaperInputs:
    return load_v06_paper_inputs() if v06 is None else v06


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return r"\textit{N/A}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}f}"


def signed(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return r"\textit{N/A}"
    return f"{float(value):+.{digits}f}"


def interval(record: dict[str, object], digits: int = 3) -> str:
    return (
        "["
        + signed(record["ci95_lower"], digits)
        + ", "
        + signed(record["ci95_upper"], digits)
        + "]"
    )


def write_fragment(path: Path, body: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "% Generated by finauth_audit/paper/generate_tables.py. Do not edit manually.\n"
        + body.strip()
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return sha256(path)


def tabular(
    headers: list[str], rows: list[list[str]], alignment: str | None = None
) -> str:
    align = alignment or ("l" + "r" * (len(headers) - 1))
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def generate_dataset_summary(out: Path, v06: V06PaperInputs | None = None) -> str:
    v06 = _resolve_v06_inputs(v06)
    main_manifest = json.loads(
        (ROOT / "manifests" / "main_data_manifest.json").read_text()
    )
    provenance_manifest = json.loads(
        (ROOT / "manifests" / "provenance_main_manifest.json").read_text()
    )
    public_manifest = json.loads(
        (
            ROOT / "results" / "public_audit" / "public_validation_manifest.json"
        ).read_text()
    )
    public_dataset_manifest = json.loads(
        (ROOT / "manifests" / "public_polymarket_dataset.json").read_text()
    )
    training_manifest = json.loads(
        (ROOT / "results" / "training_robustness_v03" / "manifest.json").read_text()
    )
    generator_manifest = json.loads(
        (ROOT / "manifests" / "generator_robustness_manifest.json").read_text()
    )
    binance_external = json.loads(
        (
            ROOT
            / "manifests"
            / "external_orderbook_v03"
            / "binance_dataset_manifest.json"
        ).read_text()
    )
    databento_external = json.loads(
        (
            ROOT
            / "manifests"
            / "external_orderbook_v03"
            / "databento_dataset_manifest.json"
        ).read_text()
    )
    real_agent_dataset = v06.dataset_manifest
    real_agent_registry = v06.registry
    rows = [
        [
            "Controlled core",
            f"{main_manifest['rows']:,}",
            f"{main_manifest['clusters']:,}",
            "30k / 10k / 5k / 5k clusters",
            "train / val. / paper / hidden",
        ],
        [
            "Provenance laundering",
            f"{provenance_manifest['rows']:,}",
            f"{provenance_manifest['clusters']:,}",
            "30k / 10k / 5k / 5k clusters",
            "5 attacks; 2 trace strata",
        ],
        [
            "Mechanistic generators",
            f"{sum(record['rows'] for record in generator_manifest['families'].values()):,}",
            f"{sum(record['clusters'] for record in generator_manifest['families'].values()):,}",
            "sequential / institutional",
            "validation robustness; test sealed",
        ],
        [
            "Public point-in-time",
            f"{public_dataset_manifest['rows']:,}",
            f"{public_dataset_manifest['event_clusters']:,}",
            "1.5k / 0.5k / 0.5k",
            (
                f"{public_manifest['public_clusters_evaluated']:,} validation reported; "
                r"test sealed"
            ),
        ],
        [
            "External order book",
            f"{binance_external['rows'] + databento_external['rows']:,}",
            f"{binance_external['event_clusters'] + databento_external['event_clusters']:,}",
            "Binance 120/500/83; MES 60/140/33",
            "powered public + licensed descriptive",
        ],
        [
            "Actual model proposals",
            f"{real_agent_registry['proposal_rows']:,}",
            f"{real_agent_registry['paper_test_clusters']:,}",
            (
                f"{real_agent_dataset['splits']['development']} / "
                f"{real_agent_dataset['splits']['paper_test']} / "
                f"{real_agent_dataset['splits']['community_hidden']} dates"
            ),
            (
                f"{len(real_agent_registry['models'])} models; "
                f"{len(real_agent_registry['tasks'])} tasks/date; aggregate only"
            ),
        ],
        [
            "Training utility",
            f"{training_manifest['fit_count']} fits",
            "400--20k / curriculum",
            "3 curricula; 4 learners",
            "6 mechanisms; N/A retained",
        ],
    ]
    body = tabular(
        ["Layer", "Rows", "Clusters", "Split or budget", "Coverage / boundary"],
        rows,
        (
            r"p{0.17\linewidth}"
            r">{\raggedleft\arraybackslash}p{0.10\linewidth}"
            r">{\raggedleft\arraybackslash}p{0.10\linewidth}"
            r"p{0.23\linewidth}p{0.28\linewidth}"
        ),
    )
    return write_fragment(out / "table_dataset_summary.tex", body)


def generate_external_orderbook(out: Path) -> str:
    binance = json.loads(
        (
            ROOT
            / "results"
            / "external_orderbook_v03"
            / "paper_test"
            / "binance"
            / "primary_endpoints.json"
        ).read_text()
    )
    databento = json.loads(
        (
            ROOT
            / "results"
            / "external_orderbook_v03"
            / "paper_test"
            / "databento"
            / "primary_endpoints.json"
        ).read_text()
    )
    rows = [
        [
            "Binance",
            "powered public",
            f"{binance['false_authorization_burden']['clusters']:,}",
            "False-auth. burden",
            signed(binance["false_authorization_burden"]["mean"]),
            interval(binance["false_authorization_burden"]),
            r"$\leq -0.015$",
            "pass",
        ],
        [
            "Binance",
            "powered public",
            f"{binance['laundering_burden']['clusters']:,}",
            "Laundering burden",
            signed(binance["laundering_burden"]["mean"]),
            interval(binance["laundering_burden"]),
            r"$\geq +0.020$",
            "fail",
        ],
        [
            "Binance",
            "powered public",
            f"{binance['false_authorization_burden']['clusters']:,}",
            "Intersection-union",
            "--",
            "--",
            "both arms",
            r"\textbf{FAIL (1/2)}",
        ],
        [
            "MES",
            "licensed descriptive",
            f"{databento['false_authorization_burden']['clusters']:,}",
            "False-auth. burden",
            signed(databento["false_authorization_burden"]["mean"]),
            interval(databento["false_authorization_burden"]),
            r"$\leq -0.015$",
            "descriptive",
        ],
        [
            "MES",
            "licensed descriptive",
            f"{databento['laundering_burden']['clusters']:,}",
            "Laundering burden",
            "1/6 by design",
            "--",
            "not applied",
            "not an estimand",
        ],
    ]
    body = tabular(
        [
            "Source",
            "Evidence",
            "Clusters",
            "Endpoint",
            "Diff.",
            r"95\% CI",
            "SESOI",
            "Status",
        ],
        rows,
        "llrlrrrp{0.18\\linewidth}",
    )
    return write_fragment(out / "table_external_orderbook.tex", body)


def generate_external_field_origins(out: Path) -> str:
    rows = [
        [
            "Observed source",
            "quotes, bars, depth, timestamps",
            "available at or before the registered decision time",
        ],
        [
            "Pre-decision derived",
            "momentum, volatility, imbalance, spread/cost proxies",
            "deterministic functions of observed source fields",
        ],
        [
            "Benchmark assigned",
            "prior family, action, confidence, source role, eligibility",
            "frozen weak-prior construction; not observed agent behavior",
        ],
        [
            "Outcome only",
            "entry/exit prices, return, utility, harm, burden",
            "sealed until the one-time evaluator; illegal for deployable rules",
        ],
    ]
    body = tabular(
        ["Origin", "Fields", "Evaluation boundary"],
        rows,
        "p{0.19\\linewidth}p{0.34\\linewidth}p{0.39\\linewidth}",
    )
    return write_fragment(out / "table_external_field_origins.tex", body)


def generate_benchmark_comparison(out: Path) -> str:
    rows = [
        [r"FinQA~\cite{chen2021finqa}", "financial QA", "no", "no", "no", "no"],
        [r"PIXIU~\cite{xie2023pixiu}", "financial tasks", "no", "no", "no", "no"],
        [r"FinBen~\cite{xie2024finben}", "financial tasks", "no", "no", "no", "no"],
        [r"AgentBench~\cite{liu2023agentbench}", "agent tasks", "no", "no", "no", "no"],
        [r"WebArena~\cite{zhou2023webarena}", "web tasks", "no", "no", "no", "no"],
        [
            r"Weak experiments~\cite{chou2025weakrules}",
            "experiment portfolios",
            "yes",
            "no",
            "no",
            "no",
        ],
        [
            r"\textbf{FinAuth-Audit}",
            "weak financial priors",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
    ]
    body = tabular(
        [
            "Benchmark",
            "Evaluation unit",
            "Rules",
            "Coverage audit",
            "Provenance",
            "Power gate",
        ],
        rows,
        "llcccc",
    )
    return write_fragment(out / "table_benchmark_comparison.tex", body)


def generate_main_results(out: Path) -> str:
    ranking = pd.read_csv(
        ROOT / "results" / "paper_test" / "controlled" / "raw_vs_certified_ranking.csv"
    )
    order = [
        "No Action",
        "Direct Prior",
        "Confidence Gate",
        "Uncertainty Gate",
        "Risk Filter",
        "Cost-Aware Gate",
        "Hard Role Gate",
        "Lifecycle Checklist",
    ]
    frame = ranking.set_index("rule").loc[order].reset_index()
    hypervolume = pd.read_csv(
        ROOT / "results" / "certification_robustness" / "continuous_hypervolume.csv"
    )[["rule", "worst_profile_hypervolume"]]
    frame = frame.merge(hypervolume, on="rule", how="left", validate="one_to_one")
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                tex_escape(row["rule"]),
                fmt(row["coverage"]),
                fmt(row["far"]),
                fmt(row["alr"]),
                fmt(row["review_rate"]),
                fmt(row["cau"], 2),
                fmt(row["moc"], 2),
                fmt(row["worst_profile_hypervolume"]),
            ]
        )
    body = tabular(
        [
            "Rule",
            "Cov.",
            "FAR",
            "ALR",
            "Review",
            "CAU",
            "MOC",
            "Val. HV",
        ],
        rows,
        "lrrrrrrr",
    )
    return write_fragment(out / "table_main_results.tex", body)


def generate_policy_families(out: Path) -> str:
    volume = pd.read_csv(
        ROOT / "results" / "certification_robustness" / "policy_volumes.csv"
    )
    continuous = pd.read_csv(
        ROOT / "results" / "certification_robustness" / "continuous_hypervolume.csv"
    )[["rule", "worst_profile_hypervolume"]]
    selected = volume[
        volume["policy"].isin(
            [
                "conservative_institution",
                "balanced_platform",
                "coverage_priority_research",
            ]
        )
    ].pivot(index="rule", columns="policy", values="worst_profile_volume")
    frame = selected.reset_index().merge(
        continuous, on="rule", how="left", validate="one_to_one"
    )
    order = [
        "No Action",
        "Direct Prior",
        "Confidence Gate",
        "Uncertainty Gate",
        "Risk Filter",
        "Cost-Aware Gate",
        "Hard Role Gate",
        "Lifecycle Checklist",
    ]
    frame = frame.set_index("rule").loc[order].reset_index()
    rows = [
        [
            tex_escape(row["rule"]),
            fmt(row["conservative_institution"]),
            fmt(row["balanced_platform"]),
            fmt(row["coverage_priority_research"]),
            fmt(row["worst_profile_hypervolume"]),
        ]
        for _, row in frame.iterrows()
    ]
    body = tabular(
        ["Rule", "Conservative", "Balanced", "Coverage-first", "Continuous HV"],
        rows,
        "lrrrr",
    )
    return write_fragment(out / "table_policy_families.tex", body)


def _ordered_v06_rules(values: pd.Series) -> list[str]:
    observed = set(values.astype(str))
    ordered = [rule for rule in V06_RULE_ORDER if rule in observed]
    return ordered + sorted(observed.difference(ordered))


def _track_label(track: str) -> str:
    return {"zero_shot": "Zero-shot", "recalibrated": "Recalibrated"}[track]


def _task_label(task: str) -> str:
    return {
        "directional_execution": "Directional",
        "risk_limit_increase": "Risk-limit",
    }[task]


def _bound_record(
    bounds: pd.DataFrame, rule: str, profile: str, metric: str
) -> dict[str, object]:
    current = bounds[
        bounds["rule"].astype(str).eq(rule)
        & bounds["profile"].astype(str).eq(profile)
        & bounds["metric"].astype(str).eq(metric)
    ]
    if len(current) != 1:
        raise ValueError(f"expected one bootstrap row for {rule}/{profile}/{metric}")
    return current.iloc[0].to_dict()


def _point_interval(record: dict[str, object], digits: int = 3) -> str:
    point = fmt(record.get("point"), digits)
    lower = fmt(record.get("lcb95"), digits)
    upper = fmt(record.get("ucb95"), digits)
    if point == r"\textit{N/A}" and lower == point and upper == point:
        return point
    return f"{point} [{lower}, {upper}]"


def generate_real_agent_task_tracks(
    out: Path, v06: V06PaperInputs | None = None
) -> str:
    v06 = _resolve_v06_inputs(v06)
    rows: list[list[str]] = []
    for track in V06_TRACKS:
        metrics = v06.metrics[track]
        bounds = v06.bootstrap_bounds[track]
        rules = _ordered_v06_rules(metrics["rule"])
        indexed = metrics.set_index(["profile", "rule"], verify_integrity=True)
        for profile in V06_TASK_PROFILES:
            task_metric = V06_TASK_METRICS[profile]
            for rule in rules:
                row = indexed.loc[(profile, rule)]
                coverage = _bound_record(bounds, rule, profile, "coverage")
                task_harm = _bound_record(bounds, rule, profile, task_metric)
                rows.append(
                    [
                        _track_label(track),
                        _task_label(profile),
                        tex_escape(rule),
                        _point_interval(coverage),
                        _point_interval(task_harm),
                        fmt(row["authority_violation_rate"]),
                        fmt(row["normalized_task_utility"]),
                    ]
                )
    body = tabular(
        [
            "Track",
            "Task",
            "Rule",
            r"Cov. [95\%]",
            r"Task harm [95\%]",
            "Authority viol.",
            "Utility",
        ],
        rows,
        "lllrrrr",
    )
    return write_fragment(out / "table_real_agent_rules.tex", body)


def generate_real_agent_overall(
    out: Path, v06: V06PaperInputs | None = None
) -> str:
    """Compact zero-shot operating points for the main result and supplement."""
    v06 = _resolve_v06_inputs(v06)
    frame = v06.metrics["zero_shot"]
    frame = frame[frame["profile"].astype(str).eq("overall")].copy()
    frame["rule"] = pd.Categorical(
        frame["rule"], categories=_ordered_v06_rules(frame["rule"]), ordered=True
    )
    frame = frame.sort_values("rule")
    rows = [
        [
            tex_escape(row["rule"]),
            f"{int(row['clusters']):,}",
            fmt(row["coverage"]),
            fmt(row["economic_loss_authorization_rate"]),
            fmt(row["material_harm_authorization_rate"]),
            fmt(row["tail_harm_authorization_rate"]),
            fmt(row["authority_violation_rate"]),
            fmt(row["normalized_task_utility"]),
            fmt(row["review_rate"]),
        ]
        for _, row in frame.iterrows()
    ]
    body = tabular(
        [
            "Rule",
            "Dates",
            "Cov.",
            "Econ. loss",
            "Material harm",
            "Tail harm",
            "Authority viol.",
            "Utility",
            "Review",
        ],
        rows,
        "lrrrrrrrr",
    )
    return write_fragment(out / "table_real_agent_overall.tex", body)


def generate_real_agent_recalibration(
    out: Path, v06: V06PaperInputs | None = None
) -> str:
    """Development-only recalibration deltas with structural N/A retained."""
    v06 = _resolve_v06_inputs(v06)
    zero = v06.metrics["zero_shot"]
    recal = v06.metrics["recalibrated"]
    zero = zero[zero["profile"].astype(str).eq("overall")].set_index("rule")
    recal = recal[recal["profile"].astype(str).eq("overall")].set_index("rule")
    rows: list[list[str]] = []
    for rule in _ordered_v06_rules(zero.index.to_series()):
        left = zero.loc[rule]
        right = recal.loc[rule]
        coverage_delta = (
            None
            if pd.isna(right["coverage"])
            else float(right["coverage"]) - float(left["coverage"])
        )
        authority_delta = (
            None
            if pd.isna(right["authority_violation_rate"])
            else float(right["authority_violation_rate"])
            - float(left["authority_violation_rate"])
        )
        utility_delta = (
            None
            if pd.isna(right["normalized_task_utility"])
            else float(right["normalized_task_utility"])
            - float(left["normalized_task_utility"])
        )
        validity = v06.registry.get("calibration_validity_by_rule", {}).get(rule)
        if isinstance(validity, dict) and validity.get("calibration_valid") is False:
            status = tex_escape(str(validity.get("status", "N/A")))
        elif pd.isna(right["coverage"]):
            status = r"\textit{N/A}"
        elif rule in {"Direct Prior", "Hard Role Gate"}:
            status = "unchanged"
        else:
            status = "calibrated"
        rows.append(
            [
                tex_escape(rule),
                fmt(left["coverage"]),
                fmt(right["coverage"]),
                signed(coverage_delta),
                signed(authority_delta),
                signed(utility_delta),
                status,
            ]
        )
    body = tabular(
        [
            "Rule",
            "Zero cov.",
            "Recal. cov.",
            r"$\Delta$ cov.",
            r"$\Delta$ authority viol.",
            r"$\Delta$ utility",
            "Status",
        ],
        rows,
        "lrrrrrl",
    )
    return write_fragment(out / "table_real_agent_recalibration.tex", body)


def rank_support_status(primary_support: object) -> str:
    if primary_support is True:
        return "supported"
    if primary_support is False:
        return "not supported"
    if primary_support is None:
        return "N/A"
    raise ValueError("primary_support must be true, false, or null")


def rank_support_sentence(primary_support: object, independent_dates: int) -> str:
    unit = f"the independent unit is one UTC date ($n={independent_dates}$)"
    if primary_support is True:
        return f"The preregistered zero-shot inverse rank-transfer criterion is supported; {unit}."
    if primary_support is False:
        return f"The preregistered zero-shot inverse rank-transfer criterion is not supported; {unit}."
    if primary_support is None:
        return (
            "The preregistered zero-shot inverse rank-transfer criterion is "
            r"\textit{N/A} because at least one registered support statistic is "
            f"undefined; {unit}."
        )
    raise ValueError("primary_support must be true, false, or null")


def generate_real_agent_transfer(out: Path, v06: V06PaperInputs | None = None) -> str:
    v06 = _resolve_v06_inputs(v06)
    rank = v06.rank_transfer
    point = rank["point"]
    exact = rank["exact_permutation"]
    bootstrap = rank["date_cluster_bootstrap"]
    status = rank_support_status(rank["primary_support"])
    bootstrap_interval = _point_interval(
        {
            "point": bootstrap["spearman_median"],
            "lcb95": bootstrap["spearman_lcb95"],
            "ucb95": bootstrap["spearman_ucb95"],
        }
    )
    rows: list[list[str]] = [
        [
            "Primary: all rules",
            f"{int(bootstrap['clusters']):,}",
            f"{int(point['n_rules'])}",
            fmt(point["spearman_rho"]),
            bootstrap_interval,
            fmt(point["kendall_tau_b"]),
            fmt(exact["exact_p_value"]),
            fmt(bootstrap["probability_below_zero"]),
            fmt(bootstrap["valid_replicate_fraction"]),
            r"\textit{N/A}" if status == "N/A" else status,
        ]
    ]
    lifecycle_loo = next(
        (
            record
            for record in rank.get("leave_one_rule_out", [])
            if record.get("omitted_rule") == "Lifecycle Checklist"
        ),
        None,
    )
    if isinstance(lifecycle_loo, dict):
        rows.append(
            [
                "Sensitivity: omit Lifecycle",
                f"{int(bootstrap['clusters']):,}",
                f"{int(lifecycle_loo['n_rules'])}",
                fmt(lifecycle_loo.get("spearman_rho")),
                r"\textit{N/A}",
                fmt(lifecycle_loo.get("kendall_tau_b")),
                fmt(lifecycle_loo.get("exact_permutation_p")),
                r"\textit{N/A}",
                r"\textit{N/A}",
                "registered sensitivity",
            ]
        )
    body = tabular(
        [
            "Endpoint",
            "Dates",
            "Rules",
            r"$\rho$",
            r"Boot. $\rho$ [95\%]",
            r"$\tau_b$",
            "Exact $p$",
            r"$\Pr(\rho<0)$",
            "Valid boot.",
            "Primary support",
        ],
        rows,
        "lrrrrrrrrl",
    )
    return write_fragment(out / "table_real_agent_transfer.tex", body)


def generate_real_agent_source_quality(
    out: Path, v06: V06PaperInputs | None = None
) -> str:
    v06 = _resolve_v06_inputs(v06)
    frame = v06.source_quality.sort_values(["task_id", "model_id"])
    rows = [
        [
            tex_escape(row["model_id"]),
            _task_label(str(row["task_id"])),
            f"{int(row['clusters']):,}",
            fmt(row["raw_schema_validity"]),
            fmt(row["malformed_placeholder_rate"]),
            f"{int(row['repairs_applied'])}",
            fmt(row["proposal_abstain_rate"]),
            fmt(row["mean_confidence"]),
            fmt(row["review_recommended_rate"]),
            fmt(row["material_harm_overconfidence_rate"]),
        ]
        for _, row in frame.iterrows()
    ]
    body = tabular(
        [
            "Proposal source",
            "Task",
            "Dates",
            "Raw valid",
            "Malformed",
            "Repairs",
            "Abstain",
            "Mean conf.",
            "Review flag",
            "Harm overconf.",
        ],
        rows,
        "llrrrrrrrr",
    )
    return write_fragment(out / "table_real_agent_sources.tex", body)


def generate_real_agent_rules(out: Path, v06: V06PaperInputs | None = None) -> str:
    return generate_real_agent_task_tracks(out, v06)


def generate_real_agent_sources(out: Path, v06: V06PaperInputs | None = None) -> str:
    return generate_real_agent_source_quality(out, v06)


def generate_real_agent_claim_macros(
    out: Path, v06: V06PaperInputs | None = None
) -> str:
    v06 = _resolve_v06_inputs(v06)
    support = v06.rank_transfer["primary_support"]
    dates = int(v06.registry["paper_test_clusters"])
    status = rank_support_status(support)
    status_tex = r"\textit{N/A}" if status == "N/A" else status
    sentence = rank_support_sentence(support, dates)
    point = v06.rank_transfer["point"]
    exact = v06.rank_transfer["exact_permutation"]
    bootstrap = v06.rank_transfer["date_cluster_bootstrap"]
    lifecycle_loo = next(
        (
            record
            for record in v06.rank_transfer.get("leave_one_rule_out", [])
            if record.get("omitted_rule") == "Lifecycle Checklist"
        ),
        {},
    )
    reversal_records = v06.rank_transfer.get("pairwise_reversal", [])
    reversal_count = sum(
        1 for record in reversal_records if record.get("reversal") is True
    )
    proposal_rows = int(v06.registry["proposal_rows"])
    body = "\n".join(
        [
            rf"\newcommand{{\RealAgentRankSupportStatus}}{{{status_tex}}}",
            rf"\newcommand{{\RealAgentRankSupportClaim}}{{{sentence}}}",
            r"\newcommand{\RealAgentRankIndependentUnit}{UTC date}",
            rf"\newcommand{{\RealAgentRankIndependentUnitCount}}{{{dates}}}",
            rf"\newcommand{{\RealAgentProposalRows}}{{{proposal_rows:,}}}",
            rf"\newcommand{{\RealAgentRankSpearman}}{{{fmt(point.get('spearman_rho'))}}}",
            rf"\newcommand{{\RealAgentRankKendall}}{{{fmt(point.get('kendall_tau_b'))}}}",
            rf"\newcommand{{\RealAgentRankExactP}}{{{fmt(exact.get('exact_p_value'))}}}",
            rf"\newcommand{{\RealAgentRankBootstrapMedian}}{{{fmt(bootstrap.get('spearman_median'))}}}",
            rf"\newcommand{{\RealAgentRankBootstrapLCB}}{{{fmt(bootstrap.get('spearman_lcb95'))}}}",
            rf"\newcommand{{\RealAgentRankBootstrapUCB}}{{{fmt(bootstrap.get('spearman_ucb95'))}}}",
            rf"\newcommand{{\RealAgentRankNegativeProbability}}{{{fmt(bootstrap.get('probability_below_zero'))}}}",
            rf"\newcommand{{\RealAgentLifecycleOmitSpearman}}{{{fmt(lifecycle_loo.get('spearman_rho'))}}}",
            rf"\newcommand{{\RealAgentPairwiseReversalCount}}{{{reversal_count}}}",
            rf"\newcommand{{\RealAgentPairwiseComparisonCount}}{{{len(reversal_records)}}}",
        ]
    )
    return write_fragment(out / "claims_real_agent_v06.tex", body)


def generate_provenance_results(out: Path) -> str:
    summary = pd.read_csv(
        ROOT / "results" / "paper_test" / "provenance" / "summary.csv"
    )
    order = [
        "No Action",
        "No Role Gate",
        "Shared Threshold",
        "Soft Penalty",
        "Hard Role Gate",
        "Provenance Hard Gate",
        "Provenance Learned Gate",
        "Lifecycle Checklist",
        "EPV Adapter",
    ]
    frame = summary.set_index("rule").loc[order].reset_index()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                tex_escape(row["rule"]),
                fmt(row["coverage"]),
                fmt(row["far"]),
                fmt(row["alr"]),
                fmt(row["direct_leakage_rate"]),
                fmt(row["indirect_leakage_rate"]),
                fmt(row["safe_delegation_coverage"]),
                fmt(row["review_rate"]),
            ]
        )
    body = tabular(
        ["Rule", "Cov.", "FAR", "ALR", "Direct", "Indirect", "Safe del.", "Review"],
        rows,
        "lrrrrrrr",
    )
    return write_fragment(out / "table_provenance_results.tex", body)


def generate_certification_profiles(out: Path) -> str:
    bounds = pd.read_csv(
        ROOT / "results" / "paper_test" / "controlled" / "bootstrap_bounds.csv"
    )
    summary = pd.read_csv(
        ROOT / "results" / "paper_test" / "controlled" / "certification_summary.csv"
    )
    volume = summary.melt(
        id_vars=["rule"],
        value_vars=["overall", "stress"],
        var_name="profile",
        value_name="certification_volume",
    )
    frame = bounds[bounds["profile"].isin(["overall", "stress"])].merge(
        volume, on=["rule", "profile"], how="left", validate="one_to_one"
    )
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                tex_escape(row["rule"]),
                tex_escape(row["profile"]),
                fmt(row["coverage_lcb95"]),
                fmt(row["far_ucb95"]),
                fmt(row["alr_ucb95"]),
                fmt(row["certification_volume"]),
            ]
        )
    body = tabular(
        ["Rule", "Profile", "Coverage LCB", "FAR UCB", "ALR UCB", "Cert. volume"],
        rows,
        "llrrrr",
    )
    return write_fragment(out / "table_certification_profiles.tex", body)


def generate_traceability(out: Path) -> str:
    frame = pd.read_csv(
        ROOT / "results" / "paper_test" / "provenance" / "by_traceability.csv"
    )
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                tex_escape(row["rule"]),
                tex_escape(row["traceability"]),
                fmt(row["coverage"]),
                fmt(row["alr"]),
                fmt(row["safe_delegation_coverage"]),
                fmt(row["false_block_rate"]),
            ]
        )
    body = tabular(
        ["Rule", "Traceability", "Cov.", "ALR", "Safe del.", "False block"],
        rows,
        "llrrrl",
    )
    return write_fragment(out / "table_provenance_traceability.tex", body)


def generate_provenance_identifiability(out: Path) -> str:
    frame = pd.read_csv(
        ROOT / "results" / "provenance_identifiability" / "ambiguity_bounds.csv"
    )
    signature_labels = {
        "current_role_only": "current role only",
        "legal_partial": "legal partial signature",
    }
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                tex_escape(row["traceability"]),
                tex_escape(signature_labels[str(row["signature"])]),
                f"{int(row['rows']):,}",
                fmt(row["collision_row_fraction"]),
                fmt(row["empirical_bayes_error_lower_bound"]),
                fmt(row["empirical_joint_separation"]),
            ]
        )
    body = tabular(
        [
            "Traceability",
            "Observable signature",
            "Rows",
            "Collision frac.",
            "Bayes risk",
            "Joint sep.",
        ],
        rows,
        "llrrrr",
    )
    return write_fragment(out / "table_provenance_identifiability.tex", body)


def generate_public_training(out: Path) -> tuple[str, str]:
    power = json.loads(
        (ROOT / "results" / "public_audit" / "public_power_gate.json").read_text()
    )
    source = pd.read_csv(
        ROOT / "results" / "public_audit" / "source_classification.csv"
    )
    classification_labels = {
        "descriptive_public_derived": "descriptive public-derived",
        "excluded_from_inference": "excluded from inference",
        "controlled_information_shock_extension": "controlled information-shock",
        "exploratory_non_vintage": "exploratory non-vintage",
        "confirmatory_candidate": "confirmatory candidate; structural only",
    }
    power_rows = [
        [
            "Public point-in-time",
            f"{power['test_clusters']:,} (structural)",
            fmt(power["estimated_power"]),
            fmt(power["power_lcb95"]),
            "0.80",
            r"exploratory\_only",
        ],
        *[
            [
                tex_escape(row["source"]),
                "",
                "",
                "",
                "",
                tex_escape(classification_labels[row["classification"]]),
            ]
            for _, row in source.iterrows()
        ],
    ]
    power_body = tabular(
        ["Layer / source", "Clusters", "Power", "LCB95", "Req.", "Classification"],
        power_rows,
        "llrrrp{0.29\\linewidth}",
    )
    power_hash = write_fragment(out / "table_public_power_gate.tex", power_body)

    primary = pd.read_csv(
        ROOT / "results" / "training_utility_smoke" / "primary_endpoint_by_seed.csv"
    )
    valid = primary.groupby("variant")["primary_endpoint_valid"].sum().astype(int)
    rows = []
    for variant in ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]:
        if variant == "D0":
            interpretation = "prediction diagnostic"
        elif variant == "D1":
            interpretation = "authorization reference"
        else:
            interpretation = "authorization variant"
        rows.append([variant, f"{int(valid.get(variant, 0))}/5", interpretation])
    rows.append(["D7 role-neutral", "0/5", "negative control; outside Holm"])
    training_body = tabular(
        ["Variant", "Valid endpoints", "Interpretation"], rows, "lrp{0.39\\linewidth}"
    )
    training_hash = write_fragment(out / "table_training_validity.tex", training_body)
    return power_hash, training_hash


def generate_holm(out: Path) -> str:
    frame = pd.read_csv(
        ROOT / "results" / "training_utility_smoke" / "holm_validation_contrasts.csv"
    )
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                tex_escape(row["variant"] + " vs " + row["reference_variant"]),
                fmt(row["mean_far_difference_vs_reference"]),
                fmt(row["valid_seed_pairs"], 0),
                fmt(row["raw_p"]),
                fmt(row["holm_adjusted_p"]),
                "no" if not bool(row["reject"]) else "yes",
            ]
        )
    body = tabular(
        ["Contrast", "Delta FAR", "Valid pairs", "Raw p", "Holm p", "Reject"],
        rows,
        "lrrrrl",
    )
    return write_fragment(out / "table_training_holm.tex", body)


def generate_stability(out: Path) -> str:
    stability = pd.read_csv(
        ROOT / "results" / "generator_robustness" / "rank_stability.csv"
    )
    paper = json.loads(
        (ROOT / "results" / "paper_test" / "rank_stability.json").read_text()
    )
    rows = [
        ["Validation vs paper test", "raw FAR", fmt(paper["raw_far_rank"]["spearman"])],
        [
            "Validation vs paper test",
            "certification",
            fmt(paper["certified_rank"]["spearman"]),
        ],
    ]
    generator_labels = {
        "institutional_workflow": "Institutional",
        "sequential_market": "Sequential",
        "utility_iid": "Utility IID",
    }
    for _, row in stability[stability["metric"] == "raw_far_rank"].iterrows():
        rows.append(
            [
                tex_escape(
                    f"{generator_labels.get(row['generator_a'], row['generator_a'])} "
                    f"vs {generator_labels.get(row['generator_b'], row['generator_b'])}"
                ),
                "raw FAR",
                fmt(row["spearman_rho"]),
            ]
        )
    body = tabular(
        ["Comparison", "Rank", "Spearman"],
        rows,
        "p{0.48\\linewidth}p{0.24\\linewidth}r",
    )
    return write_fragment(out / "table_scale_stability.tex", body)


def generate_artifact_checklist(out: Path, v06: V06PaperInputs | None = None) -> str:
    v06 = _resolve_v06_inputs(v06)
    release_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    registry = v06.registry
    rows = [
        ["Paper test", r"\path{results/paper_test/}", "one-time registry; 5k clusters"],
        ["Community hidden", r"\path{sealed/community_hidden/}", "5k clusters; unevaluated"],
        [
            "Freeze archive",
            r"\path{manifests/paper_test_snapshots/}",
            "content-addressed SHA-256",
        ],
        ["Point-in-time layer", r"\path{results/public_audit/}", "power and timestamp gates"],
        [
            "External order book",
            r"\path{results/external_orderbook_v03/}",
            "powered mixed result; 550 checks",
        ],
        [
            "Actual-model paper test",
            r"\path{results/real_agent_v06/paper_test/}",
            (
                f"{len(registry['models'])} models; "
                f"{registry['paper_test_clusters']}-date, "
                f"{len(registry['tasks'])}-task aggregate test"
            ),
        ],
        [
            "Historical-memory audit",
            r"\path{results/real_agent_v06/historical_memory_audit/}",
            "outcome-blind aggregates; no rationale text persisted",
        ],
        [
            "Training robustness",
            r"\path{results/training_robustness_v03/}",
            "144 fits; 6 mechanisms",
        ],
        ["Figures", r"\path{paper/figures/}", "PDF, SVG, PNG; text QA"],
        [
            "HTML and cards",
            r"\path{paper/html/}; \path{docs/}",
            "combined paper/supplement; local assets",
        ],
        [
            "Release archive",
            rf"\path{{dist/finauth-audit-{release_version}.tar.gz}}",
            "allowlist; clean-room hash and split scan",
        ],
        [
            "Verifier",
            r"\path{verify_artifact.py}",
            "all frozen surfaces, fields, hashes, and tests",
        ],
    ]
    body = tabular(
        ["Artifact", "Release path", "Verification"],
        rows,
        "p{0.20\\linewidth}p{0.35\\linewidth}p{0.31\\linewidth}",
    )
    return write_fragment(out / "table_artifact_checklist.tex", body)


def generate(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    v06 = load_v06_paper_inputs()
    generated = {
        "table_benchmark_comparison.tex": generate_benchmark_comparison(output),
        "table_dataset_summary.tex": generate_dataset_summary(output, v06),
        "table_main_results.tex": generate_main_results(output),
        "table_provenance_results.tex": generate_provenance_results(output),
        "table_certification_profiles.tex": generate_certification_profiles(output),
        "table_provenance_traceability.tex": generate_traceability(output),
        "table_provenance_identifiability.tex": generate_provenance_identifiability(
            output
        ),
        "table_policy_families.tex": generate_policy_families(output),
        "table_real_agent_rules.tex": generate_real_agent_task_tracks(output, v06),
        "table_real_agent_overall.tex": generate_real_agent_overall(output, v06),
        "table_real_agent_recalibration.tex": generate_real_agent_recalibration(
            output, v06
        ),
        "table_real_agent_transfer.tex": generate_real_agent_transfer(output, v06),
        "table_real_agent_sources.tex": generate_real_agent_source_quality(output, v06),
        "claims_real_agent_v06.tex": generate_real_agent_claim_macros(output, v06),
        "table_public_power_gate.tex": None,
        "table_training_validity.tex": None,
        "table_training_holm.tex": generate_holm(output),
        "table_scale_stability.tex": generate_stability(output),
        "table_artifact_checklist.tex": generate_artifact_checklist(output, v06),
        "table_external_orderbook.tex": generate_external_orderbook(output),
        "table_external_field_origins.tex": generate_external_field_origins(output),
    }
    power_hash, training_hash = generate_public_training(output)
    generated["table_public_power_gate.tex"] = power_hash
    generated["table_training_validity.tex"] = training_hash
    inputs = {}
    input_paths = [
        ROOT / "results" / "paper_test" / "controlled" / "raw_vs_certified_ranking.csv",
        ROOT / "results" / "paper_test" / "provenance" / "summary.csv",
        ROOT / "results" / "provenance_identifiability" / "ambiguity_bounds.csv",
        ROOT / "results" / "certification_robustness" / "policy_volumes.csv",
        ROOT / "results" / "paper_test" / "rank_stability.json",
        ROOT / "results" / "generator_robustness" / "rank_stability.csv",
        ROOT / "results" / "public_audit" / "public_power_gate.json",
        ROOT
        / "results"
        / "external_orderbook_v03"
        / "paper_test"
        / "binance"
        / "primary_endpoints.json",
        ROOT
        / "results"
        / "external_orderbook_v03"
        / "paper_test"
        / "databento"
        / "primary_endpoints.json",
        ROOT / "manifests" / "external_orderbook_v03_test_registry.json",
        ROOT / "results" / "training_utility_smoke" / "primary_endpoint_by_seed.csv",
        ROOT / "results" / "training_robustness_v03" / "learning_curve_summary.csv",
        *v06.input_paths,
    ]
    for path in dict.fromkeys(input_paths):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    (output / "table_manifest.json").write_text(
        json.dumps(
            {
                "generator": "finauth_audit/paper/generate_tables.py",
                "version": "0.6.0-aggregate-only-paper-integration",
                "inputs": inputs,
                "outputs": generated,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate FinAuth-Audit LaTeX table fragments."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    generate(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
