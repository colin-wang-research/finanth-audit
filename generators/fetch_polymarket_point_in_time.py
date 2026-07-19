from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "FinAuth-Audit/0.2 point-in-time research"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_history(
    path: Path, histories: dict[str, list[dict[str, object]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                json.dump(histories, handle, sort_keys=True, separators=(",", ":"))


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    retries: int,
) -> Any:
    headers = {"User-Agent": USER_AGENT}
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(4.0, 0.5 * 2**attempt))
    raise RuntimeError(f"request failed after {retries} attempts: {url}: {last_error}")


def _parse_json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(str(value))
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def _timestamp(value: object) -> float | None:
    if value is None or not str(value).strip():
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else float(parsed.timestamp())


def _utc_timestamp(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _event_cluster_id(market: dict[str, Any]) -> str:
    events = market.get("events") or []
    if events and isinstance(events[0], dict) and events[0].get("id") is not None:
        return f"polymarket-event-{events[0]['id']}"
    return f"polymarket-market-{market.get('id')}"


def _scheduled_end_timestamp(market: dict[str, Any]) -> float | None:
    scheduled = _timestamp(market.get("endDate"))
    if scheduled is not None:
        return scheduled
    events = market.get("events") or []
    if events and isinstance(events[0], dict):
        return _timestamp(events[0].get("endDate"))
    return None


def _decision_target_timestamp(
    market: dict[str, Any], lifecycle_fraction: float
) -> float | None:
    created = _timestamp(market.get("createdAt") or market.get("startDate"))
    scheduled_end = _scheduled_end_timestamp(market)
    closed = _timestamp(market.get("closedTime") or market.get("umaEndDate"))
    if (
        created is None
        or scheduled_end is None
        or closed is None
        or scheduled_end <= created
    ):
        return None
    target = created + lifecycle_fraction * (scheduled_end - created)
    return target if created < target < closed - 60 else None


def _series_key(market: dict[str, Any]) -> str:
    events = market.get("events") or []
    if events and isinstance(events[0], dict):
        series = events[0].get("series") or []
        if series and isinstance(series[0], dict):
            value = series[0].get("slug") or series[0].get("title")
            if value:
                return f"series:{value}"
    tags = market.get("tags") or []
    if tags and isinstance(tags[0], dict):
        value = tags[0].get("slug") or tags[0].get("label")
        if value:
            return f"tag:{value}"
    for field in ("category", "sportsMarketType", "feeType", "marketType"):
        if market.get(field):
            return f"{field}:{market[field]}"
    return "unclassified"


def _evenly_spaced(values: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if len(values) <= target:
        return values
    if target <= 1:
        return values[:target]
    indices = {
        int(round(position * (len(values) - 1) / (target - 1)))
        for position in range(target)
    }
    return [values[index] for index in sorted(indices)]


def _valid_candidate(market: dict[str, Any], config: dict[str, Any], cutoff: float) -> bool:
    prices = []
    for value in _parse_json_list(market.get("outcomePrices")):
        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            return False
    tokens = [str(value) for value in _parse_json_list(market.get("clobTokenIds"))]
    created = _timestamp(market.get("createdAt") or market.get("startDate"))
    closed = _timestamp(market.get("closedTime") or market.get("umaEndDate"))
    scheduled_end = _scheduled_end_timestamp(market)
    if (
        len(prices) < 2
        or len(tokens) < 2
        or created is None
        or closed is None
        or scheduled_end is None
    ):
        return False
    resolved = (prices[0] >= 0.99 and prices[1] <= 0.01) or (
        prices[1] >= 0.99 and prices[0] <= 0.01
    )
    try:
        volume = float(market.get("volumeNum") or market.get("volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    duration_hours = (closed - created) / 3600.0
    return bool(
        market.get("closed")
        and market.get("enableOrderBook")
        and resolved
        and closed <= cutoff
        and duration_hours >= float(config["min_market_duration_hours"])
        and volume >= float(config["min_market_volume"])
    )


def _candidate_score(market: dict[str, Any]) -> tuple[float, float, str]:
    try:
        volume = float(market.get("volumeNum") or market.get("volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    created = _timestamp(market.get("createdAt") or market.get("startDate")) or 0.0
    closed = _timestamp(market.get("closedTime") or market.get("umaEndDate")) or created
    return (volume, closed - created, str(market.get("id", "")))


def fetch_metadata(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoint = str(config["gamma_endpoint"])
    cutoff = (
        _utc_timestamp(config["sampling_end_date"]).timestamp()
        if config.get("sampling_end_date")
        else datetime.now(timezone.utc).timestamp()
    )
    by_event: dict[str, dict[str, Any]] = {}
    pages = 0
    markets_seen = 0
    duplicate_market_ids = 0
    seen_market_ids: set[str] = set()
    start = _utc_timestamp(config["sampling_start_date"])
    end = _utc_timestamp(datetime.fromtimestamp(cutoff, timezone.utc))
    boundaries = list(pd.date_range(start=start, end=end, freq="MS"))
    if not boundaries or boundaries[0] != start:
        boundaries.insert(0, start)
    if boundaries[-1] < end:
        boundaries.append(end)
    window_counts: dict[str, int] = {}
    truncated_windows = 0
    truncated_window_names: list[str] = []
    window_errors: list[str] = []
    pagination = str(config.get("pagination", "keyset"))
    for window_start, window_end in zip(boundaries[:-1], boundaries[1:]):
        cursor: str | None = None
        offset = 0
        local: dict[str, dict[str, Any]] = {}
        exhausted = False
        for _ in range(int(config["max_pages_per_month"])):
            params: dict[str, object] = {
                "closed": "true",
                "limit": int(config["page_size"]),
                "order": "closedTime",
                "ascending": "false",
                "include_tag": "true",
                "start_date_min": window_start.isoformat(),
                "start_date_max": window_end.isoformat(),
            }
            if pagination == "offset":
                params["offset"] = offset
            elif cursor:
                params["after_cursor"] = cursor
            try:
                response = _json_request(
                    endpoint + "?" + urllib.parse.urlencode(params),
                    retries=int(config["request_retries"]),
                )
            except RuntimeError as exc:
                window_errors.append(f"{window_start.strftime('%Y-%m')}: {exc}")
                exhausted = True
                break
            page = response if isinstance(response, list) else response.get("markets", [])
            pages += 1
            if not page:
                exhausted = True
                break
            for market in page:
                market_id = str(market.get("id"))
                if market_id in seen_market_ids:
                    duplicate_market_ids += 1
                    continue
                seen_market_ids.add(market_id)
                markets_seen += 1
                if not _valid_candidate(market, config, cutoff):
                    continue
                cluster = _event_cluster_id(market)
                current = local.get(cluster)
                if current is None or _candidate_score(market) > _candidate_score(current):
                    selected = dict(market)
                    selected["event_cluster_id"] = cluster
                    selected["sampling_month"] = window_start.strftime("%Y-%m")
                    selected["sampling_series_key"] = _series_key(market)
                    local[cluster] = selected
            if pagination == "offset":
                if len(page) < int(config["page_size"]):
                    exhausted = True
                    break
                offset += int(config["page_size"])
            else:
                cursor = response.get("next_cursor")
                if not cursor:
                    exhausted = True
                    break
            time.sleep(float(config["request_delay_seconds"]))
        if not exhausted:
            truncated_windows += 1
            truncated_window_names.append(window_start.strftime("%Y-%m"))
        selected_window: list[dict[str, Any]] = []
        series_counts: dict[str, int] = {}
        for market in sorted(local.values(), key=_candidate_score, reverse=True):
            series = str(market["sampling_series_key"])
            if series_counts.get(series, 0) >= int(config["max_clusters_per_series_per_month"]):
                continue
            selected_window.append(market)
            series_counts[series] = series_counts.get(series, 0) + 1
            if len(selected_window) >= int(config["max_clusters_per_month"]):
                break
        for market in selected_window:
            by_event[str(market["event_cluster_id"])] = market
        window_counts[window_start.strftime("%Y-%m")] = len(selected_window)
    markets = sorted(
        by_event.values(),
        key=lambda item: (
            _timestamp(item.get("closedTime") or item.get("umaEndDate")) or 0.0,
            str(item.get("id")),
        ),
    )
    markets = _evenly_spaced(markets, int(config["candidate_event_clusters"]))
    close_times = [
        _timestamp(item.get("closedTime") or item.get("umaEndDate"))
        for item in markets
    ]
    close_times = [value for value in close_times if value is not None]
    series_counts = pd.Series(
        [str(item.get("sampling_series_key", "unclassified")) for item in markets],
        dtype="object",
    ).value_counts()
    return markets, {
        "fetch_cutoff_utc": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
        "pages": pages,
        "markets_seen": markets_seen,
        "duplicate_market_ids": duplicate_market_ids,
        "candidate_event_clusters": len(markets),
        "sampling_start_date": str(config["sampling_start_date"]),
        "sampling_end_date": str(config.get("sampling_end_date")),
        "sampling_windows": len(boundaries) - 1,
        "truncated_windows": truncated_windows,
        "truncated_window_names": truncated_window_names,
        "window_error_count": len(window_errors),
        "window_errors": window_errors,
        "window_counts": window_counts,
        "candidate_close_min_utc": (
            datetime.fromtimestamp(min(close_times), timezone.utc).isoformat()
            if close_times
            else None
        ),
        "candidate_close_max_utc": (
            datetime.fromtimestamp(max(close_times), timezone.utc).isoformat()
            if close_times
            else None
        ),
        "candidate_distinct_series": int(len(series_counts)),
        "candidate_top_series": {
            str(key): int(value) for key, value in series_counts.head(10).items()
        },
    }


def fetch_histories(
    markets: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, list[dict[str, object]]], list[str], dict[str, Any]]:
    endpoint = str(config["history_endpoint"])
    batch_size = int(config["history_batch_size"])
    token_ids: list[str] = []
    for market in markets:
        tokens = [str(value) for value in _parse_json_list(market.get("clobTokenIds"))]
        if tokens:
            token_ids.append(tokens[0])
    histories: dict[str, list[dict[str, object]]] = {}
    errors: list[str] = []
    for start in range(0, len(token_ids), batch_size):
        batch = token_ids[start : start + batch_size]
        try:
            response = _json_request(
                endpoint,
                payload={
                    "markets": batch,
                    "interval": str(config["history_interval"]),
                    "fidelity": int(config["history_fidelity_minutes"]),
                },
                retries=int(config["request_retries"]),
            )
            for token, history in response.get("history", {}).items():
                histories[str(token)] = [
                    {**item, "source": "clob_prices_history"}
                    for item in history
                    if isinstance(item, dict)
                ] if isinstance(history, list) else []
        except RuntimeError as exc:
            errors.append(str(exc))
        time.sleep(float(config["request_delay_seconds"]))
    clob_sufficient = {
        token
        for token, history in histories.items()
        if len(history) >= int(config["min_history_points"])
    }
    missing_markets: list[dict[str, Any]] = []
    for market in markets:
        tokens = [str(value) for value in _parse_json_list(market.get("clobTokenIds"))]
        if tokens and tokens[0] not in clob_sufficient:
            missing_markets.append(market)
    trade_histories, trade_errors, trade_stats = fetch_trade_histories(
        missing_markets, config
    )
    errors.extend(trade_errors)
    for token, history in trade_histories.items():
        if len(history) >= int(config["min_history_points"]):
            histories[token] = history
    stats = {
        "clob_tokens_sufficient": len(clob_sufficient),
        "trade_fallback_candidates": len(missing_markets),
        **trade_stats,
    }
    return histories, errors, stats


def _trade_to_probability(
    trade: dict[str, Any], yes_token: str, no_token: str
) -> tuple[int, float] | None:
    try:
        timestamp = int(trade["timestamp"])
        price = float(trade["price"])
    except (KeyError, TypeError, ValueError):
        return None
    asset = str(trade.get("asset") or "")
    if asset == yes_token:
        probability = price
    elif asset == no_token:
        probability = 1.0 - price
    elif trade.get("outcomeIndex") == 0:
        probability = price
    elif trade.get("outcomeIndex") == 1:
        probability = 1.0 - price
    else:
        return None
    if not 0.0 <= probability <= 1.0:
        return None
    return timestamp, probability


def _fetch_market_trades(
    market: dict[str, Any], config: dict[str, Any]
) -> tuple[str, list[dict[str, object]], bool, int, str | None]:
    tokens = [str(value) for value in _parse_json_list(market.get("clobTokenIds"))]
    condition_id = str(market.get("conditionId") or "")
    target = _decision_target_timestamp(
        market, float(config["decision_lifecycle_fraction"])
    )
    created = _timestamp(market.get("createdAt") or market.get("startDate"))
    closed = _timestamp(market.get("closedTime") or market.get("umaEndDate"))
    if len(tokens) < 2 or not condition_id or target is None or created is None or closed is None:
        return tokens[0] if tokens else "", [], False, 0, "invalid_trade_query_bounds"
    end = min(
        target + float(config["trade_action_window_hours"]) * 3600.0,
        closed - 1,
    )
    common = {
        "market": condition_id,
        "limit": int(config["trade_history_limit"]),
        "takerOnly": "true",
    }
    requests = [
        {**common, "start": int(created), "end": int(target)},
        {**common, "start": int(target) + 1, "end": int(end)},
    ]
    trades_by_request: list[list[dict[str, Any]]] = []
    try:
        for params in requests:
            response = _json_request(
                str(config["trade_endpoint"]) + "?" + urllib.parse.urlencode(params),
                retries=int(config["request_retries"]),
            )
            trades_by_request.append(
                [item for item in response if isinstance(item, dict)]
                if isinstance(response, list)
                else []
            )
    except RuntimeError as exc:
        return tokens[0], [], False, len(requests), str(exc)
    trades = [trade for response in trades_by_request for trade in response]
    points: set[tuple[int, float]] = set()
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        converted = _trade_to_probability(trade, tokens[0], tokens[1])
        if converted is None:
            continue
        timestamp, probability = converted
        if created <= timestamp < closed:
            points.add((timestamp, probability))
    history = [
        {"t": timestamp, "p": probability, "source": "data_api_trade"}
        for timestamp, probability in sorted(points)
    ]
    capped = any(
        len(response) >= int(config["trade_history_limit"])
        for response in trades_by_request
    )
    return tokens[0], history, capped, len(requests), None


def fetch_trade_histories(
    markets: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, list[dict[str, object]]], list[str], dict[str, Any]]:
    histories: dict[str, list[dict[str, object]]] = {}
    errors: list[str] = []
    capped = 0
    request_count = 0
    ineligible_bounds = 0
    with ThreadPoolExecutor(max_workers=int(config["trade_history_workers"])) as executor:
        futures = {
            executor.submit(_fetch_market_trades, market, config): str(market.get("id"))
            for market in markets
        }
        for future in as_completed(futures):
            market_id = futures[future]
            try:
                token, history, was_capped, requests_used, error = future.result()
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                errors.append(f"market {market_id}: {exc}")
                continue
            if error:
                if error == "invalid_trade_query_bounds":
                    ineligible_bounds += 1
                else:
                    errors.append(f"market {market_id}: {error}")
            if token:
                histories[token] = history
            capped += int(was_capped)
            request_count += int(requests_used)
    return histories, sorted(errors), {
        "trade_markets_queried": len(markets),
        "trade_requests": request_count,
        "trade_tokens_returned": len(histories),
        "trade_tokens_with_minimum_points": sum(
            len(history) >= int(config["min_history_points"])
            for history in histories.values()
        ),
        "trade_capped_markets": capped,
        "trade_ineligible_bounds": ineligible_bounds,
    }


def run(
    config_path: Path, refresh: bool = False, refresh_histories_only: bool = False
) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = payload["polymarket"]
    metadata_path = ROOT / config["metadata_path"]
    history_path = ROOT / config["history_path"]
    manifest_path = ROOT / config["fetch_manifest"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and history_path.exists() and not refresh and not refresh_histories_only:
        print(f"using frozen metadata {metadata_path}")
        print(f"using frozen histories {history_path}")
        return manifest_path

    started = datetime.now(timezone.utc)
    if refresh_histories_only:
        if not metadata_path.exists() or not manifest_path.exists():
            raise FileNotFoundError("history-only refresh requires frozen metadata and manifest")
        markets = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata_keys = [
            "fetch_cutoff_utc",
            "pages",
            "markets_seen",
            "duplicate_market_ids",
            "candidate_event_clusters",
            "sampling_start_date",
            "sampling_end_date",
            "sampling_windows",
            "truncated_windows",
            "window_error_count",
            "window_errors",
            "window_counts",
            "candidate_close_min_utc",
            "candidate_close_max_utc",
            "candidate_distinct_series",
            "candidate_top_series",
        ]
        metadata_stats = {
            key: previous_manifest.get(key) for key in metadata_keys
        }
    else:
        markets, metadata_stats = fetch_metadata(config)
        metadata_path.write_text(
            json.dumps(markets, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    histories, errors, history_stats = fetch_histories(markets, config)
    _write_deterministic_history(history_path, histories)
    completed = datetime.now(timezone.utc)
    history_lengths = [len(value) for value in histories.values()]
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0",
        "source": "Polymarket public read-only Gamma and CLOB APIs",
        "authenticated": False,
        "trading": False,
        "wallet": False,
        "endpoints": {
            "metadata": config["gamma_endpoint"],
            "history": config["history_endpoint"],
        },
        "documentation": {
            "index": config["docs_index"],
            "price_history": config["price_history_docs"],
            "trade_history": config["trade_history_docs"],
            "fees": config["fee_docs"],
        },
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        **metadata_stats,
        "metadata_reused": refresh_histories_only,
        "history_serialization": "gzip_mtime_0_sorted_json",
        **history_stats,
        "history_tokens_requested": len(markets),
        "history_tokens_returned": len(histories),
        "history_tokens_with_minimum_points": sum(
            length >= int(config["min_history_points"]) for length in history_lengths
        ),
        "history_points_total": sum(history_lengths),
        "errors": errors,
        "outputs": {
            str(metadata_path.relative_to(ROOT)): sha256(metadata_path),
            str(history_path.relative_to(ROOT)): sha256(history_path),
        },
        "claim_boundary": (
            "Read-only historical market probabilities from CLOB price history or public "
            "Data API trades. Historical spread and depth are not fetched and must not be "
            "described as observed."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def normalize_frozen_history(config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = payload["polymarket"]
    history_path = ROOT / config["history_path"]
    manifest_path = ROOT / config["fetch_manifest"]
    if not history_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("normalization requires frozen history and fetch manifest")
    with gzip.open(history_path, "rt", encoding="utf-8") as handle:
        histories = json.load(handle)
    _write_deterministic_history(history_path, histories)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["history_serialization"] = "gzip_mtime_0_sorted_json"
    manifest["history_normalized_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"][str(history_path.relative_to(ROOT))] = sha256(history_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def audit_truncated_windows(config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = payload["polymarket"]
    manifest_path = ROOT / config["fetch_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError("truncated-window audit requires a fetch manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = [
        month
        for month, count in manifest.get("window_counts", {}).items()
        if int(count) >= int(config["max_clusters_per_month"])
    ]
    sampling_end = _utc_timestamp(config["sampling_end_date"])
    truncated: list[str] = []
    errors: list[str] = []
    for month in sorted(candidates):
        window_start = _utc_timestamp(f"{month}-01")
        window_end = min(window_start + pd.offsets.MonthBegin(1), sampling_end)
        params = {
            "closed": "true",
            "limit": int(config["page_size"]),
            "offset": (int(config["max_pages_per_month"]) - 1)
            * int(config["page_size"]),
            "order": "closedTime",
            "ascending": "false",
            "include_tag": "true",
            "start_date_min": window_start.isoformat(),
            "start_date_max": window_end.isoformat(),
        }
        try:
            response = _json_request(
                str(config["gamma_endpoint"]) + "?" + urllib.parse.urlencode(params),
                retries=int(config["request_retries"]),
            )
        except RuntimeError as exc:
            errors.append(f"{month}: {exc}")
            continue
        page = response if isinstance(response, list) else response.get("markets", [])
        if len(page) >= int(config["page_size"]):
            truncated.append(month)
    manifest["truncated_window_names"] = truncated
    manifest["truncated_window_audit_errors"] = errors
    manifest["truncated_window_audit_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch frozen Polymarket point-in-time inputs.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "public_audit.yaml"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh-histories-only", action="store_true")
    parser.add_argument("--normalize-frozen-history", action="store_true")
    parser.add_argument("--audit-truncated-windows", action="store_true")
    args = parser.parse_args()
    if args.audit_truncated_windows:
        audit_truncated_windows(Path(args.config))
        return 0
    if args.normalize_frozen_history:
        normalize_frozen_history(Path(args.config))
        return 0
    run(
        Path(args.config),
        refresh=args.refresh,
        refresh_histories_only=args.refresh_histories_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
