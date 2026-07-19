from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    date_range,
    load_config,
    month_range,
    resolve_root_path,
    sha256,
    write_json,
)
from finauth_audit.generators.fetch_binance_depth_v03 import DownloadSpec, _fetch_one


VERIFIED_STATUSES = {"CACHED_VERIFIED", "DOWNLOADED_VERIFIED"}


def assigned_symbol_for_date(current: date, symbols: list[str]) -> str:
    """Return the frozen calendar-date assignment, independent of exclusions."""

    if not symbols:
        raise ValueError("at least one Binance symbol is required")
    return str(symbols[current.toordinal() % len(symbols)])


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], date, date, list[str]]:
    if str(config.get("version")) != "0.6.0":
        raise RuntimeError("real-agent source fetcher requires frozen v0.6.0 config")
    source = config["binance"]
    if source.get("symbol_assignment") != "calendar_date_ordinal_modulo_symbol_count":
        raise RuntimeError("unexpected v0.6 symbol-assignment contract")
    if source.get("fetch_assignment") != (
        "one assigned daily depth archive per date plus monthly klines for every symbol"
    ):
        raise RuntimeError("unexpected v0.6 source-fetch contract")
    start = date.fromisoformat(str(source["start_date"]))
    end = date.fromisoformat(str(source["end_date"]))
    prior_end = date.fromisoformat(str(source["prior_inspected_period_end"]))
    if start > end:
        raise RuntimeError("real-agent source period is empty")
    if start <= prior_end:
        raise RuntimeError("real-agent v0.6 source overlaps the inspected v0.5 period")
    symbols = [str(value) for value in source["symbols"]]
    if len(symbols) != 5 or len(set(symbols)) != len(symbols):
        raise RuntimeError("real-agent v0.6 requires five unique Binance symbols")
    return source, start, end, symbols


def _specs(config: dict[str, Any]) -> list[DownloadSpec]:
    source, start, end, symbols = _validate_config(config)
    base = str(source["base_url"]).rstrip("/")
    raw_dir = resolve_root_path(source["raw_dir"])
    specs: list[DownloadSpec] = []

    for current in date_range(start, end):
        symbol = assigned_symbol_for_date(current, symbols)
        filename = f"{symbol}-bookDepth-{current.isoformat()}.zip"
        specs.append(
            DownloadSpec(
                kind="bookDepth",
                symbol=symbol,
                period=current.isoformat(),
                url=f"{base}/daily/bookDepth/{symbol}/{filename}",
                target=raw_dir / "bookDepth" / symbol / filename,
            )
        )

    for symbol in symbols:
        for year, month in month_range(start, end):
            filename = f"{symbol}-1m-{year:04d}-{month:02d}.zip"
            specs.append(
                DownloadSpec(
                    kind="klines_1m",
                    symbol=symbol,
                    period=f"{year:04d}-{month:02d}",
                    url=f"{base}/monthly/klines/{symbol}/1m/{filename}",
                    target=raw_dir / "klines_1m" / symbol / filename,
                )
            )
    return specs


def run(config_path: Path, workers: int = 8, limit: int | None = None) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    source, start, end, symbols = _validate_config(config)
    full_specs = _specs(config)
    specs = full_specs if limit is None else full_specs[:limit]

    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_fetch_one, spec): spec for spec in specs}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            records.append(future.result())
            if index % 100 == 0 or index == len(specs):
                print(f"fetched {index}/{len(specs)}")

    records.sort(key=lambda item: (str(item["kind"]), str(item["period"]), str(item["symbol"])))
    status_counts = Counter(str(record["status"]) for record in records)
    depth_specs = [spec for spec in full_specs if spec.kind == "bookDepth"]
    kline_specs = [spec for spec in full_specs if spec.kind == "klines_1m"]
    verified_count = sum(
        int(str(record["status"]) in VERIFIED_STATUSES) for record in records
    )

    manifest_path = resolve_root_path(source["source_manifest"])
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.6.0",
        "source": source["source_name"],
        "source_base_url": source["base_url"],
        "license_url": source["license_url"],
        "config_path": _display_path(config_path),
        "config_sha256": sha256(config_path),
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": symbols,
        "symbol_assignment": source["symbol_assignment"],
        "fetch_assignment": source["fetch_assignment"],
        "requested_files": len(specs),
        "complete_registered_request": len(specs) == len(full_specs),
        "registered_depth_files": len(depth_specs),
        "registered_monthly_kline_files": len(kline_specs),
        "checksum_verified_files": verified_count,
        "status_counts": dict(sorted(status_counts.items())),
        "records": records,
        "source_data_only": True,
        "model_proposals_generated": False,
        "outcome_metrics_computed": False,
        "claim_boundary": (
            "Official Binance source acquisition and checksum verification only; "
            "one calendar-assigned depth archive is requested per date and monthly "
            "one-minute bars are requested for every frozen symbol. No prompt, model "
            "proposal, outcome label, or authorization metric is computed."
        ),
    }
    write_json(manifest_path, manifest)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and checksum the frozen real-agent v0.6 Binance source."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "real_agent_v06.yaml")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(Path(args.config), workers=args.workers, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
