from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from finauth_audit.generators.external_orderbook_v03 import (
    ROOT,
    date_range,
    load_config,
    month_range,
    resolve_root_path,
    sha256,
    write_json,
)


USER_AGENT = "FinAuth-Audit/0.3 external-orderbook research"


@dataclass(frozen=True)
class DownloadSpec:
    kind: str
    symbol: str
    period: str
    url: str
    target: Path


def _specs(config: dict[str, Any]) -> list[DownloadSpec]:
    source = config["binance"]
    start = date.fromisoformat(str(source["start_date"]))
    end = date.fromisoformat(str(source["end_date"]))
    base = str(source["base_url"]).rstrip("/")
    raw_dir = resolve_root_path(source["raw_dir"])
    specs: list[DownloadSpec] = []
    for symbol in source["symbols"]:
        for current in date_range(start, end):
            filename = f"{symbol}-bookDepth-{current.isoformat()}.zip"
            url = f"{base}/daily/bookDepth/{symbol}/{filename}"
            specs.append(
                DownloadSpec(
                    kind="bookDepth",
                    symbol=str(symbol),
                    period=current.isoformat(),
                    url=url,
                    target=raw_dir / "bookDepth" / str(symbol) / filename,
                )
            )
        for year, month in month_range(start, end):
            filename = f"{symbol}-1m-{year:04d}-{month:02d}.zip"
            url = f"{base}/monthly/klines/{symbol}/1m/{filename}"
            specs.append(
                DownloadSpec(
                    kind="klines_1m",
                    symbol=str(symbol),
                    period=f"{year:04d}-{month:02d}",
                    url=url,
                    target=raw_dir / "klines_1m" / str(symbol) / filename,
                )
            )
    return specs


def _read_url(url: str, retries: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(min(8.0, 0.5 * 2**attempt))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


def _fetch_one(spec: DownloadSpec) -> dict[str, object]:
    checksum_url = spec.url + ".CHECKSUM"
    try:
        checksum_payload = _read_url(checksum_url).decode("utf-8").strip()
    except FileNotFoundError:
        return {
            "kind": spec.kind,
            "symbol": spec.symbol,
            "period": spec.period,
            "url": spec.url,
            "status": "MISSING_SOURCE",
        }
    parts = checksum_payload.split()
    if len(parts) < 2 or len(parts[0]) != 64:
        raise ValueError(f"invalid checksum payload: {checksum_url}: {checksum_payload!r}")
    expected = parts[0].lower()
    spec.target.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = spec.target.with_suffix(spec.target.suffix + ".CHECKSUM")
    if spec.target.exists() and sha256(spec.target) == expected:
        status = "CACHED_VERIFIED"
    else:
        payload = _read_url(spec.url)
        temporary = spec.target.with_suffix(spec.target.suffix + ".part")
        temporary.write_bytes(payload)
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {spec.url}: expected={expected} actual={actual}"
            )
        os.replace(temporary, spec.target)
        status = "DOWNLOADED_VERIFIED"
    checksum_path.write_text(checksum_payload + "\n", encoding="utf-8")
    return {
        "kind": spec.kind,
        "symbol": spec.symbol,
        "period": spec.period,
        "url": spec.url,
        "checksum_url": checksum_url,
        "path": str(spec.target.relative_to(ROOT)),
        "bytes": spec.target.stat().st_size,
        "sha256": expected,
        "status": status,
    }


def run(config_path: Path, workers: int = 8, limit: int | None = None) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    specs = _specs(config)
    if limit is not None:
        specs = specs[:limit]
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(_fetch_one, spec): spec for spec in specs}
        for index, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            record = future.result()
            records.append(record)
            if index % 100 == 0 or index == len(specs):
                print(f"fetched {index}/{len(specs)}")
    records.sort(key=lambda item: (str(item["kind"]), str(item["symbol"]), str(item["period"])))
    source_manifest = resolve_root_path(config["binance"]["source_manifest"])
    status_counts = pd.Series([record["status"] for record in records]).value_counts().to_dict()
    manifest = {
        "project": "FinAuth-Audit",
        "version": str(config.get("version", "0.3.0")),
        "source": config["binance"]["source_name"],
        "source_base_url": config["binance"]["base_url"],
        "license_url": config["binance"]["license_url"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "requested_files": len(specs),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "records": records,
        "outcome_metrics_computed": False,
        "claim_boundary": (
            "Official public source acquisition and checksum verification only; "
            "no authorization outcome metric is computed."
        ),
    }
    write_json(source_manifest, manifest)
    print(source_manifest)
    return source_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify Binance public order-book inputs.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "external_orderbook_v03.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(Path(args.config), workers=args.workers, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
