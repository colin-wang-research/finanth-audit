from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "finauth-audit-round75-paper-test-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_test_cluster_ids(path: Path) -> set[str]:
    """Inspect only split and event-cluster columns."""
    cluster_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        split_index = header.index("split")
        cluster_index = header.index("event_cluster_id")
        for row in reader:
            if row[split_index] == "test":
                cluster_ids.add(row[cluster_index])
    return cluster_ids


def assign_clusters(cluster_ids: set[str]) -> tuple[list[str], list[str]]:
    if len(cluster_ids) < 2 or len(cluster_ids) % 2:
        raise ValueError("paper/community split requires an even cluster count")
    ordered = sorted(
        cluster_ids,
        key=lambda cluster_id: hashlib.sha256(
            f"{NAMESPACE}|{cluster_id}".encode("utf-8")
        ).hexdigest(),
    )
    midpoint = len(ordered) // 2
    return sorted(ordered[:midpoint]), sorted(ordered[midpoint:])


def _write_cluster_list(path: Path, cluster_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_cluster_id"])
        writer.writerows([[cluster_id] for cluster_id in cluster_ids])


def stream_copy_splits(
    source: Path,
    outputs: dict[str, Path],
    paper_clusters: set[str],
    community_clusters: set[str],
) -> dict[str, int]:
    """Copy raw CSV rows while branching only on split and cluster ID."""
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen partition file: {path}")
    handles = {
        name: path.open("w", encoding="utf-8", newline="")
        for name, path in outputs.items()
    }
    counts = {name: 0 for name in outputs}
    try:
        writers = {name: csv.writer(handle) for name, handle in handles.items()}
        with source.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.reader(input_handle)
            header = next(reader)
            split_index = header.index("split")
            cluster_index = header.index("event_cluster_id")
            for writer in writers.values():
                writer.writerow(header)
            for row in reader:
                split = row[split_index]
                cluster_id = row[cluster_index]
                destination: str | None = None
                if split == "validation" and "validation" in writers:
                    destination = "validation"
                elif split == "train" and "train" in writers:
                    destination = "train"
                elif split == "test" and cluster_id in paper_clusters:
                    destination = "paper_test"
                elif split == "test" and cluster_id in community_clusters:
                    destination = "community_hidden"
                if destination is not None:
                    writers[destination].writerow(row)
                    counts[destination] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def build_partition() -> Path:
    controlled_source = ROOT / "data" / "controlled_core" / "main.csv"
    provenance_source = ROOT / "data" / "provenance_laundering" / "main.csv"
    partition_dir = ROOT / "manifests" / "paper_test_partition"
    manifest_path = partition_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "paper-test partition already exists; regeneration is prohibited"
        )

    controlled_clusters = collect_test_cluster_ids(controlled_source)
    provenance_clusters = collect_test_cluster_ids(provenance_source)
    if controlled_clusters != provenance_clusters:
        raise ValueError("controlled and provenance test cluster IDs differ")
    paper_clusters, community_clusters = assign_clusters(controlled_clusters)
    if set(paper_clusters) & set(community_clusters):
        raise AssertionError("paper and community cluster sets overlap")

    paper_list = partition_dir / "paper_test_clusters.csv"
    community_list = partition_dir / "community_hidden_clusters.csv"
    _write_cluster_list(paper_list, paper_clusters)
    _write_cluster_list(community_list, community_clusters)

    controlled_outputs = {
        "validation": ROOT / "data" / "paper_test" / "controlled_validation.csv",
        "paper_test": ROOT / "data" / "paper_test" / "controlled.csv",
        "community_hidden": ROOT
        / "sealed"
        / "community_hidden"
        / "controlled.csv",
    }
    provenance_outputs = {
        "train": ROOT / "data" / "paper_test" / "provenance_train.csv",
        "validation": ROOT / "data" / "paper_test" / "provenance_validation.csv",
        "paper_test": ROOT / "data" / "paper_test" / "provenance.csv",
        "community_hidden": ROOT
        / "sealed"
        / "community_hidden"
        / "provenance.csv",
    }
    controlled_counts = stream_copy_splits(
        controlled_source,
        controlled_outputs,
        set(paper_clusters),
        set(community_clusters),
    )
    provenance_counts = stream_copy_splits(
        provenance_source,
        provenance_outputs,
        set(paper_clusters),
        set(community_clusters),
    )

    expected_test_rows = len(paper_clusters) * 4
    if controlled_counts != {
        "validation": 40_000,
        "paper_test": expected_test_rows,
        "community_hidden": expected_test_rows,
    }:
        raise ValueError(f"unexpected controlled partition counts: {controlled_counts}")
    if provenance_counts != {
        "train": 120_000,
        "validation": 40_000,
        "paper_test": expected_test_rows,
        "community_hidden": expected_test_rows,
    }:
        raise ValueError(f"unexpected provenance partition counts: {provenance_counts}")

    for path in (
        controlled_outputs["community_hidden"],
        provenance_outputs["community_hidden"],
    ):
        os.chmod(path, 0o400)
    for path in (
        paper_list,
        community_list,
        *controlled_outputs.values(),
        *provenance_outputs.values(),
    ):
        if "community_hidden" not in str(path):
            os.chmod(path, 0o444)

    files = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            paper_list,
            community_list,
            *controlled_outputs.values(),
            *provenance_outputs.values(),
        )
    }
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.3.0-round75-paper-test-partition",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "namespace": NAMESPACE,
        "partition_fields_inspected": ["event_cluster_id", "split"],
        "outcome_fields_inspected": False,
        "opaque_rows_copied_without_semantic_field_access": True,
        "source_files": {
            str(controlled_source.relative_to(ROOT)): sha256(controlled_source),
            str(provenance_source.relative_to(ROOT)): sha256(provenance_source),
        },
        "paper_test_clusters": len(paper_clusters),
        "community_hidden_clusters": len(community_clusters),
        "paper_test_rows_per_layer": expected_test_rows,
        "community_hidden_rows_per_layer": expected_test_rows,
        "controlled_counts": controlled_counts,
        "provenance_counts": provenance_counts,
        "files": files,
        "community_hidden_outcomes_evaluated": False,
        "paper_test_outcomes_evaluated": False,
        "partition_code_sha256": sha256(Path(__file__)),
        "claim_boundary": (
            "The deterministic partition inspects only split and event-cluster IDs. "
            "Other fields are copied as opaque CSV cells and are not used for assignment."
        ),
    }
    partition_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(manifest_path, 0o444)
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the immutable paper/community test partition."
    )
    parser.parse_args()
    build_partition()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
