"""Run one ANN setting on tuning queries or a frozen setting on held-out queries."""

from __future__ import annotations

# Set thread limits before NumPy or FAISS are imported.
import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable

from annbench.benchmarks.runner import BenchmarkConfig, benchmark
from annbench.data import DATASETS, load_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_rows(
    path: Path, rows: list[dict[str, object]], *,
    replace_filter: Callable[[dict[str, str]], bool] | None = None,
) -> None:
    """Keep exact-baseline and earlier benchmark runs in one parseable CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, str]] = []
    old_fields: list[str] = []
    if path.exists():
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            old_fields = reader.fieldnames or []
            old_rows = [row for row in reader if replace_filter is None or not replace_filter(row)]
    fields = list(dict.fromkeys([*old_fields, *(key for row in rows for key in row)]))
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([*old_rows, *rows])
    temp_path.replace(path)


def load_selection(path: Path) -> tuple[BenchmarkConfig, str, str]:
    selection = json.loads(path.read_text())
    if selection.get("query_split") != "tune" or not selection.get("tune_run_id"):
        raise ValueError("selection file must identify a completed tuning run")
    config = BenchmarkConfig(**selection["config"])
    config.validate()
    return config, selection["tune_run_id"], selection["source_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="JSON setting to measure on tuning queries")
    source.add_argument("--selection-file", type=Path, help="frozen setting from a tuning run")
    parser.add_argument("--selection-out", type=Path, help="write tuning setting for later held-out run")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "results/results.csv")
    args = parser.parse_args()
    if args.selection_file and args.selection_out:
        parser.error("--selection-out is only valid with --config")

    if args.config:
        config = BenchmarkConfig.from_file(args.config)
        split = "tune"
        selected_tune_run_id = ""
        selected_source_hash = ""
    else:
        config, selected_tune_run_id, selected_source_hash = load_selection(args.selection_file)
        split = "report"

    source_path = args.data_dir / DATASETS[config.dataset].filename
    source_hash = sha256(source_path)
    if selected_source_hash and selected_source_hash != source_hash:
        raise ValueError("dataset checksum differs from the tuning run")
    print(f"Loading {config.dataset} ({config.mode}, {split}) from {source_path}", flush=True)
    data = load_dataset(source_path, DATASETS[config.dataset])
    rows = benchmark(config, data, split=split, source_sha256=source_hash)
    for row in rows:
        row["selected_tune_run_id"] = selected_tune_run_id
    save_rows(args.output, rows)
    print(
        f"Saved {len(rows)} {split} rows to {args.output}; "
        f"recall@10={rows[0]['recall_at_10']:.4f}, median QPS={rows[0]['median_qps']:.1f}",
        flush=True,
    )

    if args.selection_out:
        args.selection_out.parent.mkdir(parents=True, exist_ok=True)
        selection = {
            "query_split": "tune",
            "tune_run_id": rows[0]["run_id"],
            "source_sha256": source_hash,
            "config": asdict(config),
        }
        args.selection_out.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        print(f"Saved frozen setting to {args.selection_out}", flush=True)


if __name__ == "__main__":
    main()
