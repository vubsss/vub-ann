"""Validate and time FAISS IndexFlatL2 on both assignment datasets."""

from __future__ import annotations

# Set BLAS/OpenMP limits before importing NumPy and FAISS.
import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import csv
import gc
import hashlib
import json
import platform
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import faiss
import h5py
import numpy as np
import psutil

from annbench.benchmarks.exact import check_stored_reference, time_exact_search
from annbench.data import DATASETS, load_dataset


FIELDS = [
    "run_id", "timestamp_utc", "experiment", "dataset", "source_file",
    "source_sha256", "index_family", "distance", "query_split", "query_start",
    "query_end", "reference_query_count", "base_count", "dimensions", "k",
    "timing_mode", "repetitions", "repeat_index", "elapsed_seconds", "qps",
    "median_seconds", "median_qps", "recall_at_10", "exact_vs_stored_recall",
    "stored_vs_exact_recall", "differing_id_set_queries",
    "max_ranked_squared_l2_difference", "max_stored_distance_error",
    "max_faiss_squared_l2_error", "build_seconds", "faiss_threads",
    "omp_num_threads", "openblas_num_threads", "mkl_num_threads",
    "python_version", "numpy_version", "h5py_version", "faiss_version",
    "cpu_model", "logical_cpus", "ram_bytes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _save_rows(path: Path, rows: list[dict[str, object]], dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, str]] = []
    old_fields: list[str] = []
    if path.exists():
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            old_fields = reader.fieldnames or []
            old_rows = [
                row for row in reader
                if not (row.get("experiment") == "exact_baseline" and row.get("dataset") == dataset)
            ]
    fields = list(dict.fromkeys([*old_fields, *FIELDS]))
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([*old_rows, *rows])
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "results/results.csv")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    faiss.omp_set_num_threads(1)
    names = DATASETS if args.dataset == "both" else [args.dataset]
    for name in names:
        spec = DATASETS[name]
        source_path = args.data_dir / spec.filename
        print(f"{name}: loading {source_path}", flush=True)
        data = load_dataset(source_path, spec)
        with h5py.File(source_path, "r") as source:
            stored_distances = np.asarray(source["distances"][:, :10], dtype=np.float64)

        start = perf_counter()
        index = faiss.IndexFlatL2(spec.dimensions)
        index.add(data.base)
        build_seconds = perf_counter() - start
        if index.ntotal != spec.base_count:
            raise RuntimeError("FAISS index did not receive all base vectors")
        print(f"{name}: exact index built in {build_seconds:.3f}s; checking all queries", flush=True)

        faiss_distances, exact_ids = index.search(data.queries, 10)
        check = check_stored_reference(
            data.base, data.queries, data.neighbors, stored_distances,
            exact_ids, faiss_distances, source_distance=spec.source_distance,
        )
        print(f"{name}: reference check {json.dumps(check.__dict__, sort_keys=True)}", flush=True)
        if args.validation_only:
            del index, data
            gc.collect()
            continue

        run_id = str(uuid.uuid4())
        metadata = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "exact_baseline",
            "dataset": name,
            "source_file": spec.filename,
            "source_sha256": _sha256(source_path),
            "index_family": "IndexFlatL2",
            "distance": "squared_euclidean",
            "query_split": "report",
            "query_start": spec.tune_count,
            "query_end": spec.query_count,
            "reference_query_count": check.query_count,
            "base_count": spec.base_count,
            "dimensions": spec.dimensions,
            "k": 10,
            "recall_at_10": 1.0,
            "exact_vs_stored_recall": check.exact_vs_stored_recall,
            "stored_vs_exact_recall": check.stored_vs_exact_recall,
            "differing_id_set_queries": check.differing_id_set_queries,
            "max_ranked_squared_l2_difference": check.max_ranked_squared_l2_difference,
            "max_stored_distance_error": check.max_stored_distance_error,
            "max_faiss_squared_l2_error": check.max_faiss_squared_l2_error,
            "build_seconds": build_seconds,
            "faiss_threads": faiss.omp_get_max_threads(),
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"],
            "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "h5py_version": h5py.__version__,
            "faiss_version": faiss.__version__,
            "cpu_model": _cpu_model(),
            "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
        }

        rows = []
        for mode in ("single", "batch"):
            def on_repeat(repetition: int, elapsed: float) -> None:
                print(
                    f"{name} {mode} repeat {repetition}/{args.repetitions}: "
                    f"{elapsed:.3f}s ({len(data.report_queries) / elapsed:.1f} QPS)",
                    flush=True,
                )

            times = time_exact_search(
                index, data.report_queries, mode=mode,
                repetitions=args.repetitions, on_repeat=on_repeat,
            )
            median_seconds = statistics.median(times)
            for repeat_index, elapsed in enumerate(times, start=1):
                rows.append({
                    **metadata,
                    "timing_mode": mode,
                    "repetitions": args.repetitions,
                    "repeat_index": repeat_index,
                    "elapsed_seconds": elapsed,
                    "qps": len(data.report_queries) / elapsed,
                    "median_seconds": median_seconds,
                    "median_qps": len(data.report_queries) / median_seconds,
                })
        _save_rows(args.output, rows, name)
        print(f"{name}: saved {len(rows)} rows to {args.output}", flush=True)
        del index, data, exact_ids, faiss_distances
        gc.collect()


if __name__ == "__main__":
    main()
