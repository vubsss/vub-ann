"""Print validated dataset shapes and reproducibility metadata."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import faiss

from annbench.data import DATASETS, load_dataset


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    args = parser.parse_args()

    print(json.dumps({
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "h5py": _version("h5py"),
        "faiss_cpu": _version("faiss-cpu"),
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "planned_search_threads": 1,
        "faiss_max_threads_current": faiss.omp_get_max_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "distance_used": "euclidean",
    }, sort_keys=True))

    names = DATASETS if args.dataset == "both" else [args.dataset]
    for name in names:
        spec = DATASETS[name]
        data = load_dataset(args.data_dir / spec.filename, spec)
        print(json.dumps({
            "dataset": name,
            "base_shape": data.base.shape,
            "query_shape": data.queries.shape,
            "neighbors_shape": data.neighbors.shape,
            "base_dtype": str(data.base.dtype),
            "query_dtype": str(data.queries.dtype),
            "neighbor_id_range": [int(data.neighbors.min()), int(data.neighbors.max())],
            "tune_query_rows": [0, spec.tune_count],
            "report_query_rows": [spec.tune_count, spec.query_count],
            "zero_base_vectors": data.zero_base_vectors,
            "zero_query_vectors": data.zero_query_vectors,
            "normalized": spec.normalize,
            "distance_used": "euclidean",
        }, sort_keys=True))
        del data
        gc.collect()


if __name__ == "__main__":
    main()
