"""Configurable, single-query FAISS benchmark measurements."""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import h5py
import numpy as np
import psutil

from annbench.benchmarks.exact import recall_at_k, squared_l2_for_ids
from annbench.data import DATASETS, LoadedDataset


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    index_family: str
    build_settings: dict[str, int]
    search_settings: dict[str, int]
    seed: int = 1234
    mode: str = "quick"
    repetitions: int = 3
    k: int = 10
    quick_base_count: int = 20_000
    quick_query_count: int = 50
    tune_query_count: int | None = None

    @classmethod
    def from_file(cls, path: Path) -> BenchmarkConfig:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a JSON object")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {self.dataset}")
        if self.index_family not in {"hnsw", "ivf_flat", "ivf_pq"}:
            raise ValueError(f"unknown index family: {self.index_family}")
        if self.mode not in {"quick", "full"}:
            raise ValueError("mode must be quick or full")
        if self.seed < 0 or self.repetitions < 1 or self.k != 10:
            raise ValueError("seed must be nonnegative, repetitions positive, and k must be 10")
        if not 10 <= self.quick_base_count <= DATASETS[self.dataset].base_count:
            raise ValueError("quick_base_count must be between 10 and the dataset size")
        if not 1 <= self.quick_query_count <= DATASETS[self.dataset].tune_count:
            raise ValueError("quick_query_count must be between 1 and 5000")
        if self.tune_query_count is not None and not 1 <= self.tune_query_count <= DATASETS[self.dataset].tune_count:
            raise ValueError("tune_query_count must be between 1 and 5000")
        build_keys = {
            "hnsw": {"M", "efConstruction"},
            "ivf_flat": {"nlist"},
            "ivf_pq": {"nlist", "m", "nbits"},
        }[self.index_family]
        search_keys = {"hnsw": {"efSearch"}, "ivf_flat": {"nprobe"}, "ivf_pq": {"nprobe"}}[self.index_family]
        if set(self.build_settings) != build_keys or set(self.search_settings) != search_keys:
            raise ValueError(f"expected build keys {sorted(build_keys)} and search keys {sorted(search_keys)}")
        if any(not isinstance(v, int) or v < 1 for v in (*self.build_settings.values(), *self.search_settings.values())):
            raise ValueError("all build and search settings must be positive integers")
        if self.index_family == "ivf_pq":
            if DATASETS[self.dataset].dimensions % self.build_settings["m"]:
                raise ValueError("PQ m must divide the vector dimension")
            if not 1 <= self.build_settings["nbits"] <= 8:
                raise ValueError("PQ nbits must be between 1 and 8")
        if self.index_family != "hnsw":
            base_count = self.quick_base_count if self.mode == "quick" else DATASETS[self.dataset].base_count
            if self.build_settings["nlist"] > base_count:
                raise ValueError("nlist cannot exceed the base count")
            if self.search_settings["nprobe"] > self.build_settings["nlist"]:
                raise ValueError("nprobe cannot exceed nlist")


def build_index(config: BenchmarkConfig, base: np.ndarray) -> tuple[faiss.Index, float]:
    """Build and train one index; the time includes training and vector insertion."""
    build_threads = 4 if config.mode == "full" and config.index_family == "hnsw" else 1
    faiss.omp_set_num_threads(build_threads)
    np.random.seed(config.seed)
    dimension = base.shape[1]
    start = perf_counter()
    if config.index_family == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, config.build_settings["M"])
        index.hnsw.efConstruction = config.build_settings["efConstruction"]
    elif config.index_family == "ivf_flat":
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dimension), dimension, config.build_settings["nlist"], faiss.METRIC_L2,
        )
        index.cp.seed = config.seed
        if config.mode == "full":
            index.cp.max_points_per_centroid = 50
    else:
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension), dimension, config.build_settings["nlist"],
            config.build_settings["m"], config.build_settings["nbits"],
        )
        index.cp.seed = config.seed
        index.pq.cp.seed = config.seed
        if config.mode == "full":
            index.cp.max_points_per_centroid = 50
            index.pq.cp.max_points_per_centroid = 100
    try:
        if not index.is_trained:
            index.train(base)
        index.add(base)
        elapsed = perf_counter() - start
    finally:
        faiss.omp_set_num_threads(1)
    if index.ntotal != len(base):
        raise RuntimeError("index did not receive all base vectors")
    return index, elapsed


def set_search_settings(index: faiss.Index, config: BenchmarkConfig) -> None:
    if config.index_family == "hnsw":
        index.hnsw.efSearch = config.search_settings["efSearch"]
    else:
        index.nprobe = config.search_settings["nprobe"]


def query_slice(data: LoadedDataset, config: BenchmarkConfig, split: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return original query row bounds, preserving the tuning/held-out boundary."""
    if split not in {"tune", "report"}:
        raise ValueError("split must be tune or report")
    start = 0 if split == "tune" else data.spec.tune_count
    if config.mode == "quick":
        count = config.quick_query_count
    elif split == "tune" and config.tune_query_count is not None:
        count = config.tune_query_count
    else:
        count = data.spec.tune_count
    end = start + count
    return data.queries[start:end], data.neighbors[start:end, :config.k], start, end


def benchmark(
    config: BenchmarkConfig, data: LoadedDataset, *, split: str, source_sha256: str,
) -> list[dict[str, Any]]:
    """Measure one fixed setting. No configuration choice uses report queries."""
    config.validate()
    if data.spec.name != config.dataset:
        raise ValueError("loaded dataset does not match configuration")
    faiss.omp_set_num_threads(1)
    base = data.base[:config.quick_base_count] if config.mode == "quick" else data.base
    queries, stored_ids, _, _ = query_slice(data, config, split)
    cutoffs = reference_cutoffs(base, queries, stored_ids, config)
    index, build_seconds = build_index(config, base)
    index_size_bytes = int(faiss.serialize_index(index).nbytes)
    return benchmark_prebuilt(
        config, data, split=split, source_sha256=source_sha256,
        index=index, build_seconds=build_seconds, index_size_bytes=index_size_bytes,
        cutoffs=cutoffs,
    )


def reference_cutoffs(
    base: np.ndarray, queries: np.ndarray, stored_ids: np.ndarray, config: BenchmarkConfig,
) -> np.ndarray:
    """Get exact rank-k distances for this base, once for an entire sweep."""
    if config.mode == "quick":
        reference = faiss.IndexFlatL2(base.shape[1])
        reference.add(base)
        _, exact_ids = reference.search(queries, config.k)
        cutoffs = squared_l2_for_ids(base, queries, exact_ids).max(axis=1)
    else:
        # Ticket 02 validated the stored top-10 against exact search on all queries.
        cutoffs = squared_l2_for_ids(base, queries, stored_ids).max(axis=1)
    return cutoffs


def benchmark_prebuilt(
    config: BenchmarkConfig, data: LoadedDataset, *, split: str, source_sha256: str,
    index: faiss.Index, build_seconds: float, index_size_bytes: int,
    cutoffs: np.ndarray,
) -> list[dict[str, Any]]:
    """Measure a search setting using an index built once for its build settings."""
    config.validate()
    if data.spec.name != config.dataset:
        raise ValueError("loaded dataset does not match configuration")
    faiss.omp_set_num_threads(1)
    base = data.base[:config.quick_base_count] if config.mode == "quick" else data.base
    queries, _, query_start, query_end = query_slice(data, config, split)
    if len(cutoffs) != len(queries) or index.ntotal != len(base):
        raise ValueError("reference cutoffs or prebuilt index do not match this run")
    set_search_settings(index, config)
    _, found_ids = index.search(queries, config.k)
    found_true_distances = squared_l2_for_ids(base, queries, found_ids)
    recall = recall_at_k(found_ids, found_true_distances, cutoffs, k=config.k)

    for query in queries[:min(10, len(queries))]:
        index.search(query[None, :], config.k)
    elapsed_times: list[float] = []
    for _ in range(config.repetitions):
        start = perf_counter()
        for query in queries:
            index.search(query[None, :], config.k)
        elapsed_times.append(perf_counter() - start)
    median_seconds = statistics.median(elapsed_times)
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    else:
        cpu_model = platform.processor() or "unknown"
    metadata = {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "ann_tuning" if split == "tune" else "ann_final",
        "dataset": config.dataset,
        "source_file": DATASETS[config.dataset].filename,
        "source_sha256": source_sha256,
        "index_family": config.index_family,
        "distance": "squared_euclidean",
        "query_split": split,
        "query_start": query_start,
        "query_end": query_end,
        "reference_query_count": len(queries),
        "base_count": len(base),
        "dimensions": base.shape[1],
        "k": config.k,
        "timing_mode": "single",
        "repetitions": config.repetitions,
        "recall_at_10": recall,
        "build_seconds": build_seconds,
        "index_size_bytes": index_size_bytes,
        "bytes_per_vector": index_size_bytes / len(base),
        "benchmark_mode": config.mode,
        "build_settings": json.dumps(config.build_settings, sort_keys=True),
        "search_settings": json.dumps(config.search_settings, sort_keys=True),
        "seed": config.seed,
        "build_threads": 4 if config.mode == "full" and config.index_family == "hnsw" else 1,
        "ivf_train_max_points_per_centroid": 50 if config.mode == "full" and config.index_family != "hnsw" else "",
        "pq_train_max_points_per_centroid": 100 if config.mode == "full" and config.index_family == "ivf_pq" else "",
        "config_json": json.dumps(asdict(config), sort_keys=True),
        "faiss_threads": faiss.omp_get_max_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "faiss_version": faiss.__version__,
        "cpu_model": cpu_model,
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
    }
    return [
        {
            **metadata,
            "repeat_index": repetition,
            "elapsed_seconds": elapsed,
            "qps": len(queries) / elapsed,
            "median_seconds": median_seconds,
            "median_qps": len(queries) / median_seconds,
        }
        for repetition, elapsed in enumerate(elapsed_times, start=1)
    ]
