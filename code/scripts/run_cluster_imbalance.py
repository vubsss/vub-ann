"""Run the preregistered balanced-control and cluster-imbalance experiment."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
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
import numpy as np
import psutil

from annbench.benchmarks.exact import recall_per_query, squared_l2_for_ids
from annbench.experiments.cluster_imbalance import (
    BASE_COUNT, CENTRE_RADIUS, CLUSTER_STD, CLUSTERS, DIMENSIONS, QUERIES_PER_CLUSTER,
    RATIOS, SEED, make_base, make_shared_geometry,
)
from scripts.run_benchmark import save_rows


SETTINGS = {
    "hnsw": ({"M": 16, "efConstruction": 100}, {"efSearch": 32}),
    "ivf_flat": ({"nlist": 64}, {"nprobe": 4}),
}
K = 10
REPETITIONS = 3
TIMING_LAPS = 20


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_index(family: str, base: np.ndarray) -> tuple[faiss.Index, float]:
    """Build with one thread and a fixed IVF training seed."""
    faiss.omp_set_num_threads(1)
    start = perf_counter()
    if family == "hnsw":
        index = faiss.IndexHNSWFlat(DIMENSIONS, SETTINGS[family][0]["M"])
        index.hnsw.efConstruction = SETTINGS[family][0]["efConstruction"]
        index.hnsw.efSearch = SETTINGS[family][1]["efSearch"]
    elif family == "ivf_flat":
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(DIMENSIONS), DIMENSIONS,
            SETTINGS[family][0]["nlist"], faiss.METRIC_L2,
        )
        index.cp.seed = SEED
        index.cp.max_points_per_centroid = 100
        index.train(base)
        index.nprobe = SETTINGS[family][1]["nprobe"]
    else:
        raise ValueError(f"unknown family: {family}")
    index.add(base)
    if index.ntotal != BASE_COUNT:
        raise RuntimeError("incomplete synthetic index")
    return index, perf_counter() - start


def measure(
    family: str, ratio: int, base: np.ndarray, queries: np.ndarray,
    query_labels: np.ndarray, counts: np.ndarray, exact_ids: np.ndarray,
    centres: np.ndarray,
) -> list[dict[str, object]]:
    """Measure recall once, then three single-query timing repetitions."""
    faiss.omp_set_num_threads(1)
    exact_distances = squared_l2_for_ids(base, queries, exact_ids)
    cutoff = exact_distances.max(axis=1)
    if not np.all(recall_per_query(exact_ids, exact_distances, cutoff, k=K) == 1):
        raise RuntimeError("exact reference failed its own recall check")
    index, build_seconds = build_index(family, base)
    index_size_bytes = int(faiss.serialize_index(index).nbytes)
    found_ids = np.empty((len(queries), K), dtype=np.int64)
    for i, query in enumerate(queries):
        found_ids[i] = index.search(query[None, :], K)[1][0]
    found_distances = squared_l2_for_ids(base, queries, found_ids)
    per_query = recall_per_query(found_ids, found_distances, cutoff, k=K)
    centroid_counts_json = ""
    rare_probed_candidates = common_probed_candidates = ""
    if family == "ivf_flat":
        learned = index.quantizer.reconstruct_n(0, SETTINGS[family][0]["nlist"])
        differences = learned[:, None, :].astype(np.float64) - centres[None, :, :]
        nearest = np.einsum("ncd,ncd->nc", differences, differences).argmin(axis=1)
        centroid_counts_json = json.dumps(np.bincount(nearest, minlength=CLUSTERS).tolist())
        _, probed = index.quantizer.search(queries, SETTINGS[family][1]["nprobe"])
        sizes = np.array([index.invlists.list_size(i) for i in range(index.nlist)])
        candidates = sizes[probed].sum(axis=1)
        rare_probed_candidates = float(candidates[query_labels < 4].mean())
        common_probed_candidates = float(candidates[query_labels >= 12].mean())
    for query in queries[:10]:
        index.search(query[None, :], K)
    elapsed = []
    for _ in range(REPETITIONS):
        start = perf_counter()
        for _ in range(TIMING_LAPS):
            for query in queries:
                index.search(query[None, :], K)
        elapsed.append(perf_counter() - start)
    median_seconds = statistics.median(elapsed)
    run_id = str(uuid.uuid4())
    config = {
        "seed": SEED, "base_count": BASE_COUNT, "dimensions": DIMENSIONS,
        "clusters": CLUSTERS, "queries_per_cluster": QUERIES_PER_CLUSTER,
        "centre_radius": CENTRE_RADIUS, "cluster_std": CLUSTER_STD,
        "population_ratio": ratio, "k": K, "repetitions": REPETITIONS,
        "timing_laps": TIMING_LAPS,
        "index_family": family,
        "build_settings": SETTINGS[family][0],
        "search_settings": SETTINGS[family][1],
    }
    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "synthetic_cluster_imbalance",
        "dataset": "synthetic_clusters",
        "source_file": "seeded_generator",
        "source_sha256": digest(base),
        "index_family": family,
        "distance": "squared_euclidean",
        "query_split": "synthetic_test",
        "query_start": 0, "query_end": len(queries),
        "reference_query_count": len(queries),
        "base_count": BASE_COUNT, "dimensions": DIMENSIONS, "k": K,
        "timing_mode": "single", "repetitions": REPETITIONS,
        "timed_query_count": len(queries) * TIMING_LAPS,
        "recall_at_10": float(per_query.mean()),
        "rare_quartile_recall_at_10": float(per_query[query_labels < 4].mean()),
        "common_quartile_recall_at_10": float(per_query[query_labels >= 12].mean()),
        "ivf_centroid_counts_json": centroid_counts_json,
        "rare_probed_candidates": rare_probed_candidates,
        "common_probed_candidates": common_probed_candidates,
        "build_seconds": build_seconds,
        "index_size_bytes": index_size_bytes,
        "bytes_per_vector": index_size_bytes / BASE_COUNT,
        "benchmark_mode": "synthetic",
        "build_settings": json.dumps(SETTINGS[family][0], sort_keys=True),
        "search_settings": json.dumps(SETTINGS[family][1], sort_keys=True),
        "seed": SEED, "build_threads": 1,
        "ivf_train_max_points_per_centroid": 100 if family == "ivf_flat" else "",
        "config_json": json.dumps(config, sort_keys=True),
        "population_ratio": ratio,
        "cluster_counts_json": json.dumps(counts.tolist()),
        "cluster_min_count": int(counts.min()),
        "cluster_max_count": int(counts.max()),
        "query_cluster_counts_json": json.dumps(
            np.bincount(query_labels, minlength=CLUSTERS).tolist()
        ),
        "query_sha256": digest(queries),
        "exact_ids_sha256": digest(exact_ids),
        "faiss_threads": faiss.omp_get_max_threads(),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__, "faiss_version": faiss.__version__,
        "cpu_model": cpu_model, "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
    }
    return [
        {**metadata, "repeat_index": i, "elapsed_seconds": seconds,
         "qps": len(queries) * TIMING_LAPS / seconds,
         "median_seconds": median_seconds,
         "median_qps": len(queries) * TIMING_LAPS / median_seconds}
        for i, seconds in enumerate(elapsed, start=1)
    ]


def completed(
    results_path: Path, ratio: int, family: str, base_hash: str, query_hash: str,
) -> bool:
    if not results_path.exists():
        return False
    with results_path.open(newline="") as source:
        rows = [r for r in csv.DictReader(source) if
                r.get("experiment") == "synthetic_cluster_imbalance"
                and r.get("population_ratio") == str(ratio)
                and r.get("index_family") == family]
    return (len(rows) == REPETITIONS
            and {int(row["repeat_index"]) for row in rows} == set(range(1, REPETITIONS + 1))
            and all(row["source_sha256"] == base_hash and row["query_sha256"] == query_hash
                    and row["config_json"] == json.dumps({
                        "seed": SEED, "base_count": BASE_COUNT, "dimensions": DIMENSIONS,
                        "clusters": CLUSTERS, "queries_per_cluster": QUERIES_PER_CLUSTER,
                        "centre_radius": CENTRE_RADIUS, "cluster_std": CLUSTER_STD,
                        "population_ratio": ratio, "k": K, "repetitions": REPETITIONS,
                        "timing_laps": TIMING_LAPS,
                        "index_family": family,
                        "build_settings": SETTINGS[family][0],
                        "search_settings": SETTINGS[family][1],
                    }, sort_keys=True) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--results", type=Path, default=root / "results/results.csv")
    parser.add_argument("--force", action="store_true", help="rerun completed settings")
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    shared = make_shared_geometry()
    query_hash = digest(shared.queries)
    for ratio in RATIOS:
        base, _, counts = make_base(shared, ratio)
        base_hash = digest(base)
        pending = [family for family in SETTINGS if args.force or not completed(
            args.results, ratio, family, base_hash, query_hash,
        )]
        if not pending:
            print(f"ratio {ratio}: all runs already complete", flush=True)
            continue
        exact = faiss.IndexFlatL2(DIMENSIONS)
        exact.add(base)
        _, exact_ids = exact.search(shared.queries, K)
        if np.any(exact_ids < 0):
            raise RuntimeError("exact search returned missing neighbour")
        print(f"ratio {ratio}: cluster counts {counts.min()}–{counts.max()}, exact reference ready", flush=True)
        for family in pending:
            print(f"  building/measuring {family}", flush=True)
            rows = measure(
                family, ratio, base, shared.queries, shared.query_labels,
                counts, exact_ids, shared.centres,
            )
            save_rows(args.results, rows, replace_filter=lambda old:
                      old.get("experiment") == "synthetic_cluster_imbalance"
                      and old.get("population_ratio") == str(ratio)
                      and old.get("index_family") == family)
            print(f"  {family}: recall={rows[0]['recall_at_10']:.4f}, "
                  f"median QPS={rows[0]['median_qps']:.1f}, "
                  f"rare={rows[0]['rare_quartile_recall_at_10']:.4f}, "
                  f"common={rows[0]['common_quartile_recall_at_10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
