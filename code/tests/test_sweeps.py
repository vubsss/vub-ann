import csv
from pathlib import Path

import faiss
import numpy as np
import pytest

from annbench.benchmarks.runner import (
    benchmark_prebuilt, build_index, query_slice, reference_cutoffs,
)
from annbench.benchmarks.index_cache import load_index, save_index
from annbench.benchmarks.sweeps import choose_setting, configurations, selection_document
from annbench.data import DatasetSpec, LoadedDataset
from scripts.run_benchmark import save_rows


def _tiny_data() -> LoadedDataset:
    spec = DatasetSpec("sift", "unused.hdf5", 20, 2, query_count=4, tune_count=2)
    base = np.arange(40, dtype=np.float32).reshape(20, 2)
    queries = np.array([[0, 1], [2, 3], [36, 37], [38, 39]], dtype=np.float32)
    neighbors = np.tile(np.arange(10, dtype=np.int64), (4, 1))
    return LoadedDataset(spec, base, queries, neighbors, 0, 0)


def test_grid_filters_probes_above_nlist() -> None:
    plan = {
        "mode": "quick", "seed": 7, "repetitions": 2,
        "quick_base_count": 20, "quick_query_count": 2,
        "datasets": {"sift": {"ivf_flat": {
            "build": [{"nlist": 2}, {"nlist": 4}], "search": [1, 2, 4],
        }}},
    }
    configs = configurations(plan, "sift", "ivf_flat")
    assert [(c.build_settings["nlist"], c.search_settings["nprobe"]) for c in configs] == [
        (2, 1), (2, 2), (4, 1), (4, 2), (4, 4),
    ]


def test_selection_uses_only_tuning_and_handles_unreachable() -> None:
    plan = {
        "mode": "quick", "seed": 7, "repetitions": 2,
        "quick_base_count": 20, "quick_query_count": 2,
        "datasets": {"sift": {"hnsw": {
            "build": [{"M": 4, "efConstruction": 20}], "search": [4, 8, 16],
        }}},
    }
    configs = configurations(plan, "sift", "hnsw")
    rows = [
        {"query_split": "tune", "recall_at_10": 0.80, "median_qps": 300},
        {"query_split": "tune", "recall_at_10": 0.92, "median_qps": 200},
        {"query_split": "tune", "recall_at_10": 0.94, "median_qps": 100},
    ]
    selected, _, reached, highest = choose_setting(list(zip(configs, rows)), 0.90)
    assert selected.search_settings == {"efSearch": 8}
    assert reached and highest == 0.94
    selected, _, reached, highest = choose_setting(list(zip(configs, rows)), 0.95)
    assert selected.search_settings == {"efSearch": 16}
    assert not reached and highest == 0.94
    rows[0]["query_split"] = "report"
    with pytest.raises(ValueError, match="tuning"):
        choose_setting(list(zip(configs, rows)), 0.90)


def test_prebuilt_index_reused_across_search_settings() -> None:
    data = _tiny_data()
    plan = {
        "mode": "quick", "seed": 7, "repetitions": 2,
        "quick_base_count": 20, "quick_query_count": 2,
        "datasets": {"sift": {"hnsw": {
            "build": [{"M": 4, "efConstruction": 20}], "search": [4, 20],
        }}},
    }
    configs = configurations(plan, "sift", "hnsw")
    queries, stored, _, _ = query_slice(data, configs[0], "tune")
    cutoffs = reference_cutoffs(data.base, queries, stored, configs[0])
    index, build_seconds = build_index(configs[0], data.base)
    size = int(faiss.serialize_index(index).nbytes)
    rows = [benchmark_prebuilt(
        config, data, split="tune", source_sha256="fixture", index=index,
        build_seconds=build_seconds, index_size_bytes=size, cutoffs=cutoffs,
    ) for config in configs]
    assert all(len(group) == 2 for group in rows)
    assert rows[0][0]["build_seconds"] == rows[1][0]["build_seconds"]
    assert rows[0][0]["index_size_bytes"] == rows[1][0]["index_size_bytes"]
    assert rows[0][0]["search_settings"] != rows[1][0]["search_settings"]


def test_cached_build_round_trip_keeps_size_and_provenance(tmp_path: Path) -> None:
    data = _tiny_data()
    config = configurations({
        "mode": "quick", "seed": 7, "repetitions": 2,
        "quick_base_count": 20, "quick_query_count": 2,
        "datasets": {"sift": {"hnsw": {
            "build": [{"M": 4, "efConstruction": 20}], "search": [4],
        }}},
    }, "sift", "hnsw")[0]
    index, seconds = build_index(config, data.base)
    size = save_index(tmp_path, config, index, source_sha256="fixture", build_seconds=seconds)
    loaded = load_index(tmp_path, config, source_sha256="fixture", expected_base_count=20)
    assert loaded is not None
    assert loaded[1:] == (seconds, size)
    assert loaded[0].ntotal == 20
    with pytest.raises(ValueError, match="source"):
        load_index(tmp_path, config, source_sha256="changed", expected_base_count=20)


def test_selection_provenance_and_sweep_replacement(tmp_path: Path) -> None:
    plan = {
        "mode": "quick", "seed": 7, "repetitions": 2,
        "quick_base_count": 20, "quick_query_count": 2,
        "datasets": {"sift": {"hnsw": {
            "build": [{"M": 4, "efConstruction": 20}], "search": [4],
        }}},
    }
    config = configurations(plan, "sift", "hnsw")[0]
    row = {
        "query_split": "tune", "run_id": "first", "source_sha256": "source",
        "sweep_id": "sweep", "sweep_plan_sha256": "plan",
        "experiment": "ann_tuning_sweep", "dataset": "sift", "index_family": "hnsw",
    }
    selection = selection_document(
        config, row, target=0.95, reached=False, highest_recall=0.8,
        approved_search=[{"efSearch": 4}],
    )
    assert selection["tune_run_id"] == "first"
    assert selection["sweep_plan_sha256"] == "plan"
    path = tmp_path / "sweep.csv"
    save_rows(path, [row])
    changed = {**row, "run_id": "second"}
    save_rows(path, [changed], replace_filter=lambda old: old.get("sweep_plan_sha256") == "plan")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1 and rows[0]["run_id"] == "second"
