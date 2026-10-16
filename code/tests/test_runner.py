import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from annbench.benchmarks.runner import (
    BenchmarkConfig, benchmark, build_index, query_slice, set_search_settings,
)
from annbench.data import DatasetSpec, LoadedDataset
from scripts.run_benchmark import load_selection, save_rows


def _tiny_data() -> LoadedDataset:
    spec = DatasetSpec("sift", "unused.hdf5", 20, 2, query_count=4, tune_count=2)
    base = np.arange(40, dtype=np.float32).reshape(20, 2)
    queries = np.array([[0, 1], [2, 3], [36, 37], [38, 39]], dtype=np.float32)
    neighbors = np.tile(np.arange(10, dtype=np.int64), (4, 1))
    return LoadedDataset(spec, base, queries, neighbors, 0, 0)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset="sift", index_family="hnsw",
        build_settings={"M": 4, "efConstruction": 20},
        search_settings={"efSearch": 20}, seed=17,
        mode="quick", repetitions=2, quick_base_count=20, quick_query_count=2,
    )


def test_quick_split_stays_on_its_original_half() -> None:
    data, config = _tiny_data(), _config()
    tune, _, tune_start, tune_end = query_slice(data, config, "tune")
    report, _, report_start, report_end = query_slice(data, config, "report")
    assert (tune_start, tune_end, report_start, report_end) == (0, 2, 2, 4)
    assert np.array_equal(tune, data.queries[:2])
    assert np.array_equal(report, data.queries[2:])


def test_run_level_rows_and_csv_round_trip(tmp_path: Path) -> None:
    config = _config()
    tune = benchmark(config, _tiny_data(), split="tune", source_sha256="fixture")
    report = benchmark(config, _tiny_data(), split="report", source_sha256="fixture")
    assert len(tune) == len(report) == 2
    assert {row["query_split"] for row in tune} == {"tune"}
    assert {row["query_split"] for row in report} == {"report"}
    assert {row["experiment"] for row in tune} == {"ann_tuning"}
    assert {row["experiment"] for row in report} == {"ann_final"}
    assert all(row["faiss_threads"] == 1 for row in tune + report)
    assert all(row["index_size_bytes"] > 0 and row["bytes_per_vector"] > 0 for row in tune + report)
    assert all(row["median_qps"] > 0 and 0 <= row["recall_at_10"] <= 1 for row in tune + report)
    assert tune[0]["run_id"] == tune[1]["run_id"]
    assert tune[0]["repeat_index"] == 1 and tune[1]["repeat_index"] == 2
    path = tmp_path / "results.csv"
    save_rows(path, tune)
    save_rows(path, report)
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert {row["query_split"] for row in rows} == {"tune", "report"}
    assert json.loads(rows[0]["config_json"])["seed"] == 17


def test_final_selection_requires_tuning_provenance(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    config = asdict(_config())
    path.write_text(json.dumps({"query_split": "report", "tune_run_id": "x", "config": config}))
    with pytest.raises(ValueError, match="tuning run"):
        load_selection(path)
    path.write_text(json.dumps({
        "query_split": "tune", "tune_run_id": "run-1", "source_sha256": "abc", "config": config,
    }))
    selected, run_id, source_hash = load_selection(path)
    assert selected == _config()
    assert (run_id, source_hash) == ("run-1", "abc")


def test_csv_merge_keeps_exact_baseline_columns(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("experiment,legacy_field\nexact_baseline,kept\n")
    rows = benchmark(_config(), _tiny_data(), split="tune", source_sha256="fixture")
    save_rows(path, rows)
    with path.open(newline="") as source:
        merged = list(csv.DictReader(source))
    assert len(merged) == 3
    assert merged[0]["experiment"] == "exact_baseline"
    assert merged[0]["legacy_field"] == "kept"
    assert merged[1]["experiment"] == "ann_tuning"


def test_invalid_family_settings_are_rejected() -> None:
    bad = BenchmarkConfig(
        dataset="sift", index_family="ivf_pq", build_settings={"nlist": 10, "m": 7, "nbits": 8},
        search_settings={"nprobe": 2},
    )
    with pytest.raises(ValueError, match="divide"):
        bad.validate()


@pytest.mark.parametrize("family,build,search", [
    ("ivf_flat", {"nlist": 16}, {"nprobe": 4}),
    ("ivf_pq", {"nlist": 16, "m": 8, "nbits": 4}, {"nprobe": 4}),
])
def test_ivf_backends_build_and_search(family: str, build: dict, search: dict) -> None:
    base = np.random.default_rng(7).random((4096, 128), dtype=np.float32)
    config = BenchmarkConfig("sift", family, build, search)
    config.validate()
    index, elapsed = build_index(config, base)
    set_search_settings(index, config)
    _, ids = index.search(base[:2], 10)
    assert elapsed > 0
    assert index.ntotal == len(base)
    assert index.nprobe == 4
    assert ids.shape == (2, 10)
    assert np.all(ids >= 0)
