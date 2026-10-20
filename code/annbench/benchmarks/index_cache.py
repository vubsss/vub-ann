"""Temporary full-size indexes shared by tuning and held-out measurement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import faiss

from annbench.benchmarks.runner import BenchmarkConfig


def cache_path(directory: Path, config: BenchmarkConfig) -> Path:
    key = {
        "dataset": config.dataset,
        "family": config.index_family,
        "build_settings": config.build_settings,
        "seed": config.seed,
        "mode": config.mode,
        "quick_base_count": config.quick_base_count,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return directory / f"{config.dataset}_{config.index_family}_{digest}.faiss"


def save_index(
    directory: Path, config: BenchmarkConfig, index: faiss.Index, *,
    source_sha256: str, build_seconds: float,
) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    path = cache_path(directory, config)
    temp = path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temp))
    temp.replace(path)
    size = path.stat().st_size
    metadata = {
        "config": asdict(config), "source_sha256": source_sha256,
        "build_seconds": build_seconds, "index_size_bytes": size,
        "faiss_version": faiss.__version__, "base_count": index.ntotal,
        "build_threads": 4 if config.mode == "full" and config.index_family == "hnsw" else 1,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return size


def load_index(
    directory: Path, config: BenchmarkConfig, *, source_sha256: str,
    expected_base_count: int,
) -> tuple[faiss.Index, float, int] | None:
    path = cache_path(directory, config)
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    old = metadata["config"]
    for key in ("dataset", "index_family", "build_settings", "seed", "mode", "quick_base_count"):
        if old[key] != getattr(config, key):
            raise ValueError(f"cache config differs from selected build: {path}")
    if metadata["source_sha256"] != source_sha256 or metadata["faiss_version"] != faiss.__version__:
        raise ValueError(f"cache source or FAISS version changed: {path}")
    expected_threads = 4 if config.mode == "full" and config.index_family == "hnsw" else 1
    if metadata.get("build_threads", 1) != expected_threads:
        raise ValueError(f"cache build thread setting changed: {path}")
    if metadata["base_count"] != expected_base_count or metadata["index_size_bytes"] != path.stat().st_size:
        raise ValueError(f"cache count or size changed: {path}")
    index = faiss.read_index(str(path))
    if index.ntotal != expected_base_count:
        raise ValueError(f"cached index has wrong base count: {path}")
    return index, float(metadata["build_seconds"]), int(metadata["index_size_bytes"])
