from pathlib import Path

import h5py
import numpy as np
import pytest

from annbench.data import DatasetSpec, load_dataset


def _fixture(path: Path, *, bad_id: bool = False) -> DatasetSpec:
    spec = DatasetSpec(
        "small-glove", path.name, base_count=12, dimensions=2,
        query_count=4, tune_count=2, source_distance="angular", normalize=True,
    )
    base = np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (12, 1))
    base[0] = 0
    queries = np.array([[0, 5], [3, 4], [4, 3], [5, 0]], dtype=np.float32)
    neighbors = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    if bad_id:
        neighbors[0, 0] = 12
    with h5py.File(path, "w") as handle:
        handle.attrs["distance"] = "angular"
        handle["train"] = base
        handle["test"] = queries
        handle["neighbors"] = neighbors
    return spec


def test_glove_normalization_split_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "small.hdf5"
    data = load_dataset(path, _fixture(path))
    assert data.base.dtype == np.float32
    assert data.neighbors.dtype == np.int64
    assert data.zero_base_vectors == 1
    assert np.allclose(np.linalg.norm(data.base[1:], axis=1), 1)
    assert np.allclose(np.linalg.norm(data.queries, axis=1), 1)
    assert data.tune_queries.shape == data.report_queries.shape == (2, 2)
    assert np.array_equal(data.report_queries, data.queries[2:])
    assert np.array_equal(data.report_neighbors, data.neighbors[2:])


def test_out_of_range_neighbor_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.hdf5"
    spec = _fixture(path, bad_id=True)
    with pytest.raises(ValueError, match="neighbor ID"):
        load_dataset(path, spec)
