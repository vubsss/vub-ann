"""Load and validate the two ann-benchmarks HDF5 datasets.

The original GloVe file uses angular distance. Unit-length vectors with L2
distance preserve its cosine-neighbour ordering, as required by the assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    base_count: int
    dimensions: int
    query_count: int = 10_000
    tune_count: int = 5_000
    source_distance: str = "euclidean"
    normalize: bool = False


DATASETS = {
    "sift": DatasetSpec("sift", "sift-128-euclidean.hdf5", 1_000_000, 128),
    "glove": DatasetSpec(
        "glove", "glove-100-angular.hdf5", 1_183_514, 100,
        source_distance="angular", normalize=True,
    ),
}


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    base: np.ndarray
    queries: np.ndarray
    neighbors: np.ndarray
    zero_base_vectors: int
    zero_query_vectors: int

    @property
    def tune_queries(self) -> np.ndarray:
        return self.queries[: self.spec.tune_count]

    @property
    def report_queries(self) -> np.ndarray:
        return self.queries[self.spec.tune_count :]

    @property
    def tune_neighbors(self) -> np.ndarray:
        return self.neighbors[: self.spec.tune_count]

    @property
    def report_neighbors(self) -> np.ndarray:
        return self.neighbors[self.spec.tune_count :]


def _unit_normalize_in_place(vectors: np.ndarray, block_size: int = 65_536) -> int:
    """Normalize without allocating a second full-size base matrix.

    Zero vectors remain zero and are counted, since dividing them would make
    non-finite values. The count is exposed for later ground-truth checks.
    """
    zero_count = 0
    for start in range(0, len(vectors), block_size):
        block = vectors[start : start + block_size]
        norms = np.sqrt(np.einsum("ij,ij->i", block, block))
        nonzero = norms > 0
        zero_count += int((~nonzero).sum())
        block[nonzero] /= norms[nonzero, None]
    return zero_count


def load_dataset(path: str | Path, spec: DatasetSpec) -> LoadedDataset:
    """Read vectors, queries and reference IDs, then enforce dataset invariants.

    IDs are zero-based row positions in ``base``. Neighbor distances in the HDF5
    file are intentionally not used: GloVe changes from angular to normalized L2.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}")

    with h5py.File(path, "r") as source:
        missing = {"train", "test", "neighbors"} - set(source.keys())
        if missing:
            raise ValueError(f"{path}: missing HDF5 arrays {sorted(missing)}")

        train = source["train"]
        test = source["test"]
        stored = source["neighbors"]
        if train.shape != (spec.base_count, spec.dimensions):
            raise ValueError(f"{path}: unexpected train shape {train.shape}")
        if test.shape != (spec.query_count, spec.dimensions):
            raise ValueError(f"{path}: unexpected test shape {test.shape}")
        if len(stored.shape) != 2 or stored.shape[0] != spec.query_count or stored.shape[1] < 10:
            raise ValueError(f"{path}: unexpected neighbors shape {stored.shape}")
        if not np.issubdtype(stored.dtype, np.integer):
            raise ValueError(f"{path}: neighbor IDs must be integers, got {stored.dtype}")
        if not (0 < spec.tune_count < spec.query_count):
            raise ValueError("tune_count must split the query set into two nonempty halves")

        source_distance = source.attrs.get("distance")
        if isinstance(source_distance, bytes):
            source_distance = source_distance.decode("utf-8")
        if source_distance is not None and source_distance != spec.source_distance:
            raise ValueError(
                f"{path}: expected {spec.source_distance} source distance, got {source_distance}"
            )

        base = np.ascontiguousarray(train[:], dtype=np.float32)
        queries = np.ascontiguousarray(test[:], dtype=np.float32)
        neighbors = np.ascontiguousarray(stored[:], dtype=np.int64)

    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        raise ValueError(f"{path}: vectors contain NaN or infinity")
    if neighbors.min() < 0 or neighbors.max() >= spec.base_count:
        raise ValueError(f"{path}: stored neighbor ID is outside the base vector range")
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise ValueError(f"{path}: a query has duplicate stored neighbor IDs")

    zero_base = zero_queries = 0
    if spec.normalize:
        zero_base = _unit_normalize_in_place(base)
        zero_queries = _unit_normalize_in_place(queries)

    return LoadedDataset(spec, base, queries, neighbors, zero_base, zero_queries)
