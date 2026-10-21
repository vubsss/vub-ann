"""Seeded synthetic clusters with population balance as the only varied input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEED = 20261021
BASE_COUNT = 96_000
DIMENSIONS = 32
CLUSTERS = 16
QUERIES_PER_CLUSTER = 64
CENTRE_RADIUS = 10.0
CLUSTER_STD = 1.0
RATIOS = (1, 4, 16, 64)


def cluster_counts(ratio: int) -> np.ndarray:
    """Largest-remainder allocation of geometric weights to exact populations."""
    if ratio not in RATIOS:
        raise ValueError(f"ratio must be one of {RATIOS}")
    weights = np.geomspace(1.0, float(ratio), CLUSTERS)
    ideal = BASE_COUNT * weights / weights.sum()
    counts = np.floor(ideal).astype(np.int64)
    leftover = BASE_COUNT - int(counts.sum())
    order = np.lexsort((np.arange(CLUSTERS), -(ideal - counts)))
    counts[order[:leftover]] += 1
    assert counts.sum() == BASE_COUNT and counts.min() >= 10
    return counts


@dataclass(frozen=True)
class SharedGeometry:
    centres: np.ndarray
    base_noise: np.ndarray
    assignment_order: np.ndarray
    queries: np.ndarray
    query_labels: np.ndarray


def make_shared_geometry(seed: int = SEED) -> SharedGeometry:
    """Generate all geometry and queries once, independent of population ratio."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    centre_seed, base_seed, assignment_seed, query_seed = np.random.SeedSequence(seed).spawn(4)
    centre_rng = np.random.default_rng(centre_seed)
    centres = centre_rng.normal(size=(CLUSTERS, DIMENSIONS)).astype(np.float32)
    centres *= CENTRE_RADIUS / np.linalg.norm(centres, axis=1)[:, None]

    base_rng = np.random.default_rng(base_seed)
    base_noise = base_rng.normal(0, CLUSTER_STD, (BASE_COUNT, DIMENSIONS)).astype(np.float32)
    assignment_order = np.random.default_rng(assignment_seed).permutation(BASE_COUNT)

    query_rng = np.random.default_rng(query_seed)
    query_labels = np.repeat(np.arange(CLUSTERS, dtype=np.int64), QUERIES_PER_CLUSTER)
    query_rng.shuffle(query_labels)
    query_noise = query_rng.normal(0, CLUSTER_STD, (len(query_labels), DIMENSIONS)).astype(np.float32)
    queries = np.ascontiguousarray(centres[query_labels] + query_noise)
    return SharedGeometry(centres, base_noise, assignment_order, queries, query_labels)


def make_base(shared: SharedGeometry, ratio: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep base row order and noise fixed; assign only cluster membership."""
    counts = cluster_counts(ratio)
    labels = np.empty(BASE_COUNT, dtype=np.int64)
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    for cluster in range(CLUSTERS):
        labels[shared.assignment_order[boundaries[cluster]:boundaries[cluster + 1]]] = cluster
    base = np.ascontiguousarray(shared.centres[labels] + shared.base_noise)
    return base, labels, counts
