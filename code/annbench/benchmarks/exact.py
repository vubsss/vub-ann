"""Exact L2 reference, tie-aware recall, and single-thread search timing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import faiss
import numpy as np


def squared_l2_for_ids(
    base: np.ndarray, queries: np.ndarray, ids: np.ndarray, *, block_size: int = 256
) -> np.ndarray:
    """Compute true squared L2 distances for returned IDs, including PQ results.

    FAISS IVF-PQ search distances are approximate, so recall must use the source
    vectors rather than the scores returned by an approximate index. Missing
    IDs (-1) receive infinity. Float64 arithmetic limits boundary-rounding
    differences between indexes and the stored reference.
    """
    ids = np.asarray(ids)
    if ids.ndim != 2 or queries.ndim != 2 or ids.shape[0] != len(queries):
        raise ValueError("IDs must have shape (number of queries, number of results)")
    if base.ndim != 2 or base.shape[1] != queries.shape[1]:
        raise ValueError("base and query dimensions must agree")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("IDs must be integers")
    if np.any((ids < -1) | (ids >= len(base))):
        raise ValueError("result ID is outside the base vector range")

    distances = np.empty(ids.shape, dtype=np.float64)
    for start in range(0, len(queries), block_size):
        stop = min(start + block_size, len(queries))
        chunk_ids = ids[start:stop]
        valid = chunk_ids >= 0
        safe_ids = np.where(valid, chunk_ids, 0)
        differences = base[safe_ids].astype(np.float64) - queries[start:stop, None, :]
        squared = np.einsum("qkd,qkd->qk", differences, differences)
        distances[start:stop] = np.where(valid, squared, np.inf)
    return distances


def recall_per_query(
    found_ids: np.ndarray,
    found_true_squared_l2: np.ndarray,
    exact_kth_squared_l2: np.ndarray,
    *,
    k: int = 10,
    atol: float = 1e-6,
    rtol: float = 1e-7,
) -> np.ndarray:
    """Fraction of distinct retrieved IDs no farther than exact rank ``k``.

    Every ID tied with the kth exact neighbor is valid, even if that ID is not
    in a particular exact top-k list. Duplicate IDs cannot earn extra credit.
    A small tolerance accounts for floating-point distance evaluation.
    """
    found_ids = np.asarray(found_ids)
    distances = np.asarray(found_true_squared_l2)
    cutoff = np.asarray(exact_kth_squared_l2)
    if k <= 0 or found_ids.ndim != 2 or found_ids.shape[1] < k:
        raise ValueError("found_ids must contain at least k results per query")
    if distances.shape != found_ids.shape or cutoff.shape != (len(found_ids),):
        raise ValueError("distance and cutoff shapes must match the result IDs")
    if not np.issubdtype(found_ids.dtype, np.integer):
        raise ValueError("found_ids must be integers")
    if np.any(~np.isfinite(cutoff)):
        raise ValueError("exact kth distances must be finite")

    valid = (
        (found_ids[:, :k] >= 0)
        & (distances[:, :k] <= cutoff[:, None] + atol + rtol * np.abs(cutoff[:, None]))
    )
    scores = np.empty(len(found_ids), dtype=np.float64)
    for row in range(len(found_ids)):
        scores[row] = len(np.unique(found_ids[row, :k][valid[row]])) / k
    return scores


def recall_at_k(
    found_ids: np.ndarray,
    found_true_squared_l2: np.ndarray,
    exact_kth_squared_l2: np.ndarray,
    *,
    k: int = 10,
    atol: float = 1e-6,
    rtol: float = 1e-7,
) -> float:
    """Mean tie-aware recall over queries."""
    return float(recall_per_query(
        found_ids, found_true_squared_l2, exact_kth_squared_l2,
        k=k, atol=atol, rtol=rtol,
    ).mean())


@dataclass(frozen=True)
class ReferenceCheck:
    query_count: int
    exact_vs_stored_recall: float
    stored_vs_exact_recall: float
    differing_id_set_queries: int
    max_ranked_squared_l2_difference: float
    max_stored_distance_error: float
    max_faiss_squared_l2_error: float


def check_stored_reference(
    base: np.ndarray,
    queries: np.ndarray,
    stored_ids: np.ndarray,
    stored_distances: np.ndarray,
    exact_ids: np.ndarray,
    faiss_squared_l2: np.ndarray,
    *,
    source_distance: str,
    k: int = 10,
) -> ReferenceCheck:
    """Compare FAISS and file references, accepting only genuine boundary ties."""
    stored_ids = stored_ids[:, :k]
    stored_distances = stored_distances[:, :k]
    exact_ids = exact_ids[:, :k]
    faiss_squared_l2 = faiss_squared_l2[:, :k]
    exact_true = squared_l2_for_ids(base, queries, exact_ids)
    stored_true = squared_l2_for_ids(base, queries, stored_ids)

    exact_cutoff = exact_true.max(axis=1)
    stored_cutoff = stored_true.max(axis=1)
    stored_recall = recall_per_query(stored_ids, stored_true, exact_cutoff, k=k)
    exact_recall = recall_per_query(exact_ids, exact_true, stored_cutoff, k=k)
    if np.any(stored_recall < 1.0) or np.any(exact_recall < 1.0):
        bad = int(np.count_nonzero((stored_recall < 1.0) | (exact_recall < 1.0)))
        raise ValueError(f"exact search and stored reference disagree beyond ties on {bad} queries")

    if source_distance == "euclidean":
        converted_stored = np.sqrt(stored_true)
        distance_tolerance = 5e-3
    elif source_distance == "angular":
        converted_stored = stored_true / 2.0
        distance_tolerance = 1e-5
    else:
        raise ValueError(f"unsupported source distance: {source_distance}")
    stored_distance_error = float(np.max(np.abs(converted_stored - stored_distances)))
    if stored_distance_error > distance_tolerance:
        raise ValueError(
            f"stored distances do not match source vectors: max error {stored_distance_error}"
        )

    differing_sets = sum(
        set(exact_ids[row]) != set(stored_ids[row])
        for row in range(len(queries))
    )
    return ReferenceCheck(
        query_count=len(queries),
        exact_vs_stored_recall=float(exact_recall.mean()),
        stored_vs_exact_recall=float(stored_recall.mean()),
        differing_id_set_queries=differing_sets,
        max_ranked_squared_l2_difference=float(np.max(np.abs(
            np.sort(exact_true, axis=1) - np.sort(stored_true, axis=1)
        ))),
        max_stored_distance_error=stored_distance_error,
        max_faiss_squared_l2_error=float(np.max(np.abs(faiss_squared_l2 - exact_true))),
    )


def time_exact_search(
    index: faiss.Index, queries: np.ndarray, *, k: int = 10,
    mode: str, repetitions: int = 3,
    on_repeat: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Time an exact index on one thread, including Python per-query overhead."""
    if mode not in {"single", "batch"}:
        raise ValueError("mode must be 'single' or 'batch'")
    if repetitions < 1 or len(queries) < 1:
        raise ValueError("repetitions and query count must be positive")
    faiss.omp_set_num_threads(1)
    if mode == "single":
        for query in queries[: min(10, len(queries))]:
            index.search(query[None, :], k)
    else:
        index.search(queries[: min(10, len(queries))], k)

    times = []
    for repetition in range(1, repetitions + 1):
        start = perf_counter()
        if mode == "single":
            for query in queries:
                index.search(query[None, :], k)
        else:
            index.search(queries, k)
        elapsed = perf_counter() - start
        times.append(elapsed)
        if on_repeat is not None:
            on_repeat(repetition, elapsed)
    return times
