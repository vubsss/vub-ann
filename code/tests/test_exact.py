import numpy as np
import pytest

from annbench.benchmarks.exact import recall_at_k, recall_per_query, squared_l2_for_ids


def test_perfect_recall_even_when_order_changes() -> None:
    ids = np.array([[2, 0, 1]], dtype=np.int64)
    distances = np.array([[2.0, 0.0, 1.0]])
    assert recall_at_k(ids, distances, np.array([2.0]), k=3) == 1.0


def test_missing_neighbor_and_duplicate_do_not_get_credit() -> None:
    ids = np.array([[0, 1, 4], [0, 0, 1]], dtype=np.int64)
    distances = np.array([[0.0, 1.0, 9.0], [0.0, 0.0, 1.0]])
    scores = recall_per_query(ids, distances, np.array([2.0, 2.0]), k=3)
    assert np.allclose(scores, [2 / 3, 2 / 3])


def test_boundary_tie_counts_as_correct() -> None:
    # The exact top-3 ID list could end in 2 or 3: both lie at distance 2.
    ids = np.array([[0, 1, 3]], dtype=np.int64)
    distances = np.array([[0.0, 1.0, 2.0]])
    assert recall_at_k(ids, distances, np.array([2.0]), k=3) == 1.0


def test_true_distances_handle_missing_ids() -> None:
    base = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    queries = np.array([[0.0]], dtype=np.float32)
    ids = np.array([[0, 2, -1]], dtype=np.int64)
    distances = squared_l2_for_ids(base, queries, ids)
    assert distances[0, 0] == 0.0
    assert distances[0, 1] == 4.0
    assert np.isinf(distances[0, 2])
    with pytest.raises(ValueError, match="outside"):
        squared_l2_for_ids(base, queries, np.array([[3]]))
