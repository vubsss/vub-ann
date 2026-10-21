"""The four synthetic conditions differ only in cluster population labels."""

import numpy as np

from annbench.experiments.cluster_imbalance import (
    BASE_COUNT, CLUSTERS, QUERIES_PER_CLUSTER, RATIOS, cluster_counts,
    make_base, make_shared_geometry,
)


def test_seed_reproduces_all_geometry_and_fixed_queries():
    first = make_shared_geometry()
    second = make_shared_geometry()
    for name in ("centres", "base_noise", "assignment_order", "queries", "query_labels"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert np.bincount(first.query_labels, minlength=CLUSTERS).tolist() == [QUERIES_PER_CLUSTER] * CLUSTERS
    for ratio in RATIOS:
        base1, labels1, counts1 = make_base(first, ratio)
        base2, labels2, counts2 = make_base(second, ratio)
        assert np.array_equal(base1, base2)
        assert np.array_equal(labels1, labels2)
        assert np.array_equal(counts1, counts2)
        assert np.array_equal(np.bincount(labels1, minlength=CLUSTERS), counts1)


def test_only_populations_change_across_conditions():
    shared = make_shared_geometry()
    spread = []
    for ratio in RATIOS:
        base, labels, counts = make_base(shared, ratio)
        assert len(base) == BASE_COUNT
        assert int(counts.sum()) == BASE_COUNT
        assert np.allclose(base - shared.centres[labels], shared.base_noise, atol=2e-6)
        spread.append(float(counts.max() / counts.min()))
    assert spread[0] == 1.0
    assert all(a < b for a, b in zip(spread, spread[1:]))


def test_balanced_counts_are_exact():
    counts = cluster_counts(1)
    assert counts.tolist() == [BASE_COUNT // CLUSTERS] * CLUSTERS
