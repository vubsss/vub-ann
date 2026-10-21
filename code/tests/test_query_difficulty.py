"""Checks for the preregistered local-contrast bin definition."""

import numpy as np
import pytest

from scripts.run_query_difficulty import contrast_bins


def test_contrast_bins_are_equal_count_and_stable_at_ties():
    # The first eight scores tie. Original query order breaks the tie.
    distances = np.tile(np.linspace(1.0, 2.0, 10), (16, 1))
    distances[8:, -1] = np.arange(2.1, 2.9, 0.1)
    scores, bins = contrast_bins(distances)
    assert np.bincount(bins, minlength=5)[1:].tolist() == [4, 4, 4, 4]
    assert bins[:8].tolist() == [1] * 4 + [2] * 4
    assert np.all(np.diff(scores[8:]) > 0)
    assert bins[8:].tolist() == [3] * 4 + [4] * 4


def test_contrast_bins_reject_nonpositive_tenth_distance():
    with pytest.raises(ValueError, match="positive d10"):
        contrast_bins(np.zeros((4, 10)))
