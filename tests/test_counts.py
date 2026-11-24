"""Répartition entière."""

import numpy as np

from epidemic.simulator.counts import split_count


def test_split_count_conserves_people() -> None:
    rng = np.random.default_rng(0)
    probs = np.array([0.2, 0.5, 0.3])
    for n in (0, 1, 7, 1000):
        drawn = split_count(n, probs, stochastic=True, rng=rng)
        fixed = split_count(n, probs, stochastic=False, rng=None)
        assert int(drawn.sum()) == n
        assert int(fixed.sum()) == n
        assert np.all(drawn >= 0)
        assert np.all(fixed >= 0)


def test_split_count_is_reproducible() -> None:
    probs = np.array([0.1, 0.9])
    first = split_count(20, probs, stochastic=False, rng=None)
    second = split_count(20, probs, stochastic=False, rng=None)
    assert np.array_equal(first, second)
