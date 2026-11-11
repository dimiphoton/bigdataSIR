"""Dépistage : positifs, tests, faux négatifs."""

import numpy as np

from epidemic.simulator.observation import classify_tests


def test_false_negative_fraction_tracks_sensitivity() -> None:
    result = classify_tests(
        np.array([50_000]),
        np.array([0]),
        0,
        p_test=1.0,
        screening_rate=0.0,
        sensitivity=0.8,
        specificity=1.0,
        flu_mean=0.0,
        max_daily_tests=100_000,
        stochastic=True,
        rng=np.random.default_rng(0),
    )
    tested = result.true_positives + result.false_negatives
    assert tested == 50_000
    assert abs(result.false_negatives / tested - 0.2) < 0.015


def test_specificity_one_gives_no_positive_among_uninfected() -> None:
    result = classify_tests(
        np.array([0, 0]),
        np.array([0, 0]),
        0,
        p_test=1.0,
        screening_rate=0.0,
        sensitivity=0.8,
        specificity=1.0,
        flu_mean=5_000,
        max_daily_tests=10_000,
        stochastic=True,
        rng=np.random.default_rng(1),
    )
    assert result.false_positives == 0
    assert result.num_positive == 0
    assert result.num_tested > 0


def test_admission_counts_as_positive_and_as_test() -> None:
    result = classify_tests(
        np.array([0]),
        np.array([0]),
        4,
        p_test=1.0,
        screening_rate=0.0,
        sensitivity=1.0,
        specificity=1.0,
        flu_mean=0.0,
        max_daily_tests=0,
        stochastic=False,
        rng=None,
    )
    assert result.num_positive == 4
    assert result.num_tested == 4


def test_capacity_keeps_positives_below_tests() -> None:
    result = classify_tests(
        np.array([30, 10]),
        np.array([0, 0]),
        2,
        p_test=1.0,
        screening_rate=0.0,
        sensitivity=0.7,
        specificity=0.9,
        flu_mean=100,
        max_daily_tests=15,
        stochastic=False,
        rng=None,
    )
    assert result.num_positive <= result.num_tested
    assert result.num_tested == 15


def test_partial_specificity_creates_false_positives() -> None:
    result = classify_tests(
        np.array([0]),
        np.array([0]),
        0,
        p_test=1.0,
        screening_rate=0.0,
        sensitivity=0.8,
        specificity=0.9,
        flu_mean=100,
        max_daily_tests=1_000,
        stochastic=False,
        rng=None,
    )
    assert result.true_positives == 0
    assert result.false_positives == 10
    assert result.num_positive == 10
