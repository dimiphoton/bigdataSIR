"""Tests, faux négatifs et cas comptés à l'admission."""

from dataclasses import dataclass

import numpy as np

from epidemic.simulator.counts import round_half_up, split_count


@dataclass
class ObservedTests:
    """Résultat d'une journée de dépistage."""

    num_positive: int
    num_tested: int
    true_positives: int
    false_negatives: int
    false_positives: int
    confirmed: np.ndarray
    false_negative: np.ndarray
    untested: np.ndarray
    screening_confirmed: np.ndarray


def _take_up_to(counts: np.ndarray, limit: int) -> np.ndarray:
    total = int(counts.sum())
    if total <= limit:
        return counts.astype(np.int64)
    if limit <= 0:
        return np.zeros(counts.shape[0], dtype=np.int64)
    return split_count(limit, counts.astype(float), stochastic=False, rng=None)


def _draw_kept(
    population: np.ndarray,
    probability: float,
    *,
    stochastic: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    drawn = np.zeros(population.shape[0], dtype=np.int64)
    for index, count in enumerate(population):
        parts = split_count(
            int(count),
            np.asarray([probability, 1.0 - probability]),
            stochastic=stochastic,
            rng=rng,
        )
        drawn[index] = parts[0]
    return drawn


def _draw_positive(
    tested: np.ndarray,
    sensitivity: float,
    *,
    stochastic: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    positives = np.zeros(tested.shape[0], dtype=np.int64)
    for index, count in enumerate(tested):
        parts = split_count(
            int(count),
            np.asarray([sensitivity, 1.0 - sensitivity]),
            stochastic=stochastic,
            rng=rng,
        )
        positives[index] = parts[0]
    return positives


def classify_tests(
    new_symptomatic: np.ndarray,
    screening_candidates: np.ndarray,
    admissions_unconfirmed: int,
    *,
    p_test: float,
    screening_rate: float,
    sensitivity: float,
    specificity: float,
    flu_mean: float,
    max_daily_tests: int,
    stochastic: bool,
    rng: np.random.Generator | None,
) -> ObservedTests:
    """Compte les tests du jour. Chaque positif correspond à un test."""
    symptomatic = np.asarray(new_symptomatic, dtype=np.int64)
    candidates = np.asarray(screening_candidates, dtype=np.int64)
    desired_symptomatic = _draw_kept(
        symptomatic, p_test, stochastic=stochastic, rng=rng
    )
    if screening_rate > 0:
        desired_screening = _draw_kept(
            candidates, screening_rate, stochastic=stochastic, rng=rng
        )
    else:
        desired_screening = np.zeros(symptomatic.shape[0], dtype=np.int64)
    if stochastic:
        if rng is None:
            raise ValueError("Un générateur est requis en mode stochastique.")
        flu_tests = int(rng.poisson(max(flu_mean, 0.0)))
    else:
        flu_tests = round_half_up(flu_mean)

    remaining = max(0, max_daily_tests - int(admissions_unconfirmed))
    kept_symptomatic = _take_up_to(desired_symptomatic, remaining)
    remaining -= int(kept_symptomatic.sum())
    kept_screening = _take_up_to(desired_screening, remaining)
    remaining -= int(kept_screening.sum())
    kept_flu = min(flu_tests, remaining)

    confirmed = _draw_positive(
        kept_symptomatic, sensitivity, stochastic=stochastic, rng=rng
    )
    false_negative = kept_symptomatic - confirmed
    untested = symptomatic - kept_symptomatic
    screening_confirmed = _draw_positive(
        kept_screening,
        sensitivity,
        stochastic=stochastic,
        rng=rng,
    )
    false_positive_probability = 1.0 - specificity
    false_positives = int(
        split_count(
            kept_flu,
            np.asarray([false_positive_probability, specificity]),
            stochastic=stochastic,
            rng=rng,
        )[0]
    )
    admissions = int(admissions_unconfirmed)
    true_positives = int(confirmed.sum() + screening_confirmed.sum())
    num_positive = true_positives + false_positives + admissions
    num_tested = int(
        kept_symptomatic.sum() + kept_screening.sum() + kept_flu + admissions
    )
    return ObservedTests(
        num_positive=num_positive,
        num_tested=num_tested,
        true_positives=true_positives,
        false_negatives=int(false_negative.sum()),
        false_positives=false_positives,
        confirmed=confirmed,
        false_negative=false_negative,
        untested=untested,
        screening_confirmed=screening_confirmed,
    )
