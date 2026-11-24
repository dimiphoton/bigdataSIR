"""Règle en boucle fermée : elle ne voit que les observations bruitées."""

from dataclasses import dataclass

import numpy as np

from epidemic.decision.policies import (
    ConditionalRule,
    apply_today,
    fresh_memory,
    indicator_value,
    schedule_today,
)
from epidemic.forecast.noise import (
    expected_counts,
    sample_beta_binomial,
    sample_log1p,
    sample_negative_binomial,
)
from epidemic.forecast.project import (
    LatentSegment,
    ParameterDraws,
    prepare_horizon,
    step_one_day,
)
from epidemic.model.age_structure import AgeStructure
from epidemic.model.seirh import HOSPITAL_LOG_SCALE


@dataclass
class RuleRollout:
    """Jours futurs d'une règle. L'activité est vraie quand la mesure agit."""

    day: np.ndarray
    num_hospitalised: np.ndarray
    num_critical: np.ndarray
    num_positive: np.ndarray
    num_tested: np.ndarray
    death_increments: np.ndarray
    new_infections: np.ndarray
    rt: np.ndarray
    active: np.ndarray


def rollout_rule(
    draws: ParameterDraws,
    structure: AgeStructure,
    rule: ConditionalRule,
    *,
    n_days: int,
    contacts_off: np.ndarray,
    contacts_on: np.ndarray,
    scales: np.ndarray,
    past_hospital: np.ndarray,
    past_critical: np.ndarray,
    past_positive: np.ndarray,
    past_tested: np.ndarray,
    delay: int,
    min_days: int,
    seed: int,
    cases_likelihood: str = "negative_binomial",
    hospital_likelihood: str = "negative_binomial",
) -> RuleRollout:
    """Joue la règle du lendemain du dernier jour tiré jusqu'à ``n_days``."""
    if delay < 0 or min_days < 1:
        raise ValueError(
            "Le délai est positif ou nul, la durée minimale vaut au moins un jour."
        )
    if scales.shape != (draws.n_draws,):
        raise ValueError("Les échelles ont une valeur par tirage.")
    horizon = prepare_horizon(draws, structure, n_days, seed)
    n_past = draws.n_past
    n_future = n_days - n_past
    if n_future < 1:
        raise ValueError("La règle a besoin d'au moins un jour futur.")
    _check_past(past_hospital, past_critical, past_positive, past_tested, n_past)

    population = np.asarray(structure.populations, dtype=float)
    state = horizon.state
    memory = fresh_memory(draws.n_draws)
    hospital = _history(past_hospital, draws.n_draws, n_days)
    critical = _history(past_critical, draws.n_draws, n_days)
    positive = _history(past_positive, draws.n_draws, n_days)
    tested = _history(past_tested, draws.n_draws, n_days)
    # Le jour du bulletin est déjà connu. Il peut armer la règle, avec le délai.
    schedule_today(
        memory,
        n_past - 1,
        indicator_value(
            rule.indicator,
            hospital[:, :n_past],
            critical[:, :n_past],
            positive[:, :n_past],
            tested[:, :n_past],
        ),
        rule,
        delay,
        min_days,
    )

    infections = np.zeros((draws.n_draws, n_future), dtype=float)
    deaths = np.zeros((draws.n_draws, n_future), dtype=float)
    rt = np.zeros((draws.n_draws, n_future), dtype=float)
    active = np.zeros((draws.n_draws, n_future), dtype=bool)
    rng = np.random.default_rng(seed + 1)
    off = np.broadcast_to(contacts_off, (draws.n_draws, *contacts_off.shape))
    on = np.broadcast_to(contacts_on, (draws.n_draws, *contacts_on.shape))

    for step, day in enumerate(range(n_past, n_days)):
        today = apply_today(memory, day, min_days)
        active[:, step] = today
        beta = horizon.beta[:, day] * np.where(today, scales, 1.0)
        contacts = np.where(today[:, None, None], on, off)
        state, segment = step_one_day(
            state,
            beta,
            horizon.shocks[:, day],
            horizon.sigma,
            contacts,
            draws,
            population,
        )
        observed = _observe_day(
            segment, draws, rng, cases_likelihood, hospital_likelihood
        )
        hospital[:, day] = observed[0]
        critical[:, day] = observed[1]
        positive[:, day] = observed[2]
        tested[:, day] = observed[3]
        deaths[:, step] = observed[4]
        infections[:, step] = segment.new_infections[:, 0]
        rt[:, step] = segment.rt[:, 0]
        schedule_today(
            memory,
            day,
            indicator_value(
                rule.indicator,
                hospital[:, : day + 1],
                critical[:, : day + 1],
                positive[:, : day + 1],
                tested[:, : day + 1],
            ),
            rule,
            delay,
            min_days,
        )

    return RuleRollout(
        day=np.arange(n_past + 1, n_days + 1),
        num_hospitalised=hospital[:, n_past:],
        num_critical=critical[:, n_past:],
        num_positive=positive[:, n_past:],
        num_tested=tested[:, n_past:],
        death_increments=deaths,
        new_infections=infections,
        rt=rt,
        active=active,
    )


def _history(past: np.ndarray, n_draws: int, n_days: int) -> np.ndarray:
    series = np.zeros((n_draws, n_days), dtype=float)
    series[:, : past.shape[0]] = past
    return series


def _check_past(
    hospital: np.ndarray,
    critical: np.ndarray,
    positive: np.ndarray,
    tested: np.ndarray,
    n_past: int,
) -> None:
    for series in (hospital, critical, positive, tested):
        if series.shape != (n_past,):
            raise ValueError("L'historique observé a un jour par jour déjà tiré.")


def _observe_day(
    segment: LatentSegment,
    draws: ParameterDraws,
    rng: np.random.Generator,
    cases_likelihood: str,
    hospital_likelihood: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent = segment
    positive_mean, tested_mean, _ignored = expected_counts(
        latent.new_symptomatic,
        latent.new_hospital,
        draws.p_test,
        draws.sensitivity,
        draws.specificity,
        draws.flu_mean,
    )
    tested = sample_negative_binomial(rng, tested_mean, draws.concentration)
    if cases_likelihood == "beta_binomial":
        positive = sample_beta_binomial(
            rng, positive_mean, tested_mean, draws.concentration, tested
        )
    else:
        positive = sample_negative_binomial(rng, positive_mean, draws.concentration)
    if hospital_likelihood == "lognormal":
        hospital = sample_log1p(rng, latent.hospitalized, HOSPITAL_LOG_SCALE)
        critical = sample_log1p(rng, latent.critical, HOSPITAL_LOG_SCALE)
    else:
        hospital = sample_negative_binomial(
            rng, latent.hospitalized, draws.concentration
        )
        critical = sample_negative_binomial(rng, latent.critical, draws.concentration)
    deaths = sample_negative_binomial(rng, latent.new_deaths, draws.concentration)
    return hospital[:, 0], critical[:, 0], positive[:, 0], tested[:, 0], deaths[:, 0]
