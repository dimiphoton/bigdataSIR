"""Un pas d'un jour : force d'infection, chaîne binomiale, observation."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from epidemic.config import ScenariosConfig
from epidemic.simulator.contacts import ContactMatrices
from epidemic.simulator.counts import split_count
from epidemic.simulator.interventions import (
    AppliedPolicy,
    mask_multiplier,
    pair_factor,
    reduction_tables,
)
from epidemic.simulator.observation import classify_tests
from epidemic.simulator.parameters import TruthParameters
from epidemic.simulator.state import EpidemicState, LatentNoise

PUBLISHED_COLUMNS = (
    "Day",
    "num_positive",
    "num_tested",
    "num_hospitalised",
    "num_cumulative_hospitalizations",
    "num_critical",
    "num_fatalities",
)


@dataclass
class AdvanceResult:
    """État du lendemain et les deux lignes de sortie du jour."""

    state: EpidemicState
    latent: LatentNoise
    published: dict[str, int]
    truth_row: dict[str, Any]
    cumulative_hospitalizations: int


def leave_probability(mean_days: float, stages: int) -> float:
    """Probabilité de quitter un stade. La moyenne des stades vaut mean_days."""
    if mean_days <= 0:
        raise ValueError("La durée moyenne doit être positive.")
    rate = stages / mean_days
    return 1.0 - math.exp(-rate)


def flow_stages(
    values: np.ndarray,
    p_leave: float,
    *,
    stochastic: bool,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Avance les stades. L'entrée du jour n'en sort pas encore."""
    n_ages, n_stages = values.shape
    updated = np.zeros_like(values)
    outflow = np.zeros(n_ages, dtype=np.int64)
    for age in range(n_ages):
        stayers = np.zeros(n_stages, dtype=np.int64)
        leavers = np.zeros(n_stages, dtype=np.int64)
        for stage in range(n_stages):
            parts = split_count(
                int(values[age, stage]),
                np.asarray([p_leave, 1.0 - p_leave]),
                stochastic=stochastic,
                rng=rng,
            )
            leavers[stage] = int(parts[0])
            stayers[stage] = int(parts[1])
        updated[age, 0] = stayers[0]
        for stage in range(1, n_stages):
            updated[age, stage] = stayers[stage] + leavers[stage - 1]
        outflow[age] = leavers[-1]
    return updated, outflow


def branch_counts(
    counts: np.ndarray,
    probs: np.ndarray,
    *,
    stochastic: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Découpe chaque effectif. Retour de forme (issues, âges)."""
    n_ages, _n_ways = probs.shape
    out = np.zeros((probs.shape[1], n_ages), dtype=np.int64)
    for age in range(n_ages):
        out[:, age] = split_count(
            int(counts[age]),
            probs[age],
            stochastic=stochastic,
            rng=rng,
        )
    return out


def force_of_infection(
    populations: np.ndarray,
    contacts: np.ndarray,
    weights: np.ndarray,
    beta: float,
    susceptible_reduction: np.ndarray,
    pools: list[tuple[np.ndarray, np.ndarray]],
    contagion_multiplier: np.ndarray,
) -> np.ndarray:
    """Taux de force d'infection du jour, pour chaque tranche d'âge."""
    n_ages = int(populations.shape[0])
    n_locations = int(contacts.shape[0])
    lam = np.zeros(n_ages, dtype=float)
    for susceptible in range(n_ages):
        total = 0.0
        for infector in range(n_ages):
            if populations[infector] <= 0:
                continue
            for counts, infector_reduction in pools:
                prevalence = float(
                    counts[infector]
                    * contagion_multiplier[infector]
                    / populations[infector]
                )
                if prevalence == 0.0:
                    continue
                for location in range(n_locations):
                    factor = pair_factor(
                        float(susceptible_reduction[susceptible, location]),
                        float(infector_reduction[infector, location]),
                    )
                    total += (
                        beta
                        * float(weights[location])
                        * float(contacts[location, susceptible, infector])
                        * factor
                        * prevalence
                    )
        lam[susceptible] = total
    return lam


def _beta_around(mean: float, sd: float, rng: np.random.Generator) -> float:
    """Tirage Beta centré sur mean, avec un écart-type proche de sd."""
    center = min(max(mean, 1e-3), 1.0 - 1e-3)
    variance = min(sd**2, center * (1.0 - center) * 0.99)
    if variance <= 0:
        return center
    kappa = center * (1.0 - center) / variance - 1.0
    return float(rng.beta(center * kappa, (1.0 - center) * kappa))


def next_environment(
    latent: LatentNoise,
    truth: TruthParameters,
    day: int,
    rng: np.random.Generator | None,
) -> tuple[LatentNoise, float, float, float, float]:
    """Met à jour β(t), p_test, la sensibilité et le fond grippal."""
    updated = latent.copy()
    if not truth.sources.environmental:
        return (
            updated,
            truth.beta_per_contact,
            truth.p_test,
            truth.sensitivity,
            truth.flu_daily_mean,
        )
    if rng is None:
        raise ValueError("Le générateur environnemental est requis.")
    shock_beta = float(rng.normal())
    shock_p_test = float(rng.normal())
    shock_flu = float(rng.normal())
    # Ornstein-Uhlenbeck sur log β : un petit écart, ramené vers la valeur de départ.
    updated.log_beta_deviation = float(
        np.clip(
            updated.log_beta_deviation
            + truth.ou_theta * (0.0 - updated.log_beta_deviation)
            + truth.ou_sigma * shock_beta,
            -5.0,
            5.0,
        )
    )
    season = 1.0 + truth.seasonality_amplitude * math.cos(
        2.0 * math.pi * (day - 1) / 365.0
    )
    beta = truth.beta_per_contact * math.exp(updated.log_beta_deviation) * season
    updated.p_test = float(
        np.clip(
            updated.p_test
            + truth.ou_theta * (truth.p_test - updated.p_test)
            + truth.p_test_drift_sigma * shock_p_test,
            truth.p_test_low,
            truth.p_test_high,
        )
    )
    log_flu = math.log(max(updated.flu_mean, 1e-6))
    log_target = math.log(max(truth.flu_daily_mean, 1e-6))
    updated.flu_mean = math.exp(
        log_flu
        + truth.ou_theta * (log_target - log_flu)
        + truth.flu_drift_sigma * shock_flu
    )
    if truth.sensitivity_daily_sd > 0:
        sensitivity = _beta_around(truth.sensitivity, truth.sensitivity_daily_sd, rng)
    else:
        sensitivity = truth.sensitivity
    return updated, beta, updated.p_test, sensitivity, updated.flu_mean


def draw_compliance(
    truth: TruthParameters,
    baseline: float,
    rng: np.random.Generator | None,
) -> float:
    """Respect du jour. Fixe, sauf si l'hétérogénéité individuelle est active."""
    if not truth.sources.individual_compliance:
        return baseline
    if rng is None:
        raise ValueError("Le générateur individuel est requis.")
    return float(
        np.clip(_beta_around(baseline, truth.compliance_daily_sd, rng), 0.0, 1.0)
    )


def draw_contagion(
    state: EpidemicState,
    truth: TruthParameters,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Contagiosité du jour. Moyenne 1, plus variable s'il y a peu d'infectieux."""
    multipliers = np.ones(state.s.shape[0], dtype=float)
    if not truth.sources.individual_contagion:
        return multipliers
    if rng is None:
        raise ValueError("Le générateur individuel est requis.")
    infectious = (
        state.i_asymptomatic.sum(axis=1)
        + state.i_presymptomatic.sum(axis=1)
        + state.i_symptomatic_untested.sum(axis=1)
        + state.i_symptomatic_negative.sum(axis=1)
        + state.i_symptomatic_positive.sum(axis=1)
    )
    for age, count in enumerate(infectious):
        weight = max(int(count), 1)
        shape = truth.heterogeneity_k * weight
        drawn = float(rng.gamma(shape, 1.0 / shape))
        multipliers[age] = drawn if int(count) > 0 else 1.0
    return multipliers


def _transmission_pools(
    state: EpidemicState,
    latent: LatentNoise,
    truth: TruthParameters,
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    policies: list[AppliedPolicy],
    compliance: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, float]:
    """Découpe les infectieux et donne la part de ménages en quarantaine."""
    tables = reduction_tables(policies, scenarios, contacts, compliance)
    symptomatic_positive = state.i_symptomatic_positive.sum(axis=1).astype(float)
    symptomatic_untested = state.i_symptomatic_untested.sum(axis=1).astype(float)
    symptomatic_negative = state.i_symptomatic_negative.sum(axis=1).astype(float)
    asymptomatic = state.i_asymptomatic.sum(axis=1).astype(float)
    presymptomatic = state.i_presymptomatic.sum(axis=1).astype(float)

    coverage = 1.0
    if tables.isolation_duration_days is not None and truth.symptomatic_days > 0:
        coverage = min(1.0, tables.isolation_duration_days / truth.symptomatic_days)

    pool_people = asymptomatic + presymptomatic
    extra_people = np.minimum(latent.extra_confirmed.astype(float), pool_people)
    hidden = asymptomatic * truth.asymptomatic_infectiousness + presymptomatic
    share = np.zeros_like(pool_people)
    present = pool_people > 0
    share[present] = extra_people[present] / pool_people[present]
    extra_effect = hidden * share
    free_hidden = hidden - extra_effect

    isolate = tables.has_case_isolation or tables.has_household_quarantine
    isolated_positive = np.zeros_like(symptomatic_positive)
    free_positive = symptomatic_positive
    isolated_untested = np.zeros_like(symptomatic_untested)
    free_untested = symptomatic_untested
    isolated_extra = np.zeros_like(extra_effect)
    free_hidden_pool = hidden
    if isolate:
        isolated_positive = compliance * coverage * symptomatic_positive
        free_positive = symptomatic_positive - isolated_positive
        isolated_extra = extra_effect
        free_hidden_pool = free_hidden
        if truth.isolate_on_symptoms:
            isolated_untested = compliance * coverage * symptomatic_untested
            free_untested = symptomatic_untested - isolated_untested

    quarantine_fraction = 0.0
    if tables.has_household_quarantine:
        index_people = float(
            (compliance * coverage * (symptomatic_positive + extra_people)).sum()
        )
        quarantine_fraction = min(
            0.95,
            index_people * (contacts.mean_household_size - 1.0) / contacts.population,
        )

    pools: list[tuple[np.ndarray, np.ndarray]] = []

    def add(counts: np.ndarray, reduction: np.ndarray) -> None:
        if np.any(counts > 0):
            pools.append((counts, reduction))

    def add_with_quarantine(counts: np.ndarray) -> None:
        if quarantine_fraction <= 0:
            add(counts, tables.ordinary)
            return
        add(counts * (1.0 - quarantine_fraction), tables.ordinary)
        add(counts * quarantine_fraction, tables.targeted)

    add(isolated_positive + isolated_untested, tables.targeted)
    add(isolated_extra, tables.targeted)
    add(free_positive, tables.ordinary)
    add_with_quarantine(free_untested)
    add_with_quarantine(symptomatic_negative)
    add_with_quarantine(free_hidden_pool)

    return pools, tables.ordinary, tables.targeted, quarantine_fraction


def _infect(
    state: EpidemicState,
    lam: np.ndarray,
    *,
    stochastic: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    infections = np.zeros(state.s.shape[0], dtype=np.int64)
    for age, hazard in enumerate(lam):
        probability = 1.0 - math.exp(-max(float(hazard), 0.0))
        parts = split_count(
            int(state.s[age]),
            np.asarray([probability, 1.0 - probability]),
            stochastic=stochastic,
            rng=rng,
        )
        infections[age] = int(parts[0])
    return infections


def initial_state(contacts: ContactMatrices, truth: TruthParameters) -> EpidemicState:
    """Tout le monde est susceptible, sauf les exposés initiaux."""
    n_ages = len(contacts.groups)
    stages = truth.stage_count
    populations = np.asarray(
        [group.population for group in contacts.groups], dtype=np.int64
    )
    exposed = split_count(
        truth.initial_exposed,
        populations.astype(float),
        stochastic=False,
        rng=None,
    )
    empty = np.zeros((n_ages, stages), dtype=np.int64)
    exposed_stages = empty.copy()
    exposed_stages[:, 0] = exposed
    return EpidemicState(
        s=populations - exposed,
        e=exposed_stages,
        i_asymptomatic=empty.copy(),
        i_presymptomatic=empty.copy(),
        i_symptomatic_untested=empty.copy(),
        i_symptomatic_negative=empty.copy(),
        i_symptomatic_positive=empty.copy(),
        hospital=empty.copy(),
        icu=empty.copy(),
        recovered=np.zeros(n_ages, dtype=np.int64),
        dead=np.zeros(n_ages, dtype=np.int64),
    )


def initial_latent(contacts: ContactMatrices, truth: TruthParameters) -> LatentNoise:
    """Bruit environnemental au repos, et aucun dépistage élargi."""
    return LatentNoise(
        log_beta_deviation=0.0,
        p_test=truth.p_test,
        flu_mean=truth.flu_daily_mean,
        extra_confirmed=np.zeros(len(contacts.groups), dtype=np.int64),
    )


def advance_state(
    state: EpidemicState,
    latent: LatentNoise,
    *,
    day: int,
    truth: TruthParameters,
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    policies: list[AppliedPolicy],
    baseline_compliance: float,
    streams: dict[str, np.random.Generator],
    cumulative_hospitalizations: int,
) -> AdvanceResult:
    """Fait avancer la ville d'un jour. La population totale ne change pas."""
    stochastic = truth.sources.demographic
    demographic = streams["demographic"] if stochastic else None
    environmental = streams["environmental"] if truth.sources.environmental else None
    individual = (
        streams["individual"]
        if (truth.sources.individual_contagion or truth.sources.individual_compliance)
        else None
    )

    latent, beta, p_test, sensitivity, flu_mean = next_environment(
        latent, truth, day, environmental
    )
    compliance = draw_compliance(truth, baseline_compliance, individual)
    beta *= mask_multiplier(policies, scenarios, compliance)
    contagion = draw_contagion(state, truth, individual)

    pools, ordinary_reduction, targeted_reduction, quarantine_fraction = (
        _transmission_pools(
            state,
            latent,
            truth,
            scenarios,
            contacts,
            policies,
            compliance,
        )
    )
    populations = np.asarray(
        [group.population for group in contacts.groups], dtype=float
    )
    if quarantine_fraction <= 0.0:
        lam = force_of_infection(
            populations,
            contacts.matrices,
            contacts.weights,
            beta,
            ordinary_reduction,
            pools,
            contagion,
        )
    else:
        # max(réduction) n'est pas linéaire : on moyenne les forces, pas les réductions.
        lam_free = force_of_infection(
            populations,
            contacts.matrices,
            contacts.weights,
            beta,
            ordinary_reduction,
            pools,
            contagion,
        )
        lam_quarantine = force_of_infection(
            populations,
            contacts.matrices,
            contacts.weights,
            beta,
            targeted_reduction,
            pools,
            contagion,
        )
        lam = (
            1.0 - quarantine_fraction
        ) * lam_free + quarantine_fraction * lam_quarantine
    infections = _infect(state, lam, stochastic=stochastic, rng=demographic)

    stages = state.e.shape[1]
    p_exposed = leave_probability(truth.incubation_days, stages)
    p_presymptomatic = leave_probability(truth.presymptomatic_days, stages)
    p_asymptomatic = leave_probability(truth.asymptomatic_days, stages)
    p_symptomatic = leave_probability(truth.symptomatic_days, stages)
    p_hospital = leave_probability(truth.hospital_days, stages)
    p_icu = leave_probability(truth.icu_days, stages)

    exposed, exposed_out = flow_stages(
        state.e, p_exposed, stochastic=stochastic, rng=demographic
    )
    exposed[:, 0] += infections
    asymptomatic, asymptomatic_out = flow_stages(
        state.i_asymptomatic,
        p_asymptomatic,
        stochastic=stochastic,
        rng=demographic,
    )
    presymptomatic, presymptomatic_out = flow_stages(
        state.i_presymptomatic,
        p_presymptomatic,
        stochastic=stochastic,
        rng=demographic,
    )
    symptom_probability = np.tile(
        [truth.symptomatic_fraction, 1.0 - truth.symptomatic_fraction],
        (state.s.shape[0], 1),
    )
    to_presymptomatic, to_asymptomatic = branch_counts(
        exposed_out,
        symptom_probability,
        stochastic=stochastic,
        rng=demographic,
    )
    asymptomatic[:, 0] += to_asymptomatic
    presymptomatic[:, 0] += to_presymptomatic

    untested, untested_out = flow_stages(
        state.i_symptomatic_untested,
        p_symptomatic,
        stochastic=stochastic,
        rng=demographic,
    )
    negative, negative_out = flow_stages(
        state.i_symptomatic_negative,
        p_symptomatic,
        stochastic=stochastic,
        rng=demographic,
    )
    positive, positive_out = flow_stages(
        state.i_symptomatic_positive,
        p_symptomatic,
        stochastic=stochastic,
        rng=demographic,
    )
    hospital_probability = np.column_stack(
        [np.asarray(truth.p_hosp), 1.0 - np.asarray(truth.p_hosp)]
    )
    hospital_from_untested, recover_from_untested = branch_counts(
        untested_out,
        hospital_probability,
        stochastic=stochastic,
        rng=demographic,
    )
    hospital_from_negative, recover_from_negative = branch_counts(
        negative_out,
        hospital_probability,
        stochastic=stochastic,
        rng=demographic,
    )
    hospital_from_positive, recover_from_positive = branch_counts(
        positive_out,
        hospital_probability,
        stochastic=stochastic,
        rng=demographic,
    )

    screening_candidates = (
        state.i_asymptomatic.sum(axis=1)
        + state.i_presymptomatic.sum(axis=1)
        - presymptomatic_out
    )
    screening_candidates = np.maximum(screening_candidates, 0)
    observed = classify_tests(
        presymptomatic_out,
        screening_candidates,
        int(hospital_from_untested.sum() + hospital_from_negative.sum()),
        p_test=p_test,
        screening_rate=truth.extended_screening_rate,
        sensitivity=sensitivity,
        specificity=truth.specificity,
        flu_mean=flu_mean,
        max_daily_tests=truth.max_daily_tests,
        stochastic=stochastic,
        rng=demographic,
    )
    untested[:, 0] += observed.untested
    negative[:, 0] += observed.false_negative
    positive[:, 0] += observed.confirmed

    hospital, hospital_out = flow_stages(
        state.hospital,
        p_hospital,
        stochastic=stochastic,
        rng=demographic,
    )
    icu, icu_out = flow_stages(state.icu, p_icu, stochastic=stochastic, rng=demographic)
    hospital[:, 0] += (
        hospital_from_untested + hospital_from_negative + hospital_from_positive
    )

    p_icu_given = np.asarray(truth.p_icu, dtype=float)
    p_death_ward = np.asarray(truth.p_death_ward, dtype=float)
    ward_probs = np.column_stack(
        [
            p_icu_given,
            (1.0 - p_icu_given) * p_death_ward,
            1.0 - p_icu_given - (1.0 - p_icu_given) * p_death_ward,
        ]
    )
    to_icu, ward_deaths, ward_recoveries = branch_counts(
        hospital_out,
        ward_probs,
        stochastic=stochastic,
        rng=demographic,
    )
    icu_probs = np.column_stack(
        [np.asarray(truth.p_death_icu), 1.0 - np.asarray(truth.p_death_icu)]
    )
    icu_deaths, icu_recoveries = branch_counts(
        icu_out,
        icu_probs,
        stochastic=stochastic,
        rng=demographic,
    )
    icu[:, 0] += to_icu

    recovered = (
        state.recovered
        + asymptomatic_out
        + recover_from_untested
        + recover_from_negative
        + recover_from_positive
        + ward_recoveries
        + icu_recoveries
    )
    dead = state.dead + ward_deaths + icu_deaths
    susceptible = state.s - infections
    if int(susceptible.min()) < 0:
        raise RuntimeError(f"Jour {day} : des susceptibles sont devenus négatifs.")
    new_state = EpidemicState(
        s=susceptible,
        e=exposed,
        i_asymptomatic=asymptomatic,
        i_presymptomatic=presymptomatic,
        i_symptomatic_untested=untested,
        i_symptomatic_negative=negative,
        i_symptomatic_positive=positive,
        hospital=hospital,
        icu=icu,
        recovered=recovered,
        dead=dead,
    )
    if new_state.total() != contacts.population:
        raise RuntimeError(
            f"Jour {day} : population {new_state.total()}, "
            f"au lieu de {contacts.population}."
        )

    extra = latent.extra_confirmed.copy()
    if int(extra.sum()) > 0:
        for age in range(extra.shape[0]):
            parts = split_count(
                int(extra[age]),
                np.asarray([p_presymptomatic, 1.0 - p_presymptomatic]),
                stochastic=stochastic,
                rng=demographic,
            )
            extra[age] = int(parts[1])
    extra = extra + observed.screening_confirmed
    cap = new_state.i_asymptomatic.sum(axis=1) + new_state.i_presymptomatic.sum(axis=1)
    extra = np.minimum(extra, cap).astype(np.int64)
    latent.extra_confirmed = extra

    admissions = int(
        hospital_from_untested.sum()
        + hospital_from_negative.sum()
        + hospital_from_positive.sum()
    )
    cumulative = cumulative_hospitalizations + admissions
    hospitalized = int(new_state.hospital.sum())
    critical = int(new_state.icu.sum())
    deaths = int(new_state.dead.sum())
    published = {
        "Day": day,
        "num_positive": observed.num_positive,
        "num_tested": observed.num_tested,
        "num_hospitalised": hospitalized,
        "num_cumulative_hospitalizations": cumulative,
        "num_critical": critical,
        "num_fatalities": deaths,
    }
    infectious = int(
        new_state.i_asymptomatic.sum()
        + new_state.i_presymptomatic.sum()
        + new_state.i_symptomatic_untested.sum()
        + new_state.i_symptomatic_negative.sum()
        + new_state.i_symptomatic_positive.sum()
    )
    truth_row: dict[str, Any] = {
        "Day": day,
        "new_infections": int(infections.sum()),
        "new_symptomatic": int(presymptomatic_out.sum()),
        "infectious_prevalence": infectious,
        "susceptible": int(new_state.s.sum()),
        "hospitalized": hospitalized,
        "critical": critical,
        "deaths": deaths,
        "false_negatives": observed.false_negatives,
        "true_positives": observed.true_positives,
        "false_positives": observed.false_positives,
        "beta": beta,
        "p_test": p_test,
        "sensitivity": sensitivity,
        "policies": ",".join(policy.name for policy in policies),
    }
    return AdvanceResult(
        state=new_state,
        latent=latent,
        published=published,
        truth_row=truth_row,
        cumulative_hospitalizations=cumulative,
    )
