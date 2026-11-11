"""Probabilités, scénarios, facteurs de transmission et règles."""

import numpy as np

from epidemic.config import load_decision_config, load_scenarios_config
from epidemic.decision.catalog import enumerate_scenarios
from epidemic.decision.policies import (
    ConditionalRule,
    apply_today,
    fresh_memory,
    schedule_today,
    search_thresholds,
)
from epidemic.decision.recommend import ScoredOption, pareto_front, recommend
from epidemic.decision.reduction import aggregate, factor_samples, point_factor
from epidemic.decision.sensitivity import (
    by_alpha,
    factors_by_compliance,
    isolation_factor,
)
from epidemic.forecast.bands import Fan
from epidemic.forecast.probabilities import (
    in_central_interval,
    saturation_by_day,
    summarize,
    weekly_probability_below,
)
from epidemic.model.age_structure import load_age_structure, load_contact_matrices
from epidemic.simulator.interventions import AppliedPolicy


def test_probabilities_match_a_hand_built_ensemble() -> None:
    hospital = np.array(
        [
            [100.0, 1600.0, 100.0],
            [100.0, 100.0, 100.0],
            [2000.0, 2000.0, 10.0],
            [10.0, 10.0, 400.0],
        ]
    )
    critical = np.zeros_like(hospital)
    critical[3, 2] = 350.0
    deaths = np.zeros_like(hospital)
    deaths[:, 2] = 1.0
    rt = np.array([[1.2, 1.1, 0.8], [0.9, 0.9, 0.9], [2.0, 1.5, 1.2], [0.7, 0.6, 0.5]])
    summary = summarize(
        hospital,
        critical,
        deaths,
        rt,
        np.array([1, 2, 3]),
        max_hospitalized=1500,
        max_critical=300,
        decision_index=0,
    )
    assert summary.p_hospital == 0.5
    assert summary.p_critical == 0.25
    assert summary.p_saturation == 0.75
    assert summary.first_day[0] == 2
    assert np.isnan(summary.first_day[1])
    assert summary.days_saturated[2] == 2
    assert summary.peak_hospital[0] == 1600
    assert summary.peak_hospital_day[0] == 2
    assert summary.p_rt_now == 0.5
    assert summary.p_rt_end == 0.75
    assert np.all(summary.cumulative_deaths == 1)
    assert in_central_interval(np.array([0.0, 1.0, 2.0]), 1.0, 0.5)
    week = np.vstack([np.full(7, 0.4), np.full(7, 1.4)])
    assert weekly_probability_below(week).tolist() == [0.5]
    curve = saturation_by_day(np.array([2.0, np.nan]), np.array([1, 2, 3]))
    assert np.allclose(curve, [0.0, 0.5, 0.5])


def test_catalog_respects_the_grid_and_the_ban() -> None:
    settings = load_decision_config()
    scenarios = load_scenarios_config()
    allowed = [
        "mask",
        "case_isolation",
        "household_quarantine",
        "social_distancing",
        "lockdown",
    ]
    listed = enumerate_scenarios(allowed, scenarios, settings)
    keys = [item.key for item in listed]
    assert "aucune" in keys
    assert any("case_isolation:7:" in key for key in keys)
    assert any("social_distancing:None:70" in key for key in keys)
    for item in listed:
        names = [policy.name for policy in item.policies]
        assert not ("case_isolation" in names and "household_quarantine" in names)
    assert 20 < len(listed) < 200


def test_transmission_factor_follows_the_rules() -> None:
    settings = load_decision_config()
    scenarios = load_scenarios_config()
    contacts = load_contact_matrices(2)
    structure = load_age_structure(2)
    assert np.allclose(aggregate(contacts), structure.contacts)
    empty: tuple[AppliedPolicy, ...] = ()
    mask = (AppliedPolicy("mask"),)
    lockdown = (AppliedPolicy("lockdown"),)
    near = point_factor(empty, empty, scenarios, contacts, settings)
    assert abs(near - 1.0) < 1e-8
    mask_factor = point_factor(mask, empty, scenarios, contacts, settings)
    lockdown_factor = point_factor(lockdown, empty, scenarios, contacts, settings)
    assert lockdown_factor < mask_factor < 1.0
    short = (AppliedPolicy("case_isolation", duration_days=7),)
    long = (AppliedPolicy("case_isolation", duration_days=14),)
    quarantine = (AppliedPolicy("household_quarantine", duration_days=14),)
    assert point_factor(long, empty, scenarios, contacts, settings) < point_factor(
        short, empty, scenarios, contacts, settings
    )
    assert point_factor(
        quarantine, empty, scenarios, contacts, settings
    ) < point_factor(long, empty, scenarios, contacts, settings)
    young = (AppliedPolicy("social_distancing", age_threshold=50),)
    old = (AppliedPolicy("social_distancing", age_threshold=70),)
    assert point_factor(young, empty, scenarios, contacts, settings) < point_factor(
        old, empty, scenarios, contacts, settings
    )
    soft, hard = factors_by_compliance(
        lockdown, empty, scenarios, contacts, settings, [0.5, 1.0]
    )
    assert hard[1] < soft[1]
    low = isolation_factor(0.5, scenarios, contacts, settings, duration_days=10)
    high = isolation_factor(0.7, scenarios, contacts, settings, duration_days=10)
    assert high < low
    samples = factor_samples(
        lockdown, empty, scenarios, contacts, settings, n_draws=12, seed=1
    )
    assert float(np.std(samples)) > 0
    same = factor_samples(
        empty, empty, scenarios, contacts, settings, n_draws=8, seed=2
    )
    assert np.allclose(same, 1.0)


def test_recommendation_is_the_cheapest_under_alpha() -> None:
    options = [
        _option("aucune", 0.0, 0.20),
        _option("masque", 10.0, 0.04),
        _option("confinement", 80.0, 0.01),
        _option("cher et risqué", 40.0, 0.30),
    ]
    result = recommend(options, 0.05)
    assert result.chosen is not None
    assert result.chosen.key == "masque"
    assert by_alpha(options, [0.2, 0.05, 0.001]) == [
        (0.2, "Aucune mesure"),
        (0.05, "Masque"),
        (0.001, None),
    ]
    front = {item.key for item in pareto_front(options)}
    assert "cher et risqué" not in front
    assert {"aucune", "masque", "confinement"} <= front


def test_rule_waits_for_the_delay_and_the_minimum_duration() -> None:
    rule = ConditionalRule(
        (AppliedPolicy("lockdown"),),
        "hospital",
        high=5.0,
        low=1.0,
        hold_days=2,
    )
    memory = fresh_memory(1)
    assert not bool(apply_today(memory, 0, min_days=3)[0])
    schedule_today(memory, 0, np.array([6.0]), rule, delay=2, min_days=3)
    assert not bool(apply_today(memory, 1, min_days=3)[0])
    assert bool(apply_today(memory, 2, min_days=3)[0])
    schedule_today(memory, 2, np.array([0.0]), rule, delay=2, min_days=3)
    assert bool(apply_today(memory, 3, min_days=3)[0])
    schedule_today(memory, 3, np.array([0.0]), rule, delay=2, min_days=3)
    assert bool(apply_today(memory, 4, min_days=3)[0])
    schedule_today(memory, 4, np.array([0.0]), rule, delay=2, min_days=3)
    assert memory.off_day[0] == -1
    assert bool(apply_today(memory, 5, min_days=3)[0])
    schedule_today(memory, 5, np.array([0.0]), rule, delay=2, min_days=3)
    assert memory.off_day[0] == 7
    assert bool(apply_today(memory, 6, min_days=3)[0])
    assert not bool(apply_today(memory, 7, min_days=3)[0])

    def evaluate(candidate: ConditionalRule) -> tuple[float, float]:
        if candidate.high < 100:
            return 0.01, 10.0
        return 0.5, 2.0

    cheap = ConditionalRule((AppliedPolicy("mask"),), "hospital", 200, 50, 3)
    safe = ConditionalRule((AppliedPolicy("mask"),), "hospital", 50, 10, 3)
    best, _scores = search_thresholds([cheap, safe], evaluate, 0.05)
    assert best.rule.high == 50


def _option(key: str, cost: float, risk: float) -> ScoredOption:
    day = np.array([1])
    empty = Fan(
        day,
        day.astype(float),
        day.astype(float),
        day.astype(float),
        day.astype(float),
        day.astype(float),
    )
    return ScoredOption(
        key=key,
        label={
            "aucune": "Aucune mesure",
            "masque": "Masque",
            "confinement": "Confinement",
        }.get(key, key),
        kind="fixe",
        unit_cost=cost,
        expected_cost=cost,
        factor=1.0,
        p_hospital=risk,
        p_critical=0.0,
        p_saturation=risk,
        deaths_median=0.0,
        deaths_low=0.0,
        deaths_high=0.0,
        peak_hospital_median=0.0,
        peak_hospital_day=1.0,
        peak_critical_median=0.0,
        peak_critical_day=1.0,
        days_saturated_median=0.0,
        p_rt_now=0.0,
        p_rt_end=0.0,
        first_day=np.array([np.nan]),
        weekly_rt_below=np.array([]),
        fans={"hospital": empty},
    )
