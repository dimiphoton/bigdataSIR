"""Règles de combinaison des interventions."""

import numpy as np
import pytest

from epidemic.config import load_scenarios_config, load_simulator_config
from epidemic.simulator.contacts import build_contacts
from epidemic.simulator.dynamics import force_of_infection
from epidemic.simulator.interventions import (
    AppliedPolicy,
    assert_compatible,
    mask_multiplier,
    pair_factor,
    reduction_tables,
)


def test_pair_factor_uses_the_strongest_reduction_once() -> None:
    assert pair_factor(0.5, 0.5) == pytest.approx(0.5)
    assert pair_factor(0.0, 0.75) == pytest.approx(0.25)


def test_mask_multiplies_and_does_not_change_contacts() -> None:
    scenarios = load_scenarios_config()
    contacts = build_contacts(load_simulator_config())
    tables = reduction_tables([AppliedPolicy("mask")], scenarios, contacts, 0.9)
    assert float(tables.ordinary.sum()) == 0.0
    assert mask_multiplier([AppliedPolicy("mask")], scenarios, 0.9) == pytest.approx(
        0.82
    )
    assert mask_multiplier([], scenarios, 0.9) == 1.0


def test_lockdown_and_isolation_keep_the_max_reduction() -> None:
    scenarios = load_scenarios_config()
    contacts = build_contacts(load_simulator_config())
    tables = reduction_tables(
        [AppliedPolicy("lockdown"), AppliedPolicy("case_isolation", duration_days=10)],
        scenarios,
        contacts,
        0.9,
    )
    work = 2
    assert tables.ordinary[0, work] == pytest.approx(0.9 * 0.75)
    assert tables.targeted[0, work] == pytest.approx(1.0)


def test_social_distancing_stops_at_the_age_threshold() -> None:
    scenarios = load_scenarios_config()
    contacts = build_contacts(load_simulator_config())
    tables = reduction_tables(
        [AppliedPolicy("social_distancing", age_threshold=60)],
        scenarios,
        contacts,
        0.9,
    )
    names = [group.name for group in contacts.groups]
    young = names.index("0-17")
    old = names.index("70+")
    work = 2
    home = 0
    assert tables.ordinary[young, work] == 0.0
    assert tables.ordinary[old, work] == pytest.approx(0.9 * 0.75)
    assert tables.ordinary[old, home] == 0.0


def test_case_isolation_cannot_combine_with_household_quarantine() -> None:
    scenarios = load_scenarios_config()
    with pytest.raises(ValueError):
        assert_compatible(["case_isolation", "household_quarantine"], scenarios)


def test_force_of_infection_matches_a_hand_calculation() -> None:
    contacts = np.zeros((1, 1, 1))
    contacts[0, 0, 0] = 2.0
    pools = [(np.array([10.0]), np.zeros((1, 1)))]
    lam = force_of_infection(
        np.array([100.0]),
        contacts,
        np.array([1.0]),
        0.1,
        np.zeros((1, 1)),
        pools,
        np.array([1.0]),
    )
    assert float(lam[0]) == pytest.approx(0.02)
