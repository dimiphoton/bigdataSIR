"""Ville simulée : conservation, reproductibilité, interventions."""

from dataclasses import replace

import pandas as pd
import pytest

from epidemic.config import load_simulator_config, project_root
from epidemic.simulator.contacts import build_contacts
from epidemic.simulator.dynamics import PUBLISHED_COLUMNS
from epidemic.simulator.engine import RealitySimulator
from epidemic.simulator.interventions import AppliedPolicy


def test_contacts_come_from_the_city_stats() -> None:
    contacts = build_contacts(load_simulator_config())
    assert sum(group.population for group in contacts.groups) == 1_000_000
    assert contacts.degrees["home"] == pytest.approx(2.505, abs=0.01)
    populations = [group.population for group in contacts.groups]
    names = [group.name for group in contacts.groups]
    old = names.index("70+")
    for location in range(contacts.matrices.shape[0]):
        for left in range(len(populations)):
            for right in range(len(populations)):
                left_mass = contacts.matrices[location, left, right] * populations[left]
                right_mass = (
                    contacts.matrices[location, right, left] * populations[right]
                )
                assert left_mass == pytest.approx(right_mass)
    assert float(contacts.matrices[1, old].sum()) == 0.0


def test_secret_parameters_follow_the_seed_not_the_level() -> None:
    low = RealitySimulator.from_configs(seed=4, level=1)
    high = RealitySimulator.from_configs(seed=4, level=3)
    other = RealitySimulator.from_configs(seed=5, level=1)
    assert low.truth.beta_per_contact == high.truth.beta_per_contact
    assert low.truth.symptomatic_fraction == high.truth.symptomatic_fraction
    assert low.truth.sources.demographic is True
    assert low.truth.sources.environmental is False
    assert high.truth.sources.individual_contagion is True
    assert low.truth.beta_per_contact != other.truth.beta_per_contact


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_same_seed_replays_the_same_epidemic(level: int) -> None:
    first, _ = RealitySimulator.from_configs(seed=3, level=level).run(12)
    second, _ = RealitySimulator.from_configs(seed=3, level=level).run(12)
    assert first.equals(second)
    assert list(first.columns) == list(PUBLISHED_COLUMNS)
    assert (first["num_positive"] <= first["num_tested"]).all()
    assert int(first["num_positive"].min()) >= 0


def test_population_stays_constant() -> None:
    simulator = RealitySimulator.from_configs(seed=1, level=1)
    assert simulator.state.total() == 1_000_000
    simulator.run(20)
    assert simulator.state.total() == 1_000_000


def test_no_positive_without_infection_when_specificity_is_one() -> None:
    drawn = RealitySimulator.from_configs(seed=0, level=1).truth
    truth = replace(
        drawn,
        beta_per_contact=0.0,
        initial_exposed=0,
        flu_daily_mean=200.0,
        specificity=1.0,
    )
    simulator = RealitySimulator.from_configs(seed=0, level=1, truth=truth)
    observations, _hidden = simulator.run(5)
    assert int(observations["num_positive"].sum()) == 0
    assert int(observations["num_tested"].sum()) > 0


def test_lockdown_lowers_expected_incidence() -> None:
    drawn = RealitySimulator.from_configs(seed=0, level=0).truth
    truth = replace(
        drawn,
        beta_per_contact=0.05,
        initial_exposed=120,
        incubation_days=2.0,
        presymptomatic_days=1.5,
        symptomatic_days=6.0,
        asymptomatic_days=6.0,
        symptomatic_fraction=0.6,
        flu_daily_mean=0.0,
    )
    baseline = RealitySimulator.from_configs(seed=0, level=0, truth=truth)
    locked = RealitySimulator.from_configs(seed=0, level=0, truth=truth)
    _, baseline_truth = baseline.run(30)
    _, locked_truth = locked.run(30, policies=[AppliedPolicy("lockdown")])
    baseline_infections = int(baseline_truth["new_infections"].sum())
    locked_infections = int(locked_truth["new_infections"].sum())
    assert baseline_infections > 100
    assert locked_infections < baseline_infections


def test_course_file_replaces_the_first_published_days() -> None:
    simulator = RealitySimulator.from_configs(
        seed=0, level=0, observation_source="course"
    )
    observations, _hidden = simulator.run(2)
    course = pd.read_csv(project_root() / "data" / "external" / "data.csv")
    assert int(observations.iloc[0]["num_positive"]) == int(
        course.iloc[0]["num_positive"]
    )
    assert int(observations.iloc[1]["num_tested"]) == int(course.iloc[1]["num_tested"])
