"""Projections avec bruit de processus, et bulletin court."""

import numpy as np

from epidemic.decision.evaluate import prepare_bulletin
from epidemic.forecast.project import constant_draws, project_scenarios
from epidemic.model.age_structure import load_age_structure


def test_a_lower_contact_matrix_infects_fewer_people() -> None:
    structure = load_age_structure(2)
    draws = constant_draws(
        2,
        10,
        2,
        beta=0.08,
        e0=80,
        p_hosp=0.08,
        process_sigma=0.0,
    )
    weaker = structure.contacts * 0.25
    baseline, reduced = project_scenarios(
        draws,
        structure,
        [structure.contacts, weaker],
        n_days=10,
        switch_index=0,
        scales=np.ones((2, 2)),
        seed=3,
    )
    assert baseline.new_infections.sum() > reduced.new_infections.sum()
    again, _reduced_again = project_scenarios(
        draws,
        structure,
        [structure.contacts, weaker],
        n_days=10,
        switch_index=0,
        scales=np.ones((2, 2)),
        seed=3,
    )
    assert np.allclose(baseline.new_infections, again.new_infections)


def test_future_shocks_depend_on_the_seed() -> None:
    structure = load_age_structure(2)
    draws = constant_draws(2, 4, 2, beta=0.08, e0=60, process_sigma=0.0)
    matrix = [structure.contacts]
    scales = np.ones((1, 2))
    first = project_scenarios(
        draws, structure, matrix, n_days=9, switch_index=4, scales=scales, seed=1
    )[0]
    second = project_scenarios(
        draws, structure, matrix, n_days=9, switch_index=4, scales=scales, seed=2
    )[0]
    assert not np.allclose(first.new_infections, second.new_infections)


def test_bulletin_ranks_lockdown_below_doing_nothing() -> None:
    draws = constant_draws(2, 6, 2, beta=0.09, e0=100, p_hosp=0.1, process_sigma=0.0)
    advice = prepare_bulletin(
        draws,
        bulletin_day=6,
        allowed=["lockdown"],
        end_day=14,
        seed=4,
    )
    by_key = {option.key: option for option in advice.options}
    assert "aucune" in by_key
    lockdown = next(option for option in advice.options if "lockdown" in option.key)
    assert lockdown.factor < by_key["aucune"].factor
    assert (
        lockdown.fans["infections"].median.sum()
        < by_key["aucune"].fans["infections"].median.sum()
    )
    assert advice.recommendation.message
    assert advice.growth_text
    assert "ratio" in by_key["aucune"].fans
    assert 0.0 <= advice.p_saturation_alert <= 1.0
    assert advice.alpha_table
