"""Dynamique, observation et priors du modèle d'inférence."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace

from epidemic.config import load_inference_config
from epidemic.model import seirh as seirh_module
from epidemic.model.age_structure import bins_for, load_age_structure
from epidemic.model.dynamics import initial_state, run_dynamics
from epidemic.model.observation import expected_observations
from epidemic.model.priors import (
    beta_from_mean_sd,
    lognormal_on_interval,
    prior_catalogue,
)
from epidemic.model.seirh import model_kwargs, seirh_model


def test_age_groups_cover_the_city() -> None:
    for n_groups in (2, 3, 4):
        structure = load_age_structure(n_groups)
        assert structure.n_groups == n_groups
        assert structure.population == 1_000_000
        assert structure.contacts.shape == (n_groups, n_groups)
        assert np.min(structure.contacts) >= 0
        assert structure.older[-1]
        assert np.all(np.diff(structure.age_score) > 0)


def test_unknown_age_scheme_is_rejected() -> None:
    try:
        bins_for(5)
    except ValueError as exc:
        assert "2, 3 ou 4" in str(exc)
    else:
        raise AssertionError("5 groupes auraient dû être refusés.")


def test_observation_means_match_the_testing_story() -> None:
    new_symptomatic = jnp.array([100.0, 40.0])
    new_hospital = jnp.array([4.0, 1.0])
    positive, tested, false_negatives = expected_observations(
        new_symptomatic,
        new_hospital,
        jnp.array(0.6),
        jnp.array(0.8),
        jnp.array(1.0),
        jnp.array(50.0),
    )
    infected_tests = 0.6 * new_symptomatic
    admissions = new_hospital * (1.0 - 0.6 * 0.8)
    assert np.allclose(positive, infected_tests * 0.8 + admissions)
    assert np.allclose(tested, infected_tests + 50.0 + admissions)
    assert np.allclose(false_negatives, infected_tests * 0.2)
    assert np.all(positive <= tested + 1e-5)


def test_perfect_specificity_gives_no_false_positives() -> None:
    positive, tested, false_negatives = expected_observations(
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.6),
        jnp.array(0.8),
        jnp.array(1.0),
        jnp.array(80.0),
    )
    assert float(positive) == 0.0
    assert float(tested) == 80.0
    assert float(false_negatives) == 0.0


def test_population_is_conserved_and_non_negative() -> None:
    population = jnp.array([600_000.0, 400_000.0])
    contacts = jnp.array([[3.0, 1.0], [1.5, 2.0]])
    state0 = initial_state(population, jnp.array(40.0), jnp.array(10.0), jnp.array(0.6))
    assert np.allclose(state0.sum(axis=1), population)
    trajectory = run_dynamics(
        state0,
        jnp.full(30, 0.03),
        jnp.zeros((30, 2)),
        population=population,
        contacts=contacts,
        sigma=jnp.array(1.0 / 3.0),
        gamma=jnp.array(1.0 / 6.0),
        gamma_h=jnp.array(1.0 / 8.0),
        gamma_icu=jnp.array(1.0 / 10.0),
        symptomatic_fraction=jnp.array(0.6),
        p_hosp=jnp.array([0.01, 0.08]),
        p_icu=jnp.array([0.1, 0.25]),
        p_death_ward=jnp.array([0.01, 0.05]),
        p_death_icu=jnp.array([0.2, 0.4]),
        asymptomatic_infectiousness=jnp.array(0.5),
        process_sigma=jnp.array(0.0),
    )
    totals = trajectory["state"].sum(axis=(1, 2))
    assert np.allclose(totals, 1_000_000.0, atol=1.0)
    assert np.min(np.asarray(trajectory["state"])) >= -1e-3


def test_course_bounds_become_lognormal_intervals() -> None:
    mu, sigma = lognormal_on_interval(1.0, 5.0)
    z = 1.6448536269514722
    assert abs(np.exp(mu - z * sigma) - 1.0) < 1e-6
    assert abs(np.exp(mu + z * sigma) - 5.0) < 1e-6
    alpha, beta = beta_from_mean_sd(0.6, 0.05)
    assert abs(alpha / (alpha + beta) - 0.6) < 1e-9


def test_prior_catalogue_separates_constraints_and_hypotheses() -> None:
    rows = prior_catalogue()
    by_label = {row.label: row for row in rows}
    assert by_label["Durée d'incubation"].source == "consigne"
    assert by_label["Fond de syndromes grippaux"].source == "hypothèse"
    assert "60" in by_label["Taux de dépistage"].justification or "50" in (
        by_label["Taux de dépistage"].justification
    )


def test_the_model_package_does_not_name_the_secret_file() -> None:
    root = Path(seirh_module.__file__).resolve().parent
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert ".truth" not in text
        assert "truth_daily" not in text


def test_seeded_model_keeps_positives_within_tests() -> None:
    settings = load_inference_config().model_copy(
        update={"variant": "A", "beta_mode": "weekly_rw", "age_groups": 2}
    )
    structure = load_age_structure(2)
    kwargs = model_kwargs(structure, 14, settings, None, fixed_specificity=1.0)
    traced = trace(seed(seirh_model, jax.random.PRNGKey(0))).get_trace(**kwargs)
    positive = np.asarray(traced["expected_confirmed"]["value"])
    tested = np.asarray(traced["expected_tested"]["value"])
    totals = np.asarray(traced["compartment_total"]["value"])
    assert positive.shape == (14,)
    assert np.all(positive <= tested + 1e-3)
    assert np.allclose(totals, structure.population, atol=5.0)
    assert "false_negatives" in traced
    assert "p_hosp" in traced
    assert np.asarray(traced["p_hosp"]["value"]).shape == (2,)


def test_an_extreme_beta_draw_stays_finite() -> None:
    from numpyro.handlers import condition

    settings = load_inference_config()
    structure = load_age_structure(2)
    kwargs = model_kwargs(structure, 10, settings, None, fixed_specificity=1.0)
    extreme = condition(seirh_model, {"log_beta0": jnp.array(40.0)})
    traced = trace(seed(extreme, jax.random.PRNGKey(2))).get_trace(**kwargs)
    beta = np.asarray(traced["beta_t"]["value"])
    confirmed = np.asarray(traced["expected_confirmed"]["value"])
    assert np.isfinite(beta).all()
    assert np.isfinite(confirmed).all()
    assert float(np.max(beta)) <= 5.0


def test_variants_and_piecewise_beta_can_be_traced() -> None:
    structure = load_age_structure(2)
    base = load_inference_config()
    cases = (
        {"variant": "B", "beta_mode": "weekly_rw", "beta_breakpoints": []},
        {"variant": "C", "beta_mode": "weekly_rw", "beta_breakpoints": []},
        {"variant": "A", "beta_mode": "piecewise", "beta_breakpoints": [8]},
    )
    for update in cases:
        settings = base.model_copy(update=update)
        kwargs = model_kwargs(structure, 14, settings, None, fixed_specificity=1.0)
        traced = trace(seed(seirh_model, jax.random.PRNGKey(1))).get_trace(**kwargs)
        assert np.asarray(traced["beta_t"]["value"]).shape == (14,)
        if update["variant"] == "C":
            assert np.asarray(traced["infection_shock"]["value"]).shape == (14, 2)
        if update["variant"] == "B":
            assert "p_test_sigma" in traced
