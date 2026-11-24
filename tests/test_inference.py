"""Scores, diagnostics et séries publiques. Pas de long MCMC."""

import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest

from epidemic.config import load_inference_config
from epidemic.inference.baselines import exponential_forecast, persistence_forecast
from epidemic.inference.compare import compare_criteria
from epidemic.inference.diagnostics import DiagnosticsError, check_diagnostics
from epidemic.inference.metrics import (
    ambiguity_reduction,
    classify_width_ratio,
    coverage_rate,
    crps_ensemble,
    identifiability_table,
    log_score,
)
from epidemic.inference.predict import PredictiveDraws, extend_weekly, projection_tables
from epidemic.inference.validation import load_course_series, simulate_public_series


def test_crps_of_a_point_mass_is_the_absolute_error() -> None:
    assert crps_ensemble(np.array([1.0, 1.0, 1.0]), 1.0) == 0.0
    score = crps_ensemble(np.array([0.0, 2.0]), 1.0)
    assert abs(score - 0.5) < 1e-9


def test_coverage_rate_counts_the_hits() -> None:
    samples = {
        "inside": np.linspace(0.0, 1.0, 100),
        "outside": np.linspace(0.0, 1.0, 100),
    }
    rate = coverage_rate({"inside": 0.5, "outside": 5.0}, samples, level=0.9)
    assert rate == 0.5


def test_width_ratio_labels() -> None:
    assert classify_width_ratio(0.2, 0.5, 0.85) == "well_identified"
    assert classify_width_ratio(0.6, 0.5, 0.85) == "partial"
    assert classify_width_ratio(0.9, 0.5, 0.85) == "prior_dominated"
    table = identifiability_table({"beta0": 0.2}, well_below=0.5, partial_below=0.85)
    assert table.loc[0, "lecture"] == "bien identifié"


def test_adding_data_can_shrink_an_interval() -> None:
    table = ambiguity_reduction(
        {"p_test": 0.40},
        {"p_test": 0.25},
        {"p_test": 0.10},
    )
    assert bool(table.loc[0, "hospital_shrinks"])
    assert bool(table.loc[0, "positivity_shrinks"])


def test_log_score_is_finite_for_a_tight_forecast() -> None:
    score = log_score(np.array([10.0, 10.0, 11.0, 9.0]), 10.0)
    assert np.isfinite(score)


def test_naive_baselines() -> None:
    history = np.array([3.0, 5.0])
    assert persistence_forecast(history, 3).tolist() == [5.0, 5.0, 5.0]
    growing = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    forecast = exponential_forecast(growing, horizon=2, window=5)
    assert forecast[0] > growing[-1]


def test_bad_chains_fail_when_strict() -> None:
    idata = az.from_dict(
        posterior={"beta0": np.array([[0.0, 0.1, 0.0, 0.05], [8.0, 8.2, 7.9, 8.1]])}
    )
    with pytest.raises(DiagnosticsError):
        check_diagnostics(
            idata,
            method="nuts",
            max_rhat=1.01,
            min_ess=400,
            max_divergences=0,
            strict=True,
        )
    report = check_diagnostics(
        idata,
        method="nuts",
        max_rhat=1.01,
        min_ess=400,
        max_divergences=0,
        strict=False,
    )
    assert report.ok is False
    assert report.max_rhat is not None
    assert report.max_rhat > 1.01


def test_svi_without_a_finite_elbo_fails() -> None:
    idata = az.from_dict(posterior={"beta0": np.ones((1, 4))})
    with pytest.raises(DiagnosticsError):
        check_diagnostics(
            idata,
            method="svi",
            max_rhat=1.01,
            min_ess=10,
            max_divergences=0,
            strict=True,
            elbo=float("nan"),
        )


def test_compare_requires_log_likelihood() -> None:
    idata = az.from_dict(posterior={"beta0": np.ones((1, 5))})
    with pytest.raises(ValueError, match="log_likelihood"):
        compare_criteria({"A": idata, "B": idata})


def test_weekly_projection_reads_the_next_bulletin() -> None:
    """Deux tirages, 7 jours observés, horizon 7. Le jour 14 est la cible."""
    past = np.ones((2, 7))
    future = np.array([[2.0] * 7, [4.0] * 7])
    series = np.concatenate([past, future], axis=1)
    draws = PredictiveDraws(
        num_positive=series,
        num_tested=series,
        num_hospitalised=series,
        num_critical=series,
        death_increments=series,
        new_infections=series,
        expected_confirmed=series,
        false_negatives=series,
        prevalence=series,
        hospitalized=series,
        critical=series,
    )
    weekly, daily = projection_tables(draws, origin_day=7, horizons=[7])
    level = weekly[
        (weekly["series"] == "num_hospitalised") & (weekly["kind"] == "niveau")
    ]
    total = weekly[(weekly["series"] == "num_positive") & (weekly["kind"] == "somme")]
    assert int(level["target_day"].iloc[0]) == 14
    assert float(level["median"].iloc[0]) == 3.0
    assert float(total["median"].iloc[0]) == 21.0
    projected = daily[daily["role"] == "projection"]
    projected = projected[projected["series"] == "num_positive"]
    assert len(projected) == 7


def test_zero_volatility_does_not_move_beta() -> None:
    past = jnp.array([0.03, 0.03, 0.03])
    extended = extend_weekly(past, jnp.array(0.0), jnp.zeros(5))
    assert np.asarray(extended).shape == (8,)
    assert np.allclose(extended, 0.03)


def test_inference_config_selects_the_default_variant() -> None:
    settings = load_inference_config()
    assert settings.variant == "A"
    assert settings.method == "nuts"
    assert settings.nuts.num_chains == 4
    assert settings.likelihood.cases == "negative_binomial"
    assert settings.age_groups == 2


def test_course_file_has_71_days() -> None:
    series = load_course_series()
    assert series.n_days == 71
    assert np.all(series.num_positive <= series.num_tested)
    assert np.all(series.death_increments >= 0)


def test_public_series_from_the_simulator_stays_public() -> None:
    series = simulate_public_series(level=0, days=8, seed=1)
    assert series.n_days == 8
    assert np.all(series.num_positive <= series.num_tested)
    assert np.all(series.num_positive >= 0)
