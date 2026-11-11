"""Scores, échelle verbale, bulletin et documentation générée."""

import re
from pathlib import Path

import numpy as np

from epidemic.config import load_reporting_config
from epidemic.forecast.scoring import (
    brier_score,
    interpret_coverage,
    naive_ensemble,
    score_observation_series,
)
from epidemic.forecast.verbal import verbal_probability
from epidemic.reporting.bulletin import write_bulletin
from epidemic.reporting.render import render_docs
from epidemic.series import ObservationSeries


def _flat_series(n_days: int, level: float) -> ObservationSeries:
    value = np.full(n_days, level)
    return ObservationSeries(value, value + 5, value, value / 2, np.zeros(n_days))


def test_verbal_scale_follows_the_brief() -> None:
    assert verbal_probability(1.0) == "quasi certain"
    assert verbal_probability(0.99) == "quasi certain"
    assert verbal_probability(0.90) == "très probable"
    assert verbal_probability(0.66) == "probable"
    assert verbal_probability(0.33) == "aussi probable qu'improbable"
    assert verbal_probability(0.10) == "improbable"
    assert verbal_probability(0.01) == "très improbable"
    assert verbal_probability(0.0) == "exceptionnellement improbable"


def test_brier_rewards_a_perfect_call() -> None:
    perfect = brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    wrong = brier_score(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert perfect == 0.0
    assert wrong == 1.0


def test_coverage_comment_detects_overconfidence() -> None:
    text = interpret_coverage(0.40, 0.90, 0.05)
    assert "trop sûrs" in text
    assert "proche" in interpret_coverage(0.90, 0.90, 0.05)


def test_a_flat_series_stays_inside_the_naive_band() -> None:
    settings = load_reporting_config()
    score = score_observation_series(
        _flat_series(50, 20.0),
        label="plate",
        period=7,
        min_history=settings.min_history_days,
        horizons=[7, 14],
        low_mass=0.5,
        high_mass=0.9,
        ensemble_size=40,
        seed=settings.seed,
        residual_window=settings.residual_window,
        series_names=["num_positive", "num_hospitalised"],
        max_hospitalized=1500,
        event_horizon=14,
    )
    assert score.coverage(high=True) == 1.0
    assert score.brier() == 0.0
    ensemble = naive_ensemble(
        np.full(10, 3.0), 4, kind="persistence", n_draws=5, seed=1, window=5
    )
    assert ensemble.shape == (5, 4)


def test_bulletin_states_confirmed_and_estimated(tmp_path: Path) -> None:
    written = write_bulletin(21, destination=tmp_path)
    text = next(path for path in written if path.suffix == ".qmd").read_text(
        encoding="utf-8"
    )
    assert "confirmé" in text
    assert "estimé" in text
    assert "{{" not in text
    assert "{%" not in text
    assert any(path.suffix == ".pdf" for path in written)


def test_generated_docs_have_no_empty_fields(tmp_path: Path) -> None:
    render_docs(tmp_path)
    docs = tmp_path / "docs"
    for name in ("framework-bayesien.md", "presentation.md"):
        text = (docs / name).read_text(encoding="utf-8")
        assert "{{" not in text
        assert "{%" not in text
        assert text.strip()
        for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            assert (docs / match).is_file(), match
