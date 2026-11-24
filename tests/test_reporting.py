"""Scores, échelle verbale, bulletin et documentation générée."""

import re
from pathlib import Path

import numpy as np
import pytest

from epidemic.config import load_reporting_config
from epidemic.forecast.scoring import (
    brier_score,
    interpret_coverage,
    naive_ensemble,
    score_observation_series,
)
from epidemic.forecast.verbal import verbal_probability
from epidemic.reporting.bulletin import write_bulletin
from epidemic.reporting.render import render_docs, render_named
from epidemic.reporting.slide_titles import city_phrase, coverage_extent
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
    assert "dI/I" in text
    assert "réservoir" in text
    assert any("reservoir" in path.name for path in written)
    assert "{{" not in text
    assert "{%" not in text
    assert any(path.suffix == ".pdf" for path in written)


def test_city_phrase_and_coverage_extent() -> None:
    assert city_phrase(1_000_000) == "1 million"
    assert city_phrase(2_500_000) == "2,500,000"
    missed = coverage_extent([(0, 0.69), (1, 0.50), (2, 0.53), (3, 0.44)], 0.90, 0.05)
    assert not missed.holds
    assert not missed.partial
    assert missed.worst_level == 3
    assert missed.worst_coverage == 0.44
    partial = coverage_extent([(0, 0.90), (1, 0.88), (2, 0.40)], 0.90, 0.05)
    assert partial.partial
    assert partial.max_ok_level == 1
    held = coverage_extent([(0, 0.91), (1, 0.88)], 0.90, 0.05)
    assert held.holds
    assert held.max_ok_level == 1


def _title_context() -> dict[str, object]:
    return {
        "city_phrase": "1 million",
        "course_coverage_pct": 31,
        "n_forecasts": "80",
        "high_pct": 90,
        "calibration_holds": False,
        "calibration_partial": False,
        "max_ok_level": 0,
        "worst_level": 3,
        "worst_pct": 44,
        "posterior": {
            "available": False,
            "ok": False,
            "rhat_en": "unavailable",
            "divergences_en": "unavailable",
        },
    }


def test_slide_body_stays_under_the_word_cap() -> None:
    from epidemic.reporting.slide_titles import check_slide_text

    short = "# A title with five words here\n\nOne fact, answered.\n"
    check_slide_text(short, 40, "recruiter")
    long = "# A title with five words here\n\n" + " ".join(["word"] * 41) + "\n"
    with pytest.raises(ValueError, match="plafond"):
        check_slide_text(long, 40, "recruiter")


def test_slide_titles_follow_missing_results() -> None:
    context = _title_context()
    recruiter = render_named("slides/recruiter_titles.txt.j2", context).splitlines()
    technical = render_named("slides/technical_titles.txt.j2", context).splitlines()
    recruiter = [line for line in recruiter if line.strip()]
    technical = [line for line in technical if line.strip()]
    assert len(recruiter) == 7
    assert len(technical) == 12
    assert all(len(line.split()) >= 5 for line in recruiter + technical)
    assert "not estimated yet" in recruiter[2]
    assert "31% of 80" in recruiter[5]
    assert "NUTS" not in " ".join(recruiter)
    assert "not been run" in technical[4]
    assert "down to 44% at level 3" in technical[10]


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
    slides = docs / "slides"
    for name, count in (("recruiter_titles.txt", 7), ("technical_titles.txt", 12)):
        lines = [
            line
            for line in (slides / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == count
        assert all(len(line.split()) >= 5 for line in lines)
