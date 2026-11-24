"""Schémas et graphes de la phase 4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from epidemic.forecast.bands import fan_from_draws
from epidemic.forecast.project import LatentSegment
from epidemic.model.dynamics import N_COMPARTMENTS
from epidemic.model.priors import prior_catalogue
from epidemic.viz.dag import prior_table_markdown, render_numpyro_dag
from epidemic.viz.model_diagram import (
    compartment_dot,
    compartment_mermaid,
    inference_node_count,
)
from epidemic.viz.tracking import (
    doubling_sentence,
    growth_sentence,
    log_slope,
    plot_incidence,
    plot_pareto,
    plot_reservoir,
)


def test_compartment_diagrams_follow_the_two_models(tmp_path: Path) -> None:
    assert inference_node_count() == N_COMPARTMENTS
    inference = compartment_mermaid("inference")
    simulator = compartment_mermaid("simulator")
    assert "Ip" in simulator
    assert "Ip" not in inference
    assert "FauxNegatif" in inference
    assert 'style="dashed"' in compartment_dot("simulator")
    table = prior_table_markdown()
    assert "incubation" in table.lower()
    assert len(prior_catalogue()) >= 5
    assert render_numpyro_dag(tmp_path / "dag") in (True, False)


def test_tracking_figures_and_doubling_time() -> None:
    series = np.power(2.0, np.arange(20))
    assert abs(log_slope(series) - np.log(2.0)) < 0.05
    assert "doublement" in doubling_sentence(np.log(2.0))
    assert "division par deux" in doubling_sentence(-np.log(2.0))
    repeated = np.vstack([series, series, series])
    assert "doublement" in growth_sentence(repeated)
    assert "intervalle" in growth_sentence(repeated)
    day = np.arange(1, 11)
    samples = np.vstack([np.linspace(10, 40, 10), np.linspace(12, 55, 10)])
    fan = fan_from_draws(day, samples)
    figure = plot_incidence(samples[0], day, fan, bulletin_day=4, spans=[(3, 6)])
    assert figure.axes
    plt.close(figure)
    pareto = plot_pareto(
        ["Aucune mesure", "Masque"],
        np.array([0.0, 10.0]),
        np.array([0.4, 0.05]),
        np.array([True, True]),
        ["fixe", "fixe"],
    )
    assert pareto.axes
    plt.close(pareto)


def _latent(new_infections: np.ndarray, infectious: np.ndarray) -> LatentSegment:
    blank = np.zeros_like(new_infections)
    return LatentSegment(
        new_infections=new_infections,
        new_symptomatic=blank,
        new_hospital=blank,
        new_deaths=blank,
        hospitalized=blank,
        critical=blank,
        susceptible=np.full_like(new_infections, 1_000_000.0),
        infectious=infectious,
        rt=np.ones_like(new_infections),
        beta_effective=np.ones_like(new_infections),
        beta=np.ones_like(new_infections),
    )


def test_reservoir_plot_follows_the_unit_slope() -> None:
    """Phase exponentielle : pente 1, puis marqueur le jour où dI/I baisse."""
    time = np.arange(18, dtype=float)
    infectious = 15.0 * np.exp(0.08 * time)
    new_infections = 0.3 * infectious
    new_infections = new_infections.copy()
    new_infections[12:] *= 0.4
    jitter = np.linspace(0.85, 1.15, 4)[:, None]
    segment = _latent(new_infections * jitter, infectious * jitter)
    with pytest.raises(ValueError):
        plot_reservoir(_latent(np.ones((2, 2)), np.ones((2, 2))))
    figure = plot_reservoir(segment, intervention_days=[13])
    phase, ratio = figure.axes
    assert phase.get_xscale() == "log"
    assert phase.get_yscale() == "log"
    assert ratio.get_yscale() == "log"
    slope_line = next(line for line in phase.lines if line.get_label() == "Pente 1")
    x_line = np.asarray(slope_line.get_xdata(), dtype=float)
    y_line = np.asarray(slope_line.get_ydata(), dtype=float)
    slope = np.log(y_line[-1] / y_line[0]) / np.log(x_line[-1] / x_line[0])
    assert abs(slope - 1.0) < 1e-6
    assert np.allclose(y_line, 0.3 * x_line)
    notes = " ".join(text.get_text() for text in phase.texts)
    assert "S/N" in notes
    assert "beta" in notes
    assert "J13" in notes
    ratio_line = next(line for line in ratio.lines if line.get_label() == "dI/I")
    ratio_values = np.asarray(ratio_line.get_ydata(), dtype=float)
    assert np.allclose(ratio_values[:8], 0.3)
    assert ratio_values[-1] < 0.2
    assert any(len(collection.get_offsets()) for collection in phase.collections)
    plt.close(figure)
