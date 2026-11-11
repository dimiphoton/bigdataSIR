"""Schémas et graphes de la phase 4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from epidemic.forecast.bands import fan_from_draws
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
