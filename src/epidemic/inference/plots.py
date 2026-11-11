"""Figures de diagnostic. Le backend est non interactif."""

from pathlib import Path
from typing import Any

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KEY_PARAMETERS = (
    "beta0",
    "incubation_days",
    "symptomatic_days",
    "symptomatic_fraction",
    "sensitivity",
    "p_test",
    "flu_mean",
    "e0",
)


def _posterior(idata: az.InferenceData) -> Any:
    # Les groupes ArviZ sont dynamiques : le stub ne les déclare pas.
    loose: Any = idata
    return loose.posterior


def _existing(names: tuple[str, ...] | list[str], idata: az.InferenceData) -> list[str]:
    return [name for name in names if name in _posterior(idata)]


def save_trace_plot(idata: az.InferenceData, path: Path) -> None:
    """Traces et densités des paramètres principaux."""
    names = _existing(KEY_PARAMETERS, idata)
    if not names or "sample_stats" not in idata.groups():
        return
    axes = az.plot_trace(idata, var_names=names)
    figure = np.asarray(axes).ravel()[0].figure
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_prior_posterior_plot(
    posterior: az.InferenceData,
    prior: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Densités a priori et a posteriori sur les mêmes axes."""
    drawn = _posterior(posterior)
    names = [name for name in KEY_PARAMETERS if name in drawn and name in prior]
    if not names:
        return
    columns = 2
    rows = (len(names) + 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(8, 2.4 * rows))
    flat = np.atleast_1d(axes).ravel()
    for axis, name in zip(flat, names, strict=False):
        posterior_values = np.asarray(drawn[name]).ravel()
        prior_values = np.asarray(prior[name]).ravel()
        axis.hist(prior_values, bins=30, density=True, alpha=0.45, label="a priori")
        axis.hist(
            posterior_values, bins=30, density=True, alpha=0.45, label="a posteriori"
        )
        axis.set_title(name)
    for axis in flat[len(names) :]:
        axis.axis("off")
    flat[0].legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
