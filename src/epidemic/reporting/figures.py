"""Figures du rapport et de la documentation, écrites depuis les calculs."""

import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from epidemic.config import ReportingConfig, load_inference_config
from epidemic.model.age_structure import load_age_structure
from epidemic.model.dynamics import ICU, H, initial_state, run_dynamics
from epidemic.model.priors import (
    AGE_SLOPE_MU,
    AGE_SLOPE_SIGMA,
    BETA_MU,
    BETA_SIGMA,
    DEATH_ICU_MU_LOC,
    DEATH_WARD_MU_LOC,
    E0_MU,
    E0_SIGMA,
    HOSP_MU_LOC,
    HOSPITAL_MU,
    HOSPITAL_SIGMA,
    ICU_MU,
    ICU_MU_LOC,
    ICU_SIGMA,
    INCUBATION_MU,
    INCUBATION_SIGMA,
    SENSITIVITY_ALPHA,
    SENSITIVITY_BETA,
    SEVERITY_MU_SCALE,
    SEVERITY_TAU_SCALE,
    SYMPTOM_ALPHA,
    SYMPTOM_BETA,
    SYMPTOMATIC_MU,
    SYMPTOMATIC_SIGMA,
)
from epidemic.reporting.priors_table import density_samples
from epidemic.viz.model_diagram import INFERENCE_EDGES


def prior_trajectory_bundle(settings: ReportingConfig) -> dict[str, np.ndarray]:
    """Tirages a priori passés dans la dynamique du modèle, puis bruités."""
    inference = load_inference_config()
    structure = load_age_structure(inference.age_groups)
    generator = np.random.default_rng(settings.seed)
    population = jnp.asarray(structure.populations)
    contacts = jnp.asarray(structure.contacts)
    age_score = np.asarray(structure.age_score, dtype=float)
    n_days = settings.prior_days
    positives = []
    hospitals = []
    criticals = []
    for _ in range(settings.prior_draws):
        incubation = float(generator.lognormal(INCUBATION_MU, INCUBATION_SIGMA))
        symptomatic = float(generator.lognormal(SYMPTOMATIC_MU, SYMPTOMATIC_SIGMA))
        hospital_days = float(generator.lognormal(HOSPITAL_MU, HOSPITAL_SIGMA))
        icu_days = float(generator.lognormal(ICU_MU, ICU_SIGMA))
        p_sym = float(generator.beta(SYMPTOM_ALPHA, SYMPTOM_BETA))
        sensitivity = float(generator.beta(SENSITIVITY_ALPHA, SENSITIVITY_BETA))
        p_test = 0.5 + 0.5 * float(generator.beta(2.0, 2.0))
        beta = float(generator.lognormal(BETA_MU, BETA_SIGMA))
        exposed = float(generator.lognormal(E0_MU, E0_SIGMA))
        infectious = exposed * float(generator.beta(2.0, 5.0))
        state0 = initial_state(
            population,
            jnp.asarray(exposed),
            jnp.asarray(infectious),
            jnp.asarray(p_sym),
        )
        trajectory = run_dynamics(
            state0,
            jnp.full(n_days, beta),
            jnp.zeros((n_days, structure.n_groups)),
            population=population,
            contacts=contacts,
            sigma=jnp.asarray(1.0 / incubation),
            gamma=jnp.asarray(1.0 / symptomatic),
            gamma_h=jnp.asarray(1.0 / hospital_days),
            gamma_icu=jnp.asarray(1.0 / icu_days),
            symptomatic_fraction=jnp.asarray(p_sym),
            p_hosp=jnp.asarray(_group_probability(generator, age_score, HOSP_MU_LOC)),
            p_icu=jnp.asarray(_group_probability(generator, age_score, ICU_MU_LOC)),
            p_death_ward=jnp.asarray(
                _group_probability(generator, age_score, DEATH_WARD_MU_LOC)
            ),
            p_death_icu=jnp.asarray(
                _group_probability(generator, age_score, DEATH_ICU_MU_LOC)
            ),
            asymptomatic_infectiousness=jnp.asarray(float(generator.beta(2.0, 2.0))),
            process_sigma=jnp.asarray(0.0),
        )
        state = np.asarray(trajectory["state"])
        new_symptomatic = np.asarray(trajectory["new_symptomatic"]).sum(axis=1)
        mean_positive = np.clip(sensitivity * p_test * new_symptomatic, 0.0, None)
        positives.append(
            _gamma_poisson(generator, mean_positive, inference.fixed_concentration)
        )
        hospitals.append(state[:, :, H].sum(axis=1))
        criticals.append(state[:, :, ICU].sum(axis=1))
    return {
        "day": np.arange(1, n_days + 1),
        "num_positive": np.vstack(positives),
        "hospital": np.vstack(hospitals),
        "critical": np.vstack(criticals),
    }


def save_prior_densities(path: Path, settings: ReportingConfig) -> None:
    """Grille des densités a priori les plus lues."""
    samples = density_samples(settings.seed)
    columns = 4
    rows = math.ceil(len(samples) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11, 2.4 * rows))
    flat = np.atleast_1d(axes).ravel()
    for axis, (label, values) in zip(flat, samples, strict=False):
        axis.hist(values, bins=40, color="#1f4e79", alpha=0.85)
        axis.set_title(label, fontsize=9)
    for axis in flat[len(samples) :]:
        axis.axis("off")
    figure.suptitle("Lois a priori utilisées par le modèle")
    figure.tight_layout()
    _write(figure, path, settings.figure_dpi)


def save_compartments(path: Path, settings: ReportingConfig) -> None:
    """Schéma des compartiments du modèle d'inférence, observation en pointillés."""
    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 6)
    axis.axis("off")
    positions = {
        "S": (1, 4),
        "E": (3, 4),
        "Ia": (5, 5),
        "Is": (5, 3),
        "H": (8, 3),
        "ICU": (10.5, 3),
        "R": (13, 4.5),
        "D": (13, 2),
        "Test": (7, 1),
        "Positif": (10, 1),
        "FauxNegatif": (13, 1),
    }
    for name, (x_pos, y_pos) in positions.items():
        dashed = name in ("Test", "Positif", "FauxNegatif")
        patch = FancyBboxPatch(
            (x_pos - 0.7, y_pos - 0.35),
            1.5,
            0.7,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#f4f7fb" if not dashed else "#fff8e8",
            edgecolor="#1f4e79",
            linestyle="--" if dashed else "-",
        )
        axis.add_patch(patch)
        axis.text(x_pos, y_pos, name, ha="center", va="center", fontsize=8)
    for source, target, label, style in INFERENCE_EDGES:
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        axis.annotate(
            "",
            xy=(x1 - 0.75, y1),
            xytext=(x0 + 0.75, y0),
            arrowprops={
                "arrowstyle": "->",
                "color": "#335577",
                "linestyle": "--" if style == "dashed" else "-",
                "lw": 0.8,
            },
        )
        axis.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, label, fontsize=6, ha="center")
    axis.set_title("Modèle d'inférence : flux et observation")
    figure.tight_layout()
    _write(figure, path, settings.figure_dpi)


def save_prior_paths(
    path: Path,
    bundle: dict[str, np.ndarray],
    settings: ReportingConfig,
) -> None:
    """Éventail a priori des cas confirmés et des hospitalisés."""
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    _fan(axes[0], bundle["day"], bundle["num_positive"], "Cas confirmés")
    _fan(axes[1], bundle["day"], bundle["hospital"], "Hospitalisés")
    axes[1].set_xlabel("Jour")
    figure.suptitle("Trajectoires jugées plausibles avant les données")
    figure.tight_layout()
    _write(figure, path, settings.figure_dpi)


def save_reliability(
    path: Path,
    rows: list[dict[str, float]],
    settings: ReportingConfig,
) -> None:
    """Probabilité annoncée contre fréquence observée."""
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], color="#888888", lw=1, label="Diagonale")
    announced = [row["announced_mean"] for row in rows]
    observed = [row["observed_rate"] for row in rows]
    sizes = [20 + 4 * row["count"] for row in rows]
    axis.scatter(announced, observed, s=sizes, color="#1f4e79", zorder=3)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Probabilité annoncée")
    axis.set_ylabel("Fréquence observée")
    axis.set_title("Fiabilité de la référence naïve")
    axis.legend(frameon=False)
    figure.tight_layout()
    _write(figure, path, settings.figure_dpi)


def save_costs(
    path: Path,
    labels: list[str],
    costs: list[float],
    settings: ReportingConfig,
) -> None:
    """Coût journalier de chaque contrainte, lu dans le YAML du jeu."""
    figure, axis = plt.subplots(figsize=(7, 4))
    order = np.argsort(costs)
    ordered_labels = [labels[index] for index in order]
    ordered_costs = [costs[index] for index in order]
    axis.barh(ordered_labels, ordered_costs, color="#1f4e79")
    axis.set_xlabel("Coût par jour")
    axis.set_title("Coût de contrainte")
    figure.tight_layout()
    _write(figure, path, settings.figure_dpi)


def _group_probability(
    generator: np.random.Generator, age_score: np.ndarray, loc: float
) -> np.ndarray:
    mu = generator.normal(loc, SEVERITY_MU_SCALE)
    slope = math.exp(generator.normal(AGE_SLOPE_MU, AGE_SLOPE_SIGMA))
    tau = abs(generator.normal(0.0, SEVERITY_TAU_SCALE))
    raw = generator.normal(0.0, 1.0, size=age_score.shape[0])
    logits = np.clip(mu + slope * age_score + tau * raw, -12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _gamma_poisson(
    generator: np.random.Generator,
    mean: np.ndarray,
    concentration: float,
) -> np.ndarray:
    """Bruit binomiale négative, écrit comme un mélange gamma-Poisson."""
    observed = np.zeros(mean.shape, dtype=float)
    mask = mean > 0
    if not np.any(mask):
        return observed
    lam = generator.gamma(concentration, mean[mask] / concentration)
    observed[mask] = generator.poisson(lam)
    return observed


def _fan(axis: plt.Axes, day: np.ndarray, samples: np.ndarray, title: str) -> None:
    low90, low50, median, high50, high90 = np.quantile(
        samples, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0
    )
    axis.fill_between(day, low90, high90, color="#1f4e79", alpha=0.15, label="90 %")
    axis.fill_between(day, low50, high50, color="#1f4e79", alpha=0.30, label="50 %")
    axis.plot(day, median, color="#1f4e79", label="médiane")
    axis.set_ylabel(title)
    axis.legend(frameon=False, fontsize=8)


def _write(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
