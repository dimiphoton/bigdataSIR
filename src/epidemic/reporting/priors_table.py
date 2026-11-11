"""Tableau des priors : loi, paramètres, intervalle à 95 % tiré dans le code."""

import math

import numpy as np

from epidemic.model.priors import (
    AGE_SLOPE_MU,
    AGE_SLOPE_SIGMA,
    BETA_MU,
    BETA_SIGMA,
    CONCENTRATION_MU,
    CONCENTRATION_SIGMA,
    DEATH_ICU_MU_LOC,
    DEATH_WARD_MU_LOC,
    E0_MU,
    E0_SIGMA,
    FLU_MU,
    FLU_SIGMA,
    HOSP_MU_LOC,
    HOSPITAL_MU,
    HOSPITAL_SIGMA,
    ICU_MU,
    ICU_MU_LOC,
    ICU_SIGMA,
    INCUBATION_MU,
    INCUBATION_SIGMA,
    PIECEWISE_BETA_SCALE,
    PROCESS_SIGMA_SCALE,
    SENSITIVITY_ALPHA,
    SENSITIVITY_BETA,
    SEVERITY_MU_SCALE,
    SYMPTOM_ALPHA,
    SYMPTOM_BETA,
    SYMPTOMATIC_MU,
    SYMPTOMATIC_SIGMA,
    WEEKLY_BETA_SCALE,
    prior_catalogue,
)
from epidemic.reporting.text import format_auto

# Quantile d'une loi normale pour un intervalle central à 95 %.
_Z95 = 1.959963984540054


def prior_rows(seed: int, sample_size: int = 20_000) -> list[dict[str, str]]:
    """Une ligne par prior du catalogue, avec un intervalle calculé."""
    generator = np.random.default_rng(seed)
    intervals = _intervals(generator, sample_size)
    rows = []
    for item in prior_catalogue():
        if item.label not in intervals:
            raise ValueError(f"Pas d'intervalle pour le prior « {item.label} ».")
        rows.append(
            {
                "label": item.label,
                "family": item.family,
                "parameters": item.parameters,
                "interval": intervals[item.label],
                "source": item.source,
                "justification": item.justification,
            }
        )
    return rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    """Tableau markdown. Les cellules ne doivent pas contenir de barre verticale."""
    header = (
        "| Paramètre | Loi | Paramètres | Intervalle à 95 % | Source | Justification |"
    )
    rule = "|---|---|---|---|---|---|"
    lines = [header, rule]
    for row in rows:
        cells = (
            row["label"],
            row["family"],
            row["parameters"],
            row["interval"],
            row["source"],
            row["justification"],
        )
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
    return "\n".join(lines)


def density_samples(
    seed: int, sample_size: int = 4_000
) -> list[tuple[str, np.ndarray]]:
    """Échantillons pour le graphe des lois, dans l'ordre du catalogue court."""
    generator = np.random.default_rng(seed)
    named = (
        (
            "Incubation (jours)",
            generator.lognormal(INCUBATION_MU, INCUBATION_SIGMA, sample_size),
        ),
        (
            "Symptômes (jours)",
            generator.lognormal(SYMPTOMATIC_MU, SYMPTOMATIC_SIGMA, sample_size),
        ),
        (
            "Fraction symptomatique",
            generator.beta(SYMPTOM_ALPHA, SYMPTOM_BETA, sample_size),
        ),
        (
            "Sensibilité",
            generator.beta(SENSITIVITY_ALPHA, SENSITIVITY_BETA, sample_size),
        ),
        (
            "Taux de dépistage",
            0.5 + 0.5 * generator.beta(2.0, 2.0, sample_size),
        ),
        ("β₀", generator.lognormal(BETA_MU, BETA_SIGMA, sample_size)),
        (
            "Séjour hôpital (jours)",
            generator.lognormal(HOSPITAL_MU, HOSPITAL_SIGMA, sample_size),
        ),
        ("Exposés initiaux", generator.lognormal(E0_MU, E0_SIGMA, sample_size)),
    )
    return [(label, values) for label, values in named]


def _intervals(generator: np.random.Generator, sample_size: int) -> dict[str, str]:
    return {
        "Durée d'incubation": _lognormal(INCUBATION_MU, INCUBATION_SIGMA, " j"),
        "Durée symptomatique": _lognormal(SYMPTOMATIC_MU, SYMPTOMATIC_SIGMA, " j"),
        "Fraction symptomatique": _sampled(
            generator.beta(SYMPTOM_ALPHA, SYMPTOM_BETA, sample_size)
        ),
        "Sensibilité des tests": _sampled(
            generator.beta(SENSITIVITY_ALPHA, SENSITIVITY_BETA, sample_size)
        ),
        "Taux de dépistage": _sampled(
            0.5 + 0.5 * generator.beta(2.0, 2.0, sample_size)
        ),
        "Spécificité des tests": "1 (fixée)",
        "Transmission β₀": _lognormal(BETA_MU, BETA_SIGMA, ""),
        "Volatilité hebdomadaire de β": _sampled(
            np.abs(generator.normal(0.0, WEEKLY_BETA_SCALE, sample_size))
        ),
        "Saut de β entre deux dates": _sampled(
            np.abs(generator.normal(0.0, PIECEWISE_BETA_SCALE, sample_size))
        ),
        "Contagiosité des asymptomatiques": _sampled(
            generator.beta(2.0, 2.0, sample_size)
        ),
        "Durée d'hospitalisation": _lognormal(HOSPITAL_MU, HOSPITAL_SIGMA, " j"),
        "Durée en soins intensifs": _lognormal(ICU_MU, ICU_SIGMA, " j"),
        "Probabilité d'hospitalisation": _sampled(
            _sigmoid_normal(generator, HOSP_MU_LOC, sample_size)
        ),
        "Pente d'âge de la gravité": _lognormal(AGE_SLOPE_MU, AGE_SLOPE_SIGMA, ""),
        "Passage en soins intensifs": _sampled(
            _sigmoid_normal(generator, ICU_MU_LOC, sample_size)
        ),
        "Décès en service": _sampled(
            _sigmoid_normal(generator, DEATH_WARD_MU_LOC, sample_size)
        ),
        "Décès en soins intensifs": _sampled(
            _sigmoid_normal(generator, DEATH_ICU_MU_LOC, sample_size)
        ),
        "Exposés initiaux": _lognormal(E0_MU, E0_SIGMA, ""),
        "Part initiale déjà infectieuse": _sampled(
            generator.beta(2.0, 5.0, sample_size)
        ),
        "Fond de syndromes grippaux": _lognormal(FLU_MU, FLU_SIGMA, ""),
        "Surdispersion des observations": _lognormal(
            CONCENTRATION_MU, CONCENTRATION_SIGMA, ""
        ),
        "Bruit de la dynamique latente": _sampled(
            np.abs(generator.normal(0.0, PROCESS_SIGMA_SCALE, sample_size))
        ),
    }


def _lognormal(mu: float, sigma: float, unit: str) -> str:
    median = math.exp(mu)
    low = math.exp(mu - _Z95 * sigma)
    high = math.exp(mu + _Z95 * sigma)
    return f"{format_auto(median)} ({format_auto(low)} à {format_auto(high)}{unit})"


def _sampled(values: np.ndarray) -> str:
    low, median, high = np.quantile(
        np.asarray(values, dtype=float), [0.025, 0.5, 0.975]
    )
    low_text = format_auto(float(low))
    high_text = format_auto(float(high))
    return f"{format_auto(float(median))} ({low_text} à {high_text})"


def _sigmoid_normal(
    generator: np.random.Generator, loc: float, sample_size: int
) -> np.ndarray:
    """Intervalle de la moyenne logit-normale, avant la pente d'âge."""
    logits = generator.normal(loc, SEVERITY_MU_SCALE, sample_size)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
