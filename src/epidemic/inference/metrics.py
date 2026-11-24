"""Scores de prévision : CRPS, couverture, log-score, identifiabilité."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FRENCH_STATUS = {
    "well_identified": "bien identifié",
    "partial": "partiellement identifié",
    "prior_dominated": "dicté par le prior",
}


def crps_ensemble(samples: np.ndarray, truth: float) -> float:
    """CRPS d'un ensemble de tirages. Un seul tirage se réduit à l'erreur absolue."""
    values = np.asarray(samples, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("L'ensemble de prévision est vide.")
    absolute = float(np.mean(np.abs(values - truth)))
    if values.size == 1:
        return absolute
    pairwise = float(np.mean(np.abs(values[:, None] - values[None, :])))
    return absolute - 0.5 * pairwise


def interval_covers(samples: np.ndarray, truth: float, level: float) -> bool:
    """Vrai si truth est dans l'intervalle central de niveau level."""
    if not 0.0 < level < 1.0:
        raise ValueError("Le niveau de l'intervalle doit être entre 0 et 1.")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(np.asarray(samples, dtype=float), [tail, 1.0 - tail])
    return bool(low <= truth <= high)


def hdi_width(samples: np.ndarray, level: float = 0.9) -> float:
    """Largeur de l'intervalle central."""
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(np.asarray(samples, dtype=float), [tail, 1.0 - tail])
    return float(high - low)


def coverage_rate(
    truths: dict[str, float],
    samples: dict[str, np.ndarray],
    level: float = 0.9,
) -> float:
    """Part des paramètres dont l'intervalle couvre la vraie valeur."""
    if not truths:
        raise ValueError("Aucun paramètre à vérifier.")
    covered = [
        interval_covers(samples[name], value, level) for name, value in truths.items()
    ]
    return float(np.mean(covered))


def negative_binomial_logpmf(count: float, mean: float, concentration: float) -> float:
    """Log-densité d'une binomiale négative NB2, au sens de NumPyro."""
    if count < 0 or mean <= 0 or concentration <= 0:
        return -math.inf
    whole = int(round(count))
    success = concentration / (concentration + mean)
    return (
        math.lgamma(whole + concentration)
        - math.lgamma(concentration)
        - math.lgamma(whole + 1)
        + concentration * math.log(success)
        + whole * math.log(1.0 - success)
    )


def log_score(samples: np.ndarray, truth: float) -> float:
    """Log-score d'une binomiale négative calée sur les tirages."""
    values = np.asarray(samples, dtype=float).ravel()
    mean = float(np.mean(values))
    variance = float(np.var(values))
    if mean <= 0:
        return 0.0 if truth == 0 else -math.inf
    excess = max(variance - mean, 1e-6)
    concentration = max(mean**2 / excess, 1e-6)
    return negative_binomial_logpmf(truth, mean, concentration)


def classify_width_ratio(ratio: float, well_below: float, partial_below: float) -> str:
    """Classe un paramètre selon le rétrécissement de son intervalle."""
    if ratio < well_below:
        return "well_identified"
    if ratio < partial_below:
        return "partial"
    return "prior_dominated"


def identifiability_table(
    ratios: dict[str, float],
    *,
    well_below: float,
    partial_below: float,
) -> pd.DataFrame:
    """Tableau : paramètre, rapport de largeurs, lecture."""
    rows = []
    for name, ratio in ratios.items():
        status = classify_width_ratio(ratio, well_below, partial_below)
        rows.append(
            {
                "parameter": name,
                "width_ratio": ratio,
                "status": status,
                "lecture": FRENCH_STATUS[status],
            }
        )
    return pd.DataFrame(rows)


def ambiguity_reduction(
    cases_only: dict[str, float],
    with_hospital: dict[str, float],
    with_positivity: dict[str, float],
) -> pd.DataFrame:
    """Compare les largeurs quand on ajoute l'hôpital, puis le taux de positivité."""
    names = sorted(set(cases_only) & set(with_hospital) & set(with_positivity))
    rows = []
    for name in names:
        first = cases_only[name]
        second = with_hospital[name]
        third = with_positivity[name]
        rows.append(
            {
                "parameter": name,
                "cases_only": first,
                "with_hospital": second,
                "with_positivity": third,
                "hospital_shrinks": second < first,
                "positivity_shrinks": third < second,
            }
        )
    return pd.DataFrame(rows)
