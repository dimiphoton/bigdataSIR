"""Éventails : médiane, bande à 50 % et bande à 90 %."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fan:
    """Quantiles d'une série. L'axe des jours est en numéro de jeu, à partir de 1."""

    day: np.ndarray
    low90: np.ndarray
    low50: np.ndarray
    median: np.ndarray
    high50: np.ndarray
    high90: np.ndarray


def fan_from_draws(
    day: np.ndarray,
    samples: np.ndarray,
    masses: tuple[float, ...] = (0.5, 0.9),
) -> Fan:
    """Résume des tirages de forme (tirages, jours)."""
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("Les tirages ont la forme (tirages, jours), avec au moins un.")
    if day.shape != (samples.shape[1],):
        raise ValueError("L'axe des jours ne couvre pas la série.")
    ordered = tuple(sorted(masses))
    if len(ordered) != 2 or not 0 < ordered[0] < ordered[1] < 1:
        raise ValueError("Il faut deux masses d'intervalle, entre 0 et 1, distinctes.")
    inner, outer = ordered

    def edges(mass: float) -> tuple[np.ndarray, np.ndarray]:
        tail = (1.0 - mass) / 2.0
        low, high = np.quantile(samples, [tail, 1.0 - tail], axis=0)
        return np.asarray(low, dtype=float), np.asarray(high, dtype=float)

    low50, high50 = edges(inner)
    low90, high90 = edges(outer)
    median = np.asarray(np.quantile(samples, 0.5, axis=0), dtype=float)
    return Fan(
        day=np.asarray(day, dtype=int),
        low90=low90,
        low50=low50,
        median=median,
        high50=high50,
        high90=high90,
    )
