"""Probabilités de saturation, pics et décès, à partir de trajectoires déjà tirées."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskSummary:
    """Résumé d'un ensemble de trajectoires. Les jours sont des numéros de jeu."""

    p_hospital: float
    p_critical: float
    p_saturation: float
    first_day: np.ndarray
    days_saturated: np.ndarray
    peak_hospital: np.ndarray
    peak_hospital_day: np.ndarray
    peak_critical: np.ndarray
    peak_critical_day: np.ndarray
    p_rt_now: float
    p_rt_end: float
    cumulative_deaths: np.ndarray


def summarize(
    hospital: np.ndarray,
    critical: np.ndarray,
    death_increments: np.ndarray,
    rt: np.ndarray,
    day: np.ndarray,
    *,
    max_hospitalized: float,
    max_critical: float,
    decision_index: int,
) -> RiskSummary:
    """Calcule les probabilités demandées pour la décision.

    ``hospital`` et ``critical`` ont la forme (tirages, jours). Un jour est
    en saturation si l'un des deux stocks dépasse son plafond.
    """
    _check_pair(hospital, critical)
    if death_increments.shape != hospital.shape or rt.shape != hospital.shape:
        raise ValueError("Décès, Rt, H et ICU n'ont pas la même forme.")
    if day.shape != (hospital.shape[1],):
        raise ValueError("L'axe des jours ne couvre pas la série.")
    if decision_index < 0 or decision_index >= hospital.shape[1]:
        raise ValueError("Le jour de décision est hors de la série.")

    over_h = hospital > max_hospitalized
    over_c = critical > max_critical
    saturated = over_h | over_c
    ever = saturated.any(axis=1)
    # Le minimum des jours saturés. +inf s'il n'y en a pas, puis NaN.
    first = np.where(saturated, day, np.inf).min(axis=1)
    first = np.where(ever, first, np.nan)
    peak_h, peak_h_day = _peak(hospital, day)
    peak_c, peak_c_day = _peak(critical, day)
    deaths = np.cumsum(np.clip(death_increments, 0.0, None), axis=1)[:, -1]
    return RiskSummary(
        p_hospital=float(np.mean(over_h.any(axis=1))),
        p_critical=float(np.mean(over_c.any(axis=1))),
        p_saturation=float(np.mean(ever)),
        first_day=first.astype(float),
        days_saturated=saturated.sum(axis=1).astype(float),
        peak_hospital=peak_h,
        peak_hospital_day=peak_h_day,
        peak_critical=peak_c,
        peak_critical_day=peak_c_day,
        p_rt_now=float(np.mean(rt[:, decision_index] < 1.0)),
        p_rt_end=float(np.mean(rt[:, -1] < 1.0)),
        cumulative_deaths=deaths.astype(float),
    )


def probability_over(
    series: np.ndarray,
    start: int,
    horizon: int,
    threshold: float,
) -> float:
    """Part des tirages qui dépassent le seuil dans la fenêtre qui commence à start."""
    if series.ndim != 2 or series.shape[0] < 1:
        raise ValueError("La série a la forme (tirages, jours).")
    if horizon < 1:
        raise ValueError("L'horizon compte au moins un jour.")
    if start < 0 or start >= series.shape[1]:
        raise ValueError("Le début de la fenêtre est hors de la série.")
    window = series[:, start : start + horizon]
    return float(np.mean(np.any(window > threshold, axis=1)))


def in_central_interval(samples: np.ndarray, realized: float, mass: float) -> bool:
    """Dit si une valeur réalisée tombe dans l'intervalle central."""
    if samples.ndim != 1 or samples.shape[0] < 1:
        raise ValueError("Il faut un vecteur de tirages.")
    if not 0 < mass < 1:
        raise ValueError("La masse de l'intervalle est entre 0 et 1.")
    tail = (1.0 - mass) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return bool(low <= realized <= high)


def weekly_probability_below(samples: np.ndarray, level: float = 1.0) -> np.ndarray:
    """Part des tirages dont le Rt médian de la semaine est sous le niveau."""
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("Les tirages ont la forme (tirages, jours).")
    n_weeks = samples.shape[1] // 7
    if n_weeks == 0:
        return np.array([], dtype=float)
    # Une semaine = 7 colonnes consécutives, à partir du premier jour.
    weeks = samples[:, : n_weeks * 7].reshape(samples.shape[0], n_weeks, 7)
    week_level = np.median(weeks, axis=2)
    return np.mean(week_level < level, axis=0).astype(float)


def saturation_by_day(first_day: np.ndarray, day: np.ndarray) -> np.ndarray:
    """P(saturation au plus tard à chaque jour). NaN veut dire jamais."""
    if first_day.ndim != 1 or first_day.size < 1:
        raise ValueError("Il faut un jour de première saturation par tirage.")
    if day.ndim != 1 or day.size < 1:
        raise ValueError("L'axe des jours est un vecteur.")
    # NaN <= t est faux : un tirage jamais saturé ne compte pas.
    return np.mean(first_day[:, None] <= day[None, :], axis=0).astype(float)


def death_interval(deaths: np.ndarray, mass: float = 0.9) -> tuple[float, float, float]:
    """Médiane et bornes de l'intervalle central des décès cumulés."""
    if deaths.ndim != 1 or deaths.shape[0] < 1:
        raise ValueError("Il faut un vecteur de décès cumulés.")
    tail = (1.0 - mass) / 2.0
    low, median, high = np.quantile(deaths, [tail, 0.5, 1.0 - tail])
    return float(median), float(low), float(high)


def _check_pair(hospital: np.ndarray, critical: np.ndarray) -> None:
    if hospital.ndim != 2 or hospital.shape[0] < 1:
        raise ValueError("H a la forme (tirages, jours), avec au moins un tirage.")
    if critical.shape != hospital.shape:
        raise ValueError("H et ICU n'ont pas la même forme.")


def _peak(series: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.argmax(series, axis=1)
    rows = np.arange(series.shape[0])
    return series[rows, index].astype(float), day[index].astype(float)
