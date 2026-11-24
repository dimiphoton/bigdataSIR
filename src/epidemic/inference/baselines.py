"""Références naïves : persistance et croissance exponentielle."""

import numpy as np


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Répète la dernière valeur observée."""
    if horizon < 1:
        raise ValueError("L'horizon doit être au moins 1.")
    if len(history) < 1:
        raise ValueError("L'historique est vide.")
    return np.repeat(float(history[-1]), horizon)


def exponential_forecast(
    history: np.ndarray, horizon: int, window: int = 14
) -> np.ndarray:
    """Prolonge la pente de log(1 + y) sur les derniers jours."""
    if horizon < 1:
        raise ValueError("L'horizon doit être au moins 1.")
    values = np.asarray(history, dtype=float)
    if values.size < 2:
        return persistence_forecast(values, horizon)
    used = values[-window:]
    time = np.arange(used.size, dtype=float)
    slope, intercept = np.polyfit(time, np.log1p(np.clip(used, 0.0, None)), deg=1)
    future_time = np.arange(used.size, used.size + horizon, dtype=float)
    return np.expm1(intercept + slope * future_time).clip(min=0.0)
