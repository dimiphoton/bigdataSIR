"""Couverture, Brier, CRPS et diagramme de fiabilité.

Les fonctions de ce module ne lisent pas l'état caché du simulateur.
Elles comparent un ensemble de prévisions à une série déjà publiée.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epidemic.inference.baselines import exponential_forecast, persistence_forecast
from epidemic.inference.metrics import crps_ensemble
from epidemic.series import ObservationSeries


@dataclass(frozen=True)
class ForecastHit:
    """Une prévision ponctuelle : un bulletin, un horizon, une série."""

    bulletin_day: int
    horizon: int
    series: str
    covered_low: bool
    covered_high: bool
    crps_persistence: float
    crps_exponential: float


@dataclass(frozen=True)
class EventScore:
    """Un événement binaire annoncé avec une probabilité."""

    bulletin_day: int
    horizon: int
    probability: float
    happened: bool


@dataclass(frozen=True)
class SeriesScore:
    """Tous les coups d'une même trajectoire publiée."""

    label: str
    hits: tuple[ForecastHit, ...]
    events: tuple[EventScore, ...]

    def coverage(self, high: bool) -> float:
        """Part des prévisions tombées dans l'intervalle demandé."""
        if not self.hits:
            raise ValueError(f"Aucune prévision pour {self.label}.")
        flags = [item.covered_high if high else item.covered_low for item in self.hits]
        return float(np.mean(flags))

    def mean_crps(self, exponential: bool) -> float:
        """CRPS moyen. exponential sélectionne la croissance, sinon la persistance."""
        if not self.hits:
            raise ValueError(f"Aucune prévision pour {self.label}.")
        values = [
            item.crps_exponential if exponential else item.crps_persistence
            for item in self.hits
        ]
        return float(np.mean(values))

    def brier(self) -> float:
        """Score de Brier des événements. 0 est parfait, 1 est tout faux."""
        if not self.events:
            raise ValueError(f"Aucun événement pour {self.label}.")
        probabilities = np.array([item.probability for item in self.events])
        outcomes = np.array([item.happened for item in self.events], dtype=float)
        return brier_score(probabilities, outcomes)


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Moyenne des écarts au carré entre la probabilité annoncée et 0 ou 1."""
    announced = np.asarray(probabilities, dtype=float).ravel()
    happened = np.asarray(outcomes, dtype=float).ravel()
    if announced.shape != happened.shape or announced.size == 0:
        raise ValueError("Probabilités et réalisations n'ont pas la même taille.")
    if np.any((announced < 0) | (announced > 1)):
        raise ValueError("Une probabilité annoncée est hors de [0, 1].")
    return float(np.mean((announced - happened) ** 2))


def interpret_coverage(empirical: float, nominal: float, tolerance: float) -> str:
    """Dit si les intervalles sont trop étroits, trop larges, ou bien calés."""
    if not 0 <= empirical <= 1 or not 0 < nominal < 1:
        raise ValueError("Les couvertures sont des parts entre 0 et 1.")
    if tolerance <= 0:
        raise ValueError("La tolérance est strictement positive.")
    gap = empirical - nominal
    if gap < -tolerance:
        return (
            "Les intervalles sont trop sûrs d'eux : "
            "la valeur réalisée tombe dedans moins souvent que le niveau annoncé."
        )
    if gap > tolerance:
        return (
            "Les intervalles sont trop prudents : "
            "la valeur réalisée tombe dedans plus souvent que le niveau annoncé."
        )
    return "La couverture est proche du niveau annoncé."


def naive_ensemble(
    history: np.ndarray,
    horizon: int,
    *,
    kind: str,
    n_draws: int,
    seed: int,
    window: int,
) -> np.ndarray:
    """Ensemble (tirages, horizon) autour d'une référence naïve.

    L'écart-type vient des erreurs à un jour sur la fenêtre récente.
    Il grandit comme la racine de l'horizon, comme une marche aléatoire.
    """
    if kind not in ("persistence", "exponential"):
        raise ValueError("La référence est persistence ou exponential.")
    if horizon < 1 or n_draws < 1:
        raise ValueError("L'horizon et le nombre de tirages sont au moins 1.")
    values = np.asarray(history, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("L'historique a au moins deux jours.")
    if kind == "persistence":
        point = persistence_forecast(values, horizon)
        residuals = np.diff(values[-window:])
    else:
        point = exponential_forecast(values, horizon, window)
        residuals = _exponential_residuals(values, window)
    sigma = float(np.std(residuals)) if residuals.size > 1 else 1.0
    sigma = max(sigma, 1e-6)
    generator = np.random.default_rng(seed)
    steps = np.arange(1, horizon + 1, dtype=float)
    noise = generator.normal(0.0, sigma, size=(n_draws, horizon)) * np.sqrt(steps)
    return np.clip(point + noise, 0.0, None)


def reliability_table(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    """Fréquence observée contre probabilité moyenne, par classe."""
    announced = np.asarray(probabilities, dtype=float).ravel()
    happened = np.asarray(outcomes, dtype=float).ravel()
    if announced.shape != happened.shape or announced.size == 0:
        raise ValueError("Le diagramme de fiabilité demande des événements.")
    if n_bins < 2:
        raise ValueError("Il faut au moins deux classes.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Le bord droit de la dernière classe inclut 1.
    classes = np.clip(np.digitize(announced, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = classes == index
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin": index,
                "announced_mean": float(announced[mask].mean()),
                "observed_rate": float(happened[mask].mean()),
                "count": count,
            }
        )
    if not rows:
        raise ValueError("Aucune classe du diagramme de fiabilité n'est remplie.")
    return pd.DataFrame(rows)


def score_observation_series(
    series: ObservationSeries,
    *,
    label: str,
    period: int,
    min_history: int,
    horizons: list[int],
    low_mass: float,
    high_mass: float,
    ensemble_size: int,
    seed: int,
    residual_window: int,
    series_names: list[str],
    max_hospitalized: float,
    event_horizon: int,
) -> SeriesScore:
    """Score les références naïves sur une série publiée."""
    hits: list[ForecastHit] = []
    events: list[EventScore] = []
    for name in series_names:
        values = _column(series, name)
        for horizon in horizons:
            for origin in _origins(values.size, period, min_history, horizon):
                history = values[:origin]
                realized = float(values[origin + horizon - 1])
                draw_seed = (
                    seed + origin * 1_000 + horizon * 10 + series_names.index(name)
                )
                persistence = naive_ensemble(
                    history,
                    horizon,
                    kind="persistence",
                    n_draws=ensemble_size,
                    seed=draw_seed,
                    window=residual_window,
                )
                exponential = naive_ensemble(
                    history,
                    horizon,
                    kind="exponential",
                    n_draws=ensemble_size,
                    seed=draw_seed + 1,
                    window=residual_window,
                )
                target = persistence[:, horizon - 1]
                hits.append(
                    ForecastHit(
                        bulletin_day=origin,
                        horizon=horizon,
                        series=name,
                        covered_low=_covers(target, realized, low_mass),
                        covered_high=_covers(target, realized, high_mass),
                        crps_persistence=crps_ensemble(target, realized),
                        crps_exponential=crps_ensemble(
                            exponential[:, horizon - 1], realized
                        ),
                    )
                )
        if name == "num_hospitalised":
            events.extend(
                _hospital_events(
                    values,
                    period=period,
                    min_history=min_history,
                    horizon=event_horizon,
                    ensemble_size=ensemble_size,
                    seed=seed,
                    residual_window=residual_window,
                    threshold=max_hospitalized,
                )
            )
    if not hits:
        raise ValueError(
            f"La série {label} est trop courte pour les horizons demandés."
        )
    return SeriesScore(label, tuple(hits), tuple(events))


def summary_rows(
    score: SeriesScore, labels: dict[str, str]
) -> list[dict[str, str | int | float]]:
    """Agrège les coups par série et par horizon."""
    rows: list[dict[str, str | int | float]] = []
    keys = {(item.series, item.horizon) for item in score.hits}
    for series_name, horizon in sorted(keys, key=lambda item: (item[0], item[1])):
        chosen = [
            item
            for item in score.hits
            if item.series == series_name and item.horizon == horizon
        ]
        rows.append(
            {
                "series": labels.get(series_name, series_name),
                "horizon": horizon,
                "n": len(chosen),
                "coverage_low": float(np.mean([item.covered_low for item in chosen])),
                "coverage_high": float(np.mean([item.covered_high for item in chosen])),
                "crps_persistence": float(
                    np.mean([item.crps_persistence for item in chosen])
                ),
                "crps_exponential": float(
                    np.mean([item.crps_exponential for item in chosen])
                ),
            }
        )
    return rows


def _covers(samples: np.ndarray, realized: float, mass: float) -> bool:
    tail = (1.0 - mass) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return bool(low <= realized <= high)


def _origins(n_days: int, period: int, min_history: int, horizon: int) -> list[int]:
    origins = []
    day = period
    while day + horizon <= n_days:
        if day >= min_history:
            origins.append(day)
        day += period
    return origins


def _column(series: ObservationSeries, name: str) -> np.ndarray:
    if not hasattr(series, name):
        raise ValueError(f"Série inconnue : {name}.")
    return np.asarray(getattr(series, name), dtype=float)


def _exponential_residuals(values: np.ndarray, window: int) -> np.ndarray:
    residuals = []
    start = max(2, values.size - window)
    for index in range(start, values.size):
        predicted = float(exponential_forecast(values[:index], 1, window)[-1])
        residuals.append(float(values[index]) - predicted)
    return np.asarray(residuals, dtype=float)


def _hospital_events(
    values: np.ndarray,
    *,
    period: int,
    min_history: int,
    horizon: int,
    ensemble_size: int,
    seed: int,
    residual_window: int,
    threshold: float,
) -> list[EventScore]:
    events = []
    for origin in _origins(values.size, period, min_history, horizon):
        future = values[origin : origin + horizon]
        happened = bool(np.any(future > threshold))
        ensemble = naive_ensemble(
            values[:origin],
            horizon,
            kind="persistence",
            n_draws=ensemble_size,
            seed=seed + origin,
            window=residual_window,
        )
        probability = float(np.mean(np.any(ensemble > threshold, axis=1)))
        events.append(EventScore(origin, horizon, probability, happened))
    return events
