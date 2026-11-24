"""Séries publiées, sans l'état caché du simulateur."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservationSeries:
    """Colonnes du cours, plus les nouveaux décès de chaque jour."""

    num_positive: np.ndarray
    num_tested: np.ndarray
    num_hospitalised: np.ndarray
    num_critical: np.ndarray
    death_increments: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.num_positive),
            len(self.num_tested),
            len(self.num_hospitalised),
            len(self.num_critical),
            len(self.death_increments),
        }
        if lengths != {self.n_days}:
            raise ValueError("Les séries observées n'ont pas la même longueur.")

    @property
    def n_days(self) -> int:
        return int(self.num_positive.shape[0])

    def cut(self, n_days: int) -> "ObservationSeries":
        """Garde les n_days premiers jours."""
        if n_days < 1 or n_days > self.n_days:
            raise ValueError("La coupe doit rester dans la série.")
        return ObservationSeries(
            self.num_positive[:n_days],
            self.num_tested[:n_days],
            self.num_hospitalised[:n_days],
            self.num_critical[:n_days],
            self.death_increments[:n_days],
        )


def observations_from_frame(frame: pd.DataFrame) -> ObservationSeries:
    """Lit un tableau au format de data.csv. Les décès cumulés deviennent des flux."""
    required = (
        "num_positive",
        "num_tested",
        "num_hospitalised",
        "num_critical",
        "num_fatalities",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}.")
    deaths = frame["num_fatalities"].to_numpy(dtype=float)
    previous = np.concatenate([np.zeros(1), deaths[:-1]])
    increments = np.clip(deaths - previous, 0.0, None)
    return ObservationSeries(
        num_positive=frame["num_positive"].to_numpy(dtype=float),
        num_tested=frame["num_tested"].to_numpy(dtype=float),
        num_hospitalised=frame["num_hospitalised"].to_numpy(dtype=float),
        num_critical=frame["num_critical"].to_numpy(dtype=float),
        death_increments=increments,
    )
