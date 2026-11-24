"""État entier de la ville, par tranche d'âge."""

from dataclasses import dataclass

import numpy as np


def _as_float(value: float | list[int]) -> float:
    """Lit un nombre dans un JSON déjà parsé."""
    if isinstance(value, list):
        raise ValueError("Un nombre était attendu.")
    return float(value)


def _as_int_array(values: list[list[int]] | list[int], *, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != ndim:
        raise ValueError(f"Un tableau à {ndim} dimensions était attendu.")
    return array


@dataclass
class EpidemicState:
    """Comptes entiers. Les stades servent aux durées gamma (niveau 3)."""

    s: np.ndarray
    e: np.ndarray
    i_asymptomatic: np.ndarray
    i_presymptomatic: np.ndarray
    i_symptomatic_untested: np.ndarray
    i_symptomatic_negative: np.ndarray
    i_symptomatic_positive: np.ndarray
    hospital: np.ndarray
    icu: np.ndarray
    recovered: np.ndarray
    dead: np.ndarray

    def total(self) -> int:
        """Somme de tous les compartiments."""
        return int(
            self.s.sum()
            + self.e.sum()
            + self.i_asymptomatic.sum()
            + self.i_presymptomatic.sum()
            + self.i_symptomatic_untested.sum()
            + self.i_symptomatic_negative.sum()
            + self.i_symptomatic_positive.sum()
            + self.hospital.sum()
            + self.icu.sum()
            + self.recovered.sum()
            + self.dead.sum()
        )

    def to_dict(self) -> dict[str, list[int] | list[list[int]]]:
        """Convertit l'état en listes JSON."""
        return {
            "s": self.s.astype(int).tolist(),
            "e": self.e.astype(int).tolist(),
            "i_asymptomatic": self.i_asymptomatic.astype(int).tolist(),
            "i_presymptomatic": self.i_presymptomatic.astype(int).tolist(),
            "i_symptomatic_untested": self.i_symptomatic_untested.astype(int).tolist(),
            "i_symptomatic_negative": self.i_symptomatic_negative.astype(int).tolist(),
            "i_symptomatic_positive": self.i_symptomatic_positive.astype(int).tolist(),
            "hospital": self.hospital.astype(int).tolist(),
            "icu": self.icu.astype(int).tolist(),
            "recovered": self.recovered.astype(int).tolist(),
            "dead": self.dead.astype(int).tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, list[int] | list[list[int]]]) -> "EpidemicState":
        """Recharge un état sauvegardé."""
        return cls(
            s=_as_int_array(raw["s"], ndim=1),
            e=_as_int_array(raw["e"], ndim=2),
            i_asymptomatic=_as_int_array(raw["i_asymptomatic"], ndim=2),
            i_presymptomatic=_as_int_array(raw["i_presymptomatic"], ndim=2),
            i_symptomatic_untested=_as_int_array(raw["i_symptomatic_untested"], ndim=2),
            i_symptomatic_negative=_as_int_array(raw["i_symptomatic_negative"], ndim=2),
            i_symptomatic_positive=_as_int_array(raw["i_symptomatic_positive"], ndim=2),
            hospital=_as_int_array(raw["hospital"], ndim=2),
            icu=_as_int_array(raw["icu"], ndim=2),
            recovered=_as_int_array(raw["recovered"], ndim=1),
            dead=_as_int_array(raw["dead"], ndim=1),
        )


@dataclass
class LatentNoise:
    """Bruit lent et cas confirmés par dépistage élargi."""

    log_beta_deviation: float
    p_test: float
    flu_mean: float
    extra_confirmed: np.ndarray

    def copy(self) -> "LatentNoise":
        """Copie indépendante, pour ne pas modifier l'original."""
        return LatentNoise(
            log_beta_deviation=self.log_beta_deviation,
            p_test=self.p_test,
            flu_mean=self.flu_mean,
            extra_confirmed=self.extra_confirmed.copy(),
        )

    def to_dict(self) -> dict[str, float | list[int]]:
        """Convertit le bruit latent en JSON."""
        return {
            "log_beta_deviation": self.log_beta_deviation,
            "p_test": self.p_test,
            "flu_mean": self.flu_mean,
            "extra_confirmed": self.extra_confirmed.astype(int).tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, float | list[int]]) -> "LatentNoise":
        """Recharge le bruit latent."""
        confirmed = raw["extra_confirmed"]
        if not isinstance(confirmed, list):
            raise ValueError("extra_confirmed doit être une liste.")
        return cls(
            log_beta_deviation=_as_float(raw["log_beta_deviation"]),
            p_test=_as_float(raw["p_test"]),
            flu_mean=_as_float(raw["flu_mean"]),
            extra_confirmed=_as_int_array(confirmed, ndim=1),
        )
