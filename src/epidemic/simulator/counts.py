"""Répartition entière de personnes, sans perdre ni créer d'individu."""

import math

import numpy as np


def split_count(
    n: int,
    probs: np.ndarray,
    *,
    stochastic: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Répartit n personnes selon des probabilités. La somme vaut exactement n."""
    if n < 0:
        raise ValueError("n doit être positif ou nul.")
    weights = np.asarray(probs, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Les probabilités doivent être un vecteur.")
    if n == 0:
        return np.zeros(weights.shape[0], dtype=np.int64)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Les probabilités sont toutes nulles.")
    weights = weights / total
    if stochastic:
        if rng is None:
            raise ValueError("Un générateur est requis en mode stochastique.")
        drawn = rng.multinomial(int(n), weights)
        return drawn.astype(np.int64)
    raw = weights * n
    floors = np.floor(raw).astype(np.int64)
    short = int(n - int(floors.sum()))
    # Plus grand reste d'abord. Le tri est stable pour que le résultat soit fixe.
    fractional = raw - floors
    order = np.argsort(-fractional, kind="mergesort")
    for rank in range(short):
        floors[int(order[rank])] += 1
    return floors


def round_half_up(value: float) -> int:
    """Arrondit un nombre positif au plus proche, moitié vers le haut."""
    if value <= 0:
        return 0
    return int(math.floor(value + 0.5))


def scale_to_total(counts: np.ndarray, total: int) -> np.ndarray:
    """Ramène des effectifs à une somme donnée, en gardant les proportions."""
    values = np.asarray(counts, dtype=float)
    current = float(values.sum())
    if total < 0:
        raise ValueError("Le total doit être positif ou nul.")
    if total == 0 or current <= 0:
        return np.zeros(values.shape[0], dtype=np.int64)
    return split_count(total, values / current, stochastic=False, rng=None)
