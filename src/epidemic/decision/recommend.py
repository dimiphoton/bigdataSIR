"""Recommandation : le paquet le moins coûteux sous le seuil de saturation."""

from dataclasses import dataclass

import numpy as np

from epidemic.forecast.bands import Fan


@dataclass(frozen=True)
class ScoredOption:
    """Un scénario ou une règle, déjà chiffré."""

    key: str
    label: str
    kind: str
    unit_cost: float
    expected_cost: float
    factor: float
    p_hospital: float
    p_critical: float
    p_saturation: float
    deaths_median: float
    deaths_low: float
    deaths_high: float
    peak_hospital_median: float
    peak_hospital_day: float
    peak_critical_median: float
    peak_critical_day: float
    days_saturated_median: float
    p_rt_now: float
    p_rt_end: float
    first_day: np.ndarray
    weekly_rt_below: np.ndarray
    fans: dict[str, Fan]


@dataclass(frozen=True)
class Recommendation:
    """Le choix, la frontière, et les options qui respectent le seuil."""

    chosen: ScoredOption | None
    alpha: float
    feasible: tuple[ScoredOption, ...]
    pareto: tuple[ScoredOption, ...]
    message: str


def pareto_front(options: list[ScoredOption]) -> list[ScoredOption]:
    """Options non dominées : on minimise le coût cumulé et P(saturation)."""
    front: list[ScoredOption] = []
    for candidate in options:
        dominated = False
        for other in options:
            if other is candidate:
                continue
            cheaper = other.expected_cost <= candidate.expected_cost
            safer = other.p_saturation <= candidate.p_saturation
            strict = (
                other.expected_cost < candidate.expected_cost
                or other.p_saturation < candidate.p_saturation
            )
            if cheaper and safer and strict:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda option: (option.expected_cost, option.p_saturation))


def recommend(options: list[ScoredOption], alpha: float) -> Recommendation:
    """Retient le coût cumulé le plus bas parmi les options sous le seuil."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha est entre 0 et 1.")
    if not options:
        raise ValueError("Il n'y a aucune option à comparer.")
    feasible = tuple(
        sorted(
            (option for option in options if option.p_saturation <= alpha),
            key=_rank,
        )
    )
    front = tuple(pareto_front(options))
    if not feasible:
        safest = min(
            options, key=lambda option: (option.p_saturation, option.expected_cost)
        )
        message = (
            f"Toutes les options dépassent le seuil de {alpha:.0%} de saturation. "
            f"La plus sûre est « {safest.label} », "
            f"avec une probabilité de {safest.p_saturation:.0%}."
        )
        return Recommendation(None, alpha, (), front, message)
    chosen = feasible[0]
    message = (
        f"{chosen.label} : coût cumulé {chosen.expected_cost:.0f}, "
        f"probabilité de saturation {chosen.p_saturation:.0%}."
    )
    return Recommendation(chosen, alpha, feasible, front, message)


def _rank(option: ScoredOption) -> tuple[float, float, str]:
    return (option.expected_cost, option.p_saturation, option.key)
