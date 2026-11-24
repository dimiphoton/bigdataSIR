"""Règles qui réagissent aux observations, avec hystérésis.

La règle ne reçoit qu'un indicateur déjà calculé. Elle ne voit pas l'état latent.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from epidemic.config import DecisionConfig, PolicyGrid, ScenariosConfig
from epidemic.simulator.interventions import LABELS, AppliedPolicy, assert_compatible


@dataclass(frozen=True)
class ConditionalRule:
    """Active un paquet au-dessus du seuil haut, et le lève sous le seuil bas."""

    policies: tuple[AppliedPolicy, ...]
    indicator: str
    high: float
    low: float
    hold_days: int

    def __post_init__(self) -> None:
        if self.indicator not in ("hospital", "icu", "cases_7d", "positivity"):
            raise ValueError(f"Indicateur inconnu : {self.indicator}.")
        if self.low >= self.high:
            raise ValueError("Le seuil bas doit être inférieur au seuil haut.")
        if self.hold_days < 1:
            raise ValueError("L'hystérésis dure au moins un jour.")
        if not self.policies:
            raise ValueError("Une règle active au moins une mesure.")

    @property
    def label(self) -> str:
        names = ", ".join(
            LABELS.get(policy.name, policy.name) for policy in self.policies
        )
        return (
            f"{names} si {self.indicator} > {self.high:g}, "
            f"levée sous {self.low:g} pendant {self.hold_days} j"
        )

    @property
    def key(self) -> str:
        names = "|".join(policy.name for policy in self.policies)
        return f"{names}:{self.indicator}:{self.high:g}:{self.low:g}:{self.hold_days}"


@dataclass
class RuleMemory:
    """Mémoire par tirage. Les tableaux sont modifiés sur place."""

    active: np.ndarray
    days_on: np.ndarray
    days_below: np.ndarray
    on_day: np.ndarray
    off_day: np.ndarray


def fresh_memory(n_draws: int) -> RuleMemory:
    """Personne n'est encore sous la règle."""
    return RuleMemory(
        active=np.zeros(n_draws, dtype=bool),
        days_on=np.zeros(n_draws, dtype=int),
        days_below=np.zeros(n_draws, dtype=int),
        on_day=np.full(n_draws, -1, dtype=int),
        off_day=np.full(n_draws, -1, dtype=int),
    )


def apply_today(memory: RuleMemory, day: int, min_days: int) -> np.ndarray:
    """Applique les bascules prévues pour ce jour et renvoie qui est actif."""
    turn_on = memory.on_day == day
    memory.active = memory.active | turn_on
    memory.days_on = np.where(turn_on, 0, memory.days_on)
    memory.on_day = np.where(turn_on, -1, memory.on_day)
    turn_off = (memory.off_day == day) & (memory.days_on >= min_days)
    memory.active = np.where(turn_off, False, memory.active)
    memory.days_on = np.where(turn_off, 0, memory.days_on)
    memory.days_below = np.where(turn_off, 0, memory.days_below)
    memory.off_day = np.where(turn_off, -1, memory.off_day)
    memory.days_on = np.where(memory.active, memory.days_on + 1, 0)
    return memory.active.copy()


def schedule_today(
    memory: RuleMemory,
    day: int,
    indicator: np.ndarray,
    rule: ConditionalRule,
    delay: int,
    min_days: int,
) -> None:
    """Programme une bascule à partir de l'indicateur du jour. Effet après le délai."""
    if indicator.shape != memory.active.shape:
        raise ValueError("L'indicateur a une valeur par tirage.")
    can_on = (~memory.active) & (memory.on_day < 0) & (indicator > rule.high)
    memory.on_day = np.where(can_on, day + delay, memory.on_day)
    mature = memory.active & (memory.days_on >= min_days)
    below = mature & (indicator < rule.low)
    stayed = np.where(memory.active, 0, memory.days_below)
    memory.days_below = np.where(below, memory.days_below + 1, stayed)
    can_off = below & (memory.days_below >= rule.hold_days) & (memory.off_day < 0)
    memory.off_day = np.where(can_off, day + delay, memory.off_day)


def indicator_value(
    name: str,
    hospital: np.ndarray,
    critical: np.ndarray,
    positive: np.ndarray,
    tested: np.ndarray,
) -> np.ndarray:
    """Indicateur du dernier jour. Les séries ont la forme (tirages, jours)."""
    if name == "hospital":
        return hospital[:, -1]
    if name == "icu":
        return critical[:, -1]
    if name == "cases_7d":
        return positive[:, -7:].sum(axis=1)
    if name == "positivity":
        denominator = np.clip(tested[:, -7:].sum(axis=1), 1.0, None)
        return positive[:, -7:].sum(axis=1) / denominator
    raise ValueError(f"Indicateur inconnu : {name}.")


def iter_rules(
    allowed: list[str],
    current: tuple[AppliedPolicy, ...],
    scenarios: ScenariosConfig,
    settings: DecisionConfig,
    indicators: list[str] | None = None,
) -> list[ConditionalRule]:
    """Grille des seuils. Une règle ajoute une mesure qui n'est pas déjà en vigueur."""
    chosen = list(indicators or settings.policy_grid.default_indicators)
    current_names = {policy.name for policy in current}
    duration = settings.isolation_durations[len(settings.isolation_durations) // 2]
    age = settings.age_thresholds[len(settings.age_thresholds) // 2]
    rules: list[ConditionalRule] = []
    for name in allowed:
        if name in current_names or name not in scenarios.interventions:
            continue
        policies = (_policy(name, scenarios, duration, age),)
        if not _compatible(tuple(current_names | {name}), scenarios):
            continue
        for indicator in chosen:
            highs, lows = _levels(settings.policy_grid, indicator)
            for high in highs:
                for low in lows:
                    if low >= high:
                        continue
                    for hold in settings.policy_grid.hold_days:
                        rules.append(
                            ConditionalRule(
                                policies, indicator, float(high), float(low), hold
                            )
                        )
    return rules


@dataclass(frozen=True)
class RuleScore:
    """Résultat d'une règle sur les trajectoires."""

    rule: ConditionalRule
    p_saturation: float
    expected_cost: float
    feasible: bool


def search_thresholds(
    rules: list[ConditionalRule],
    evaluate: Callable[[ConditionalRule], tuple[float, float]],
    alpha: float,
) -> tuple[RuleScore, list[RuleScore]]:
    """Minimise le coût parmi les règles dont P(saturation) reste sous alpha."""
    if not rules:
        raise ValueError("La grille des règles est vide.")
    scores: list[RuleScore] = []
    for rule in rules:
        p_saturation, cost = evaluate(rule)
        scores.append(
            RuleScore(
                rule, float(p_saturation), float(cost), float(p_saturation) <= alpha
            )
        )
    feasible = [score for score in scores if score.feasible]
    pool = feasible or scores
    if feasible:
        best = min(
            pool,
            key=lambda score: (score.expected_cost, score.p_saturation, score.rule.key),
        )
    else:
        best = min(
            pool,
            key=lambda score: (score.p_saturation, score.expected_cost, score.rule.key),
        )
    return best, scores


def _policy(
    name: str, scenarios: ScenariosConfig, duration: int, age: int
) -> AppliedPolicy:
    kind = scenarios.interventions[name].parameter
    if kind == "duration_days":
        return AppliedPolicy(name, duration_days=duration)
    if kind == "age_threshold":
        return AppliedPolicy(name, age_threshold=age)
    return AppliedPolicy(name)


def _levels(grid: PolicyGrid, indicator: str) -> tuple[list[float], list[float]]:
    table = {
        "hospital": (grid.hospital_high, grid.hospital_low),
        "icu": (grid.icu_high, grid.icu_low),
        "cases_7d": (grid.cases_high, grid.cases_low),
        "positivity": (grid.positivity_high, grid.positivity_low),
    }
    if indicator not in table:
        raise ValueError(f"Indicateur inconnu : {indicator}.")
    return table[indicator]


def _compatible(names: tuple[str, ...], scenarios: ScenariosConfig) -> bool:
    try:
        assert_compatible(list(names), scenarios)
    except ValueError:
        return False
    return True
