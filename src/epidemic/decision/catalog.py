"""Liste les combinaisons d'interventions autorisées à un palier."""

from dataclasses import dataclass
from itertools import combinations, product

from epidemic.config import DecisionConfig, ScenariosConfig
from epidemic.simulator.interventions import LABELS, AppliedPolicy, assert_compatible


@dataclass(frozen=True)
class Scenario:
    """Un paquet de mesures, appliqué tel quel à partir du jour d'effet."""

    key: str
    label: str
    policies: tuple[AppliedPolicy, ...]


def scenario_key(policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy]) -> str:
    """Clé stable. L'ordre des mesures ne change pas la clé."""
    ordered = sorted(
        policies,
        key=lambda policy: (
            policy.name,
            policy.duration_days or 0,
            policy.age_threshold or 0,
        ),
    )
    if not ordered:
        return "aucune"
    return "|".join(
        f"{policy.name}:{policy.duration_days}:{policy.age_threshold}"
        for policy in ordered
    )


def scenario_label(policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy]) -> str:
    """Libellé lisible, dans l'ordre des consignes."""
    if not policies:
        return "Aucune mesure"
    order = {name: index for index, name in enumerate(LABELS)}
    ordered = sorted(policies, key=lambda policy: order.get(policy.name, 99))
    return " + ".join(_one_label(policy) for policy in ordered)


def enumerate_scenarios(
    allowed: list[str],
    scenarios: ScenariosConfig,
    settings: DecisionConfig,
) -> list[Scenario]:
    """Toutes les combinaisons compatibles, avec la grille des durées et des âges."""
    names = _unique(allowed)
    for name in names:
        if name not in scenarios.interventions:
            raise ValueError(f"Intervention inconnue : {name}.")
    built = [Scenario("aucune", "Aucune mesure", ())]
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            if not _compatible(combo, scenarios):
                continue
            for policies in _expand(combo, scenarios, settings):
                built.append(
                    Scenario(scenario_key(policies), scenario_label(policies), policies)
                )
    return built


def _unique(names: list[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def _compatible(names: tuple[str, ...], scenarios: ScenariosConfig) -> bool:
    try:
        assert_compatible(list(names), scenarios)
    except ValueError:
        return False
    return True


def _expand(
    names: tuple[str, ...],
    scenarios: ScenariosConfig,
    settings: DecisionConfig,
) -> list[tuple[AppliedPolicy, ...]]:
    choices: list[list[AppliedPolicy]] = []
    for name in names:
        kind = scenarios.interventions[name].parameter
        if kind is None:
            choices.append([AppliedPolicy(name)])
        elif kind == "duration_days":
            choices.append(
                [
                    AppliedPolicy(name, duration_days=day)
                    for day in settings.isolation_durations
                ]
            )
        else:
            choices.append(
                [
                    AppliedPolicy(name, age_threshold=age)
                    for age in settings.age_thresholds
                ]
            )
    return [tuple(item) for item in product(*choices)]


def _one_label(policy: AppliedPolicy) -> str:
    label = LABELS.get(policy.name, policy.name)
    if policy.duration_days is not None:
        return f"{label} {policy.duration_days} j"
    if policy.age_threshold is not None:
        return f"{label} {policy.age_threshold} ans"
    return label
