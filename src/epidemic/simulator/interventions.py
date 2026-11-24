"""Règles de combinaison des interventions publiques."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from epidemic.config import ScenariosConfig
from epidemic.simulator.contacts import LOCATIONS, ContactMatrices

LABELS = {
    "mask": "Masque",
    "case_isolation": "Isolement des cas",
    "household_quarantine": "Quarantaine du ménage",
    "lockdown": "Confinement",
    "social_distancing": "Distanciation sociale",
}


@dataclass(frozen=True)
class AppliedPolicy:
    """Intervention déjà acceptée, vue par le simulateur un jour donné."""

    name: str
    duration_days: int | None = None
    age_threshold: int | None = None


@dataclass
class ReductionTables:
    """Réductions effectives par âge et par lieu."""

    ordinary: np.ndarray
    targeted: np.ndarray
    has_case_isolation: bool
    has_household_quarantine: bool
    isolation_duration_days: int | None


def _pair_key(left: str, right: str) -> tuple[str, str]:
    if left <= right:
        return (left, right)
    return (right, left)


def assert_compatible(names: list[str], scenarios: ScenariosConfig) -> None:
    """Refuse les paires interdites et les noms inconnus."""
    known = set(scenarios.interventions)
    banned = {_pair_key(left, right) for left, right in scenarios.incompatible_pairs}
    seen: set[str] = set()
    for name in names:
        if name not in known:
            raise ValueError(f"Intervention inconnue : {name}.")
        if name in seen:
            raise ValueError(f"Intervention en double : {name}.")
        seen.add(name)
    for left, right in combinations(names, 2):
        if _pair_key(left, right) in banned:
            raise ValueError(
                f"{LABELS.get(left, left)} incompatible avec "
                f"{LABELS.get(right, right)}."
            )


def assert_parameters(
    policies: list[AppliedPolicy], scenarios: ScenariosConfig
) -> None:
    """Vérifie que chaque mesure a le paramètre prévu par les consignes."""
    for policy in policies:
        kind = scenarios.interventions[policy.name].parameter
        if kind is None and (
            policy.duration_days is not None or policy.age_threshold is not None
        ):
            raise ValueError(
                f"{LABELS.get(policy.name, policy.name)} n'a pas de paramètre."
            )
        if kind == "duration_days":
            if policy.duration_days is None or policy.duration_days < 1:
                raise ValueError(
                    f"{LABELS.get(policy.name, policy.name)} demande une durée."
                )
        if kind == "age_threshold":
            if policy.age_threshold is None or policy.age_threshold < 0:
                raise ValueError(
                    f"{LABELS.get(policy.name, policy.name)} demande un âge seuil."
                )


def _location_reduction(scenarios: ScenariosConfig, name: str, location: str) -> float:
    reduction = scenarios.interventions[name].contact_reduction
    if reduction is None:
        return 0.0
    value = getattr(reduction, location)
    return float(value)


def mask_multiplier(
    policies: list[AppliedPolicy],
    scenarios: ScenariosConfig,
    compliance: float,
) -> float:
    """Le masque multiplie le taux d'infection, hors règle du maximum."""
    if not any(policy.name == "mask" for policy in policies):
        return 1.0
    reduction = scenarios.interventions["mask"].infection_rate_reduction
    if reduction is None:
        raise ValueError("Le masque n'a pas de réduction du taux d'infection.")
    return 1.0 - compliance * reduction


def reduction_tables(
    policies: list[AppliedPolicy],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    compliance: float,
) -> ReductionTables:
    """Réduction ordinaire (après respect) et réduction d'une personne qui s'isole."""
    names = [policy.name for policy in policies]
    assert_compatible(names, scenarios)
    group_count = len(contacts.groups)
    nominal = np.zeros((group_count, len(LOCATIONS)), dtype=float)
    targeted_extra = np.zeros_like(nominal)
    has_case_isolation = False
    has_household_quarantine = False
    isolation_duration: int | None = None

    for policy in policies:
        if policy.name == "mask":
            continue
        if policy.name in ("case_isolation", "household_quarantine"):
            if policy.name == "case_isolation":
                has_case_isolation = True
            else:
                has_household_quarantine = True
            isolation_duration = policy.duration_days
            for location_index, location in enumerate(LOCATIONS):
                value = _location_reduction(scenarios, policy.name, location)
                targeted_extra[:, location_index] = np.maximum(
                    targeted_extra[:, location_index],
                    value,
                )
            continue
        if policy.name == "lockdown":
            for location_index, location in enumerate(LOCATIONS):
                value = _location_reduction(scenarios, policy.name, location)
                nominal[:, location_index] = np.maximum(
                    nominal[:, location_index], value
                )
            continue
        if policy.name == "social_distancing":
            if policy.age_threshold is None:
                raise ValueError("La distanciation demande un âge seuil.")
            for group_index in range(group_count):
                fraction = contacts.fraction_at_or_above(
                    group_index, policy.age_threshold
                )
                for location_index, location in enumerate(LOCATIONS):
                    value = fraction * _location_reduction(
                        scenarios, policy.name, location
                    )
                    nominal[group_index, location_index] = max(
                        float(nominal[group_index, location_index]),
                        value,
                    )
            continue
        raise ValueError(f"Intervention inconnue : {policy.name}.")

    # Une personne qui s'isole applique la plus forte réduction, pas le produit.
    targeted = np.maximum(nominal, targeted_extra)
    ordinary = compliance * nominal
    return ReductionTables(
        ordinary=ordinary,
        targeted=targeted,
        has_case_isolation=has_case_isolation,
        has_household_quarantine=has_household_quarantine,
        isolation_duration_days=isolation_duration,
    )


def pair_factor(reduction_a: float, reduction_b: float) -> float:
    """Un contact est réduit une seule fois, par la personne la plus contrainte."""
    return 1.0 - max(reduction_a, reduction_b)
