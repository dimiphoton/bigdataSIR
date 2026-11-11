"""Traduit une intervention en matrice de contact du modèle d'inférence.

Le modèle n'a qu'un β. Une mesure devient donc une matrice, construite à
partir des contacts par lieu. Le rapport de cette matrice à la matrice sans
changement de politique est le facteur de réduction. L'incertitude sur les
poids des lieux est tirée, puis appliquée comme un facteur sur β.
"""

from dataclasses import dataclass

import numpy as np

from epidemic.config import DecisionConfig, ScenariosConfig
from epidemic.simulator.contacts import LOCATIONS, ContactMatrices
from epidemic.simulator.interventions import (
    AppliedPolicy,
    assert_compatible,
    assert_parameters,
    mask_multiplier,
    reduction_tables,
)


@dataclass(frozen=True)
class ReachDraw:
    """Un tirage de la part des infectieux touchés par l'isolement."""

    symptomatic_fraction: float
    confirmation_probability: float
    infectious_days: float


def aggregate(
    contacts: ContactMatrices, weights: np.ndarray | None = None
) -> np.ndarray:
    """Somme des lieux, pondérée. C'est la matrice qu'utilise le modèle."""
    used = contacts.weights if weights is None else np.asarray(weights, dtype=float)
    return np.tensordot(used, np.asarray(contacts.matrices, dtype=float), axes=(0, 0))


def transmission_mass(matrix: np.ndarray, populations: np.ndarray) -> float:
    """Contacts pondérés reçus dans la ville, en une journée."""
    row = np.asarray(matrix, dtype=float).sum(axis=1)
    return float(np.sum(row * populations))


def isolation_reach(
    policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    reach: ReachDraw,
    household_size: float,
) -> float:
    """Part de la population dont les contacts suivent la réduction ciblée."""
    targeted = next(
        (
            policy
            for policy in policies
            if policy.name in ("case_isolation", "household_quarantine")
        ),
        None,
    )
    if targeted is None or targeted.duration_days is None:
        return 0.0
    coverage = min(1.0, targeted.duration_days / reach.infectious_days)
    share = reach.symptomatic_fraction * reach.confirmation_probability * coverage
    if targeted.name == "household_quarantine":
        share *= household_size
    return float(np.clip(share, 0.0, 1.0))


def relative_contacts(
    new_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    current_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    compliance: float,
    reach_new: float,
    reach_current: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Matrice à utiliser après le changement, vu depuis la politique en cours.

    Si la politique ne change pas, on retrouve la matrice de base du modèle :
    β a posteriori porte déjà l'effet des mesures en place.
    """
    _check(new_policies, scenarios)
    _check(current_policies, scenarios)
    ratio, mask_ratio = _relative_retention(
        list(new_policies),
        list(current_policies),
        scenarios,
        contacts,
        compliance,
        reach_new,
        reach_current,
    )
    used = contacts.weights if weights is None else np.asarray(weights, dtype=float)
    total = np.zeros((len(contacts.groups), len(contacts.groups)), dtype=float)
    for location in range(len(LOCATIONS)):
        total += (
            contacts.matrices[location]
            * ratio[:, location : location + 1]
            * float(used[location])
        )
    return total * mask_ratio


def point_factor(
    new_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    current_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    settings: DecisionConfig,
    compliance: float | None = None,
) -> float:
    """Facteur de transmission au point de référence des hypothèses de portée."""
    compliance_value = settings_compliance(settings, compliance)
    reach = ReachDraw(
        settings.isolation_reach.symptomatic_fraction,
        settings.isolation_reach.confirmation_probability,
        settings.isolation_reach.infectious_days,
    )
    return _factor_at(
        new_policies,
        current_policies,
        scenarios,
        contacts,
        compliance_value,
        reach,
        reach,
        weights=None,
    )


def factor_samples(
    new_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    current_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    settings: DecisionConfig,
    *,
    n_draws: int,
    seed: int,
    compliance: float | None = None,
) -> np.ndarray:
    """Un facteur par tirage. Les parts de lieu et la portée de l'isolement bougent."""
    if n_draws < 1:
        raise ValueError("Il faut au moins un tirage de parts.")
    compliance_value = settings_compliance(settings, compliance)
    weights, reaches = _uncertainty_draws(contacts, settings, n_draws, seed)
    factors = np.empty(n_draws, dtype=float)
    for index in range(n_draws):
        factors[index] = _factor_at(
            new_policies,
            current_policies,
            scenarios,
            contacts,
            compliance_value,
            reaches[index],
            reaches[index],
            weights=weights[index],
        )
    return factors


def beta_scales(factors: np.ndarray) -> np.ndarray:
    """Ramène la moyenne des facteurs à 1, pour ne pas décaler la matrice moyenne."""
    mean = float(np.mean(factors))
    if mean <= 0:
        raise ValueError("Le facteur de transmission moyen n'est pas positif.")
    return factors / mean


def settings_compliance(settings: DecisionConfig, compliance: float | None) -> float:
    """Le taux demandé, sinon le premier de la liste, sinon celui des scénarios."""
    if compliance is not None:
        return compliance
    if settings.compliances:
        return float(settings.compliances[len(settings.compliances) // 2])
    return 0.9


def _factor_at(
    new_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    current_policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    compliance: float,
    reach_new: ReachDraw,
    reach_current: ReachDraw,
    weights: np.ndarray | None,
) -> float:
    populations = np.asarray(
        [group.population for group in contacts.groups], dtype=float
    )
    matrix = relative_contacts(
        new_policies,
        current_policies,
        scenarios,
        contacts,
        compliance,
        isolation_reach(new_policies, reach_new, contacts.mean_household_size),
        isolation_reach(current_policies, reach_current, contacts.mean_household_size),
        weights,
    )
    base = aggregate(contacts, weights)
    base_mass = transmission_mass(base, populations)
    if base_mass <= 0:
        raise ValueError("La matrice de contact de base est vide.")
    return transmission_mass(matrix, populations) / base_mass


def _uncertainty_draws(
    contacts: ContactMatrices,
    settings: DecisionConfig,
    n_draws: int,
    seed: int,
) -> tuple[np.ndarray, list[ReachDraw]]:
    raw = _raw_mass(contacts)
    transmission = contacts.weights * raw
    shares = transmission / transmission.sum()
    # La concentration règle l'écart autour des parts du YAML.
    alpha = np.clip(shares * settings.share_concentration, 1e-3, None)
    rng = np.random.default_rng(seed)
    drawn_shares = rng.dirichlet(alpha, size=n_draws)
    weights = drawn_shares / np.clip(raw, 1e-9, None)
    reach = settings.isolation_reach
    spread = settings.reach_spread
    symptomatic = np.clip(
        rng.normal(
            reach.symptomatic_fraction, spread.symptomatic_fraction, size=n_draws
        ),
        0.05,
        0.95,
    )
    confirmation = np.clip(
        rng.normal(
            reach.confirmation_probability,
            spread.confirmation_probability,
            size=n_draws,
        ),
        0.05,
        1.0,
    )
    days = np.clip(
        rng.normal(reach.infectious_days, spread.infectious_days, size=n_draws),
        1.0,
        30.0,
    )
    reaches = [
        ReachDraw(float(symptomatic[i]), float(confirmation[i]), float(days[i]))
        for i in range(n_draws)
    ]
    return weights, reaches


def _raw_mass(contacts: ContactMatrices) -> np.ndarray:
    populations = np.asarray(
        [group.population for group in contacts.groups], dtype=float
    )
    mass = np.zeros(len(LOCATIONS), dtype=float)
    for location in range(len(LOCATIONS)):
        row = contacts.matrices[location].sum(axis=1)
        mass[location] = float(np.sum(row * populations))
    return mass


def _relative_retention(
    new_policies: list[AppliedPolicy],
    current_policies: list[AppliedPolicy],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    compliance: float,
    reach_new: float,
    reach_current: float,
) -> tuple[np.ndarray, float]:
    new_retention, new_mask = _retention(
        new_policies, scenarios, contacts, compliance, reach_new
    )
    current_retention, current_mask = _retention(
        current_policies, scenarios, contacts, compliance, reach_current
    )
    ratio = new_retention / np.clip(current_retention, 1e-3, None)
    return ratio, new_mask / current_mask


def _retention(
    policies: list[AppliedPolicy],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    compliance: float,
    reach: float,
) -> tuple[np.ndarray, float]:
    tables = reduction_tables(policies, scenarios, contacts, compliance)
    general = 1.0 - tables.ordinary
    targeted = 1.0 - tables.targeted
    retention = (1.0 - reach) * general + reach * targeted
    return retention, mask_multiplier(policies, scenarios, compliance)


def _check(
    policies: tuple[AppliedPolicy, ...] | list[AppliedPolicy],
    scenarios: ScenariosConfig,
) -> None:
    if not policies:
        return
    assert_compatible([policy.name for policy in policies], scenarios)
    assert_parameters(list(policies), scenarios)
