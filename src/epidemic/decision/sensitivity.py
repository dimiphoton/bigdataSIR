"""Sensibilité : alpha, respect des mesures, portée de l'isolement."""

from epidemic.config import DecisionConfig, ScenariosConfig
from epidemic.decision.recommend import Recommendation, ScoredOption, recommend
from epidemic.decision.reduction import ReachDraw, point_factor
from epidemic.simulator.contacts import ContactMatrices
from epidemic.simulator.interventions import AppliedPolicy


def by_alpha(
    options: list[ScoredOption], alphas: list[float]
) -> list[tuple[float, str | None]]:
    """Pour chaque seuil, le libellé retenu. None si aucune option ne passe."""
    rows: list[tuple[float, str | None]] = []
    for alpha in alphas:
        result: Recommendation = recommend(options, alpha)
        label = None if result.chosen is None else result.chosen.label
        rows.append((alpha, label))
    return rows


def factors_by_compliance(
    new_policies: tuple[AppliedPolicy, ...],
    current_policies: tuple[AppliedPolicy, ...],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    settings: DecisionConfig,
    compliances: list[float],
) -> list[tuple[float, float]]:
    """Facteur de transmission pour chaque taux de respect."""
    return [
        (
            compliance,
            point_factor(
                new_policies,
                current_policies,
                scenarios,
                contacts,
                settings,
                compliance=compliance,
            ),
        )
        for compliance in compliances
    ]


def isolation_factor(
    symptomatic_fraction: float,
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    settings: DecisionConfig,
    *,
    duration_days: int,
    compliance: float | None = None,
) -> float:
    """Facteur d'un isolement des cas, pour une fraction symptomatique donnée."""
    policies = (AppliedPolicy("case_isolation", duration_days=duration_days),)
    saved = settings.isolation_reach
    tweaked = settings.model_copy(
        update={
            "isolation_reach": saved.model_copy(
                update={"symptomatic_fraction": symptomatic_fraction}
            )
        }
    )
    return point_factor(
        policies, (), scenarios, contacts, tweaked, compliance=compliance
    )


def reference_reach(settings: DecisionConfig) -> ReachDraw:
    """Point de référence écrit dans le YAML."""
    reach = settings.isolation_reach
    return ReachDraw(
        reach.symptomatic_fraction,
        reach.confirmation_probability,
        reach.infectious_days,
    )
