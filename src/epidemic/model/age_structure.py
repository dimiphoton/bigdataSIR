"""Groupes d'âge du modèle d'inférence et matrice de contact agrégée.

Les matrices détaillées viennent des statistiques publiques de la ville.
Le modèle ne lit pas l'état caché du simulateur.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from epidemic.config import AgeBin, SeverityRates, load_simulator_config
from epidemic.simulator.contacts import ContactMatrices, build_contacts

# Deux, trois ou quatre groupes. Le dernier commence toujours à 60 ans.
_SCHEMES: dict[int, tuple[tuple[str, int, int], ...]] = {
    2: (("0-59", 0, 59), ("60+", 60, 120)),
    3: (("0-17", 0, 17), ("18-59", 18, 59), ("60+", 60, 120)),
    4: (("0-17", 0, 17), ("18-49", 18, 49), ("50-59", 50, 59), ("60+", 60, 120)),
}


@dataclass(frozen=True)
class AgeStructure:
    """Population et contacts moyens entre les groupes du modèle."""

    names: tuple[str, ...]
    populations: np.ndarray
    contacts: np.ndarray
    age_score: np.ndarray
    older: np.ndarray

    @property
    def n_groups(self) -> int:
        return len(self.names)

    @property
    def population(self) -> float:
        return float(self.populations.sum())


def bins_for(n_groups: int) -> tuple[AgeBin, ...]:
    """Découpage configurable. Deux groupes au minimum, quatre au maximum."""
    try:
        rows = _SCHEMES[n_groups]
    except KeyError as exc:
        raise ValueError("Le modèle accepte 2, 3 ou 4 groupes d'âge.") from exc
    return tuple(AgeBin(name=name, low=low, high=high) for name, low, high in rows)


def _built_contacts(n_groups: int) -> ContactMatrices:
    """Matrices par lieu, sur les groupes du modèle d'inférence.

    ``build_contacts`` exige une gravité par tranche. Cette gravité n'entre
    pas dans les matrices : les taux du modèle ont leurs propres priors.
    On met donc une valeur neutre, ignorée ensuite.
    """
    simulator = load_simulator_config()
    bins = bins_for(n_groups)
    unused_severity = {
        age_bin.name: SeverityRates(
            p_hosp=0.01,
            p_icu=0.1,
            p_death_ward=0.01,
            p_death_icu=0.2,
        )
        for age_bin in bins
    }
    config = simulator.model_copy(
        update={"age_bins": list(bins), "severity": unused_severity}
    )
    return build_contacts(config)


@lru_cache(maxsize=4)
def load_contact_matrices(n_groups: int = 2) -> ContactMatrices:
    """Contacts par lieu. Le modèle d'inférence les additionne ensuite."""
    return _built_contacts(n_groups)


@lru_cache(maxsize=4)
def load_age_structure(n_groups: int = 2) -> AgeStructure:
    """Construit les contacts pondérés pour le nombre de groupes demandé."""
    bins = bins_for(n_groups)
    built = _built_contacts(n_groups)
    weights = np.asarray(built.weights, dtype=float)
    # Somme sur les lieux, avec le poids de transmission de chaque lieu.
    contacts = np.tensordot(
        weights, np.asarray(built.matrices, dtype=float), axes=(0, 0)
    )
    populations = np.asarray([group.population for group in built.groups], dtype=float)
    if populations.shape != (n_groups,) or contacts.shape != (n_groups, n_groups):
        raise ValueError("La matrice de contact n'a pas la taille attendue.")
    if float(populations.min()) <= 0:
        raise ValueError("Un groupe d'âge n'a personne.")
    midpoints = np.asarray(
        [(age_bin.low + age_bin.high) / 2.0 for age_bin in bins], dtype=float
    )
    span = float(midpoints.max() - midpoints.min())
    if span <= 0:
        raise ValueError("Les groupes d'âge n'ont pas des âges distincts.")
    age_score = (midpoints - float(midpoints.min())) / span
    older = np.asarray([age_bin.low >= 60 for age_bin in bins], dtype=bool)
    return AgeStructure(
        names=tuple(group.name for group in built.groups),
        populations=populations,
        contacts=contacts,
        age_score=age_score,
        older=older,
    )
