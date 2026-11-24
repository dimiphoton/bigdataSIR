"""Matrices de contact construites à partir des statistiques de la ville."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epidemic.config import SimulatorConfig, resolve_path
from epidemic.simulator.counts import scale_to_total

LOCATIONS = ("home", "school", "work", "community")


@dataclass
class AgeGroup:
    """Une tranche d'âge et sa place à l'école et au travail."""

    name: str
    low: int
    high: int
    population: int
    school_share: float
    work_share: float
    ages: tuple[tuple[int, int], ...]


@dataclass
class ContactMatrices:
    """Contacts moyens d'une personne d'âge i avec le groupe j, par lieu."""

    groups: tuple[AgeGroup, ...]
    matrices: np.ndarray
    weights: np.ndarray
    population: int
    mean_household_size: float
    degrees: dict[str, float]

    def fraction_at_or_above(self, group_index: int, threshold: int) -> float:
        """Part du groupe dont l'âge est supérieur ou égal au seuil.

        Le code 80 du fichier d'âges regroupe les 80 ans et plus. Il est
        compté comme ayant 80 ans : un seuil supérieur à 80 l'exclut.
        """
        rows = self.groups[group_index].ages
        total = sum(count for _, count in rows)
        if total == 0:
            return 0.0
        above = sum(count for age, count in rows if age >= threshold)
        return above / total

    def mean_weighted_contacts(self) -> float:
        """Nombre moyen de contacts pondérés par personne et par jour."""
        total = 0.0
        for index, group in enumerate(self.groups):
            contacts = 0.0
            for location in range(len(LOCATIONS)):
                contacts += float(
                    self.weights[location] * self.matrices[location, index].sum()
                )
            total += contacts * group.population
        return total / self.population


def _rows(path: Path) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        left, right = text.split(",")
        parsed.append((left.strip(), right.strip()))
    return parsed


def _mean_capped_degree(sizes: np.ndarray, weights: np.ndarray, cap: float) -> float:
    degree = np.minimum(cap, np.maximum(0.0, sizes - 1.0))
    return float(np.average(degree, weights=weights))


def _household_degree(path: Path) -> tuple[float, float]:
    people = 0
    households = 0
    contact_mass = 0
    for size_text, count_text in _rows(path):
        size = int(size_text)
        count = int(count_text)
        people += size * count
        households += count
        contact_mass += count * size * (size - 1)
    if people == 0 or households == 0:
        raise ValueError("Le fichier des ménages est vide.")
    return contact_mass / people, people / households


def _place_degree(path: Path, cap: float) -> float:
    sizes: list[float] = []
    weights: list[float] = []
    for range_text, count_text in _rows(path):
        low_text, high_text = range_text.split("-")
        size = (float(low_text) + float(high_text)) / 2.0
        count = int(count_text)
        sizes.append(size)
        weights.append(size * count)
    if not sizes:
        raise ValueError(f"Aucun groupe dans {path.name}.")
    return _mean_capped_degree(np.asarray(sizes), np.asarray(weights), cap)


def _community_degree(path: Path, cap: float) -> float:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("communities.json doit contenir une liste.")
    sizes = np.asarray([float(item["totalPopulation"]) for item in raw], dtype=float)
    if float(sizes.sum()) <= 0:
        raise ValueError("Les communautés n'ont pas de population.")
    return _mean_capped_degree(sizes, sizes, cap)


def _share(
    ages: tuple[tuple[int, int], ...], low: int, high: int, participation: float
) -> float:
    total = sum(count for _, count in ages)
    if total == 0:
        return 0.0
    inside = sum(count for age, count in ages if low <= age <= high)
    return participation * inside / total


def build_contacts(config: SimulatorConfig) -> ContactMatrices:
    """Lit stats/*.json et construit les quatre matrices de contact."""
    stats_dir = resolve_path(config.stats_dir)
    age_rows = [(int(age), int(count)) for age, count in _rows(stats_dir / "age.json")]
    raw_counts = np.asarray([count for _, count in age_rows], dtype=float)
    scaled = scale_to_total(raw_counts, config.population)
    age_counts = {age: int(scaled[index]) for index, (age, _) in enumerate(age_rows)}

    grouped_ages: list[list[tuple[int, int]]] = [[] for _ in config.age_bins]
    for age, count in age_counts.items():
        if count == 0:
            continue
        placed = False
        for index, age_bin in enumerate(config.age_bins):
            if age_bin.low <= age <= age_bin.high:
                grouped_ages[index].append((age, count))
                placed = True
                break
        if not placed:
            raise ValueError(
                f"L'âge {age} ne tombe dans aucune tranche de configs/simulator.yaml."
            )

    groups: list[AgeGroup] = []
    for index, age_bin in enumerate(config.age_bins):
        ages = tuple(grouped_ages[index])
        population = sum(count for _, count in ages)
        groups.append(
            AgeGroup(
                name=age_bin.name,
                low=age_bin.low,
                high=age_bin.high,
                population=population,
                school_share=_share(
                    ages,
                    config.school_ages.low,
                    config.school_ages.high,
                    config.school_participation,
                ),
                work_share=_share(
                    ages,
                    config.work_ages.low,
                    config.work_ages.high,
                    config.work_participation,
                ),
                ages=ages,
            )
        )

    home_degree, mean_household_size = _household_degree(stats_dir / "households.json")
    school_degree = _place_degree(
        stats_dir / "schools.json", config.contact_caps.school
    )
    work_degree = _place_degree(stats_dir / "workplaces.json", config.contact_caps.work)
    community_degree = _community_degree(
        stats_dir / "communities.json", config.contact_caps.community
    )

    group_count = len(groups)
    matrices = np.zeros((len(LOCATIONS), group_count, group_count), dtype=float)
    populations = np.asarray([group.population for group in groups], dtype=float)
    school_mass = populations * np.asarray([group.school_share for group in groups])
    work_mass = populations * np.asarray([group.work_share for group in groups])
    school_total = float(school_mass.sum())
    work_total = float(work_mass.sum())
    city = float(config.population)

    for i, group in enumerate(groups):
        for j in range(group_count):
            matrices[0, i, j] = home_degree * populations[j] / city
            if school_total > 0:
                matrices[1, i, j] = (
                    group.school_share * school_degree * school_mass[j] / school_total
                )
            if work_total > 0:
                matrices[2, i, j] = (
                    group.work_share * work_degree * work_mass[j] / work_total
                )
            matrices[3, i, j] = community_degree * populations[j] / city

    weights = np.asarray(
        [
            config.location_weights.home,
            config.location_weights.school,
            config.location_weights.work,
            config.location_weights.community,
        ],
        dtype=float,
    )
    return ContactMatrices(
        groups=tuple(groups),
        matrices=matrices,
        weights=weights,
        population=config.population,
        mean_household_size=mean_household_size,
        degrees={
            "home": home_degree,
            "school": school_degree,
            "work": work_degree,
            "community": community_degree,
        },
    )
