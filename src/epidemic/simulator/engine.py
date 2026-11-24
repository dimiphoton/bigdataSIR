"""Simulateur de la réalité : paramètres cachés, observations publiques."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epidemic.config import (
    PriorsConfig,
    ScenariosConfig,
    SimulatorConfig,
    load_priors_config,
    load_scenarios_config,
    load_simulator_config,
    resolve_path,
)
from epidemic.simulator.contacts import ContactMatrices, build_contacts
from epidemic.simulator.dynamics import (
    PUBLISHED_COLUMNS,
    advance_state,
    initial_latent,
    initial_state,
)
from epidemic.simulator.interventions import AppliedPolicy
from epidemic.simulator.parameters import (
    TruthParameters,
    approximate_r0,
    draw_truth,
)
from epidemic.simulator.randomness import make_streams
from epidemic.simulator.state import EpidemicState, LatentNoise


def load_course_rows(path: Path) -> dict[int, dict[str, int]]:
    """Charge le data.csv du cours, indexé par jour."""
    frame = pd.read_csv(path)
    missing = [column for column in PUBLISHED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {missing}.")
    rows: dict[int, dict[str, int]] = {}
    for record in frame.to_dict(orient="records"):
        day = int(record["Day"])
        rows[day] = {column: int(record[column]) for column in PUBLISHED_COLUMNS}
    return rows


class RealitySimulator:
    """Ville simulée. Les observations publiées ne contiennent pas l'état caché."""

    def __init__(
        self,
        config: SimulatorConfig,
        priors: PriorsConfig,
        scenarios: ScenariosConfig,
        contacts: ContactMatrices,
        truth: TruthParameters,
        streams: dict[str, np.random.Generator],
        course: dict[int, dict[str, int]] | None = None,
    ) -> None:
        self.config = config
        self.priors = priors
        self.scenarios = scenarios
        self.contacts = contacts
        self.truth = truth
        self.streams = streams
        self.course = course or {}
        self.day = 0
        self.state: EpidemicState = initial_state(contacts, truth)
        self.latent: LatentNoise = initial_latent(contacts, truth)
        self.cumulative_hospitalizations = 0
        self.published_rows: list[dict[str, int]] = []
        self.truth_rows: list[dict[str, Any]] = []

    @classmethod
    def from_configs(
        cls,
        *,
        seed: int | None = None,
        level: int | None = None,
        observation_source: str | None = None,
        truth: TruthParameters | None = None,
    ) -> "RealitySimulator":
        """Construit un simulateur à partir des YAML."""
        config = load_simulator_config()
        updates: dict[str, Any] = {}
        if truth is not None and seed is None:
            seed = truth.seed
        if seed is not None:
            updates["seed"] = seed
        if level is not None:
            updates["stochasticity_level"] = level
        if observation_source is not None:
            updates["observation_source"] = observation_source
        if updates:
            config = config.model_copy(update=updates)
        priors = load_priors_config()
        scenarios = load_scenarios_config()
        contacts = build_contacts(config)
        streams = make_streams(config.seed)
        if truth is None:
            truth = draw_truth(config, priors, streams["parameters"])
        course = None
        if config.observation_source == "course":
            course = load_course_rows(resolve_path(config.course_data))
        return cls(config, priors, scenarios, contacts, truth, streams, course)

    def step(self, policies: list[AppliedPolicy] | None = None) -> dict[str, int]:
        """Avance d'un jour et renvoie la ligne publiée."""
        self.day += 1
        result = advance_state(
            self.state,
            self.latent,
            day=self.day,
            truth=self.truth,
            scenarios=self.scenarios,
            contacts=self.contacts,
            policies=list(policies or []),
            baseline_compliance=self.scenarios.compliance,
            streams=self.streams,
            cumulative_hospitalizations=self.cumulative_hospitalizations,
        )
        self.state = result.state
        self.latent = result.latent
        self.cumulative_hospitalizations = result.cumulative_hospitalizations
        published = dict(result.published)
        if self.day in self.course:
            published = dict(self.course[self.day])
        self.published_rows.append(published)
        self.truth_rows.append(result.truth_row)
        return published

    def run(
        self,
        days: int,
        policies: list[AppliedPolicy] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Enchaîne plusieurs jours avec les mêmes interventions."""
        for _ in range(days):
            self.step(policies)
        return self.observations(), self.truth_frame()

    def observations(self) -> pd.DataFrame:
        """Séries publiées, avec les colonnes de data.csv."""
        return pd.DataFrame(self.published_rows, columns=list(PUBLISHED_COLUMNS))

    def truth_frame(self) -> pd.DataFrame:
        """Séries cachées, pour la validation seulement."""
        return pd.DataFrame(self.truth_rows)

    def secret_payload(self) -> dict[str, Any]:
        """Paramètres secrets et un résumé des contacts. À ne pas montrer au joueur."""
        return {
            "version": 1,
            "parameters": self.truth.to_dict(),
            "summary": {
                "degrees": self.contacts.degrees,
                "mean_household_size": self.contacts.mean_household_size,
                "mean_weighted_contacts": self.contacts.mean_weighted_contacts(),
                "r0_approximation": approximate_r0(self.contacts, self.truth),
                "group_populations": {
                    group.name: group.population for group in self.contacts.groups
                },
            },
        }

    def write_outputs(self, directory: Path) -> None:
        """Écrit les observations, la trajectoire cachée et .truth.json."""
        directory.mkdir(parents=True, exist_ok=True)
        self.observations().to_csv(directory / "observations.csv", index=False)
        self.truth_frame().to_csv(directory / "truth_daily.csv", index=False)
        (directory / ".truth.json").write_text(
            json.dumps(self.secret_payload(), indent=2),
            encoding="utf-8",
        )
