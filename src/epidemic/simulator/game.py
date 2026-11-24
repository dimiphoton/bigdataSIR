"""Boucle de jeu : bulletins, décisions, score."""

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from epidemic.config import (
    GameConfig,
    ScenariosConfig,
    ThresholdsConfig,
    load_game_config,
    load_scenarios_config,
    load_thresholds_config,
    resolve_path,
)
from epidemic.simulator.dynamics import PUBLISHED_COLUMNS
from epidemic.simulator.engine import RealitySimulator
from epidemic.simulator.interventions import (
    LABELS,
    AppliedPolicy,
    assert_compatible,
    assert_parameters,
)
from epidemic.simulator.randomness import dump_rng, load_rng


@dataclass(frozen=True)
class Decision:
    """Choix du joueur à un bulletin."""

    name: str
    duration_days: int | None = None
    age_threshold: int | None = None


@dataclass(frozen=True)
class ScheduledPolicy:
    """Mesure programmée, avec son délai et sa durée minimale."""

    name: str
    effect_start: int
    effect_end: int | None
    min_end: int
    duration_days: int | None
    age_threshold: int | None
    decision_day: int

    def active_on(self, day: int) -> bool:
        """Vrai si la mesure s'applique ce jour-là."""
        if day < self.effect_start:
            return False
        if self.effect_end is None:
            return True
        return day <= self.effect_end

    def same_parameters(self, decision: Decision) -> bool:
        """Vrai si la décision ne change ni la durée ni l'âge seuil."""
        return (
            self.duration_days == decision.duration_days
            and self.age_threshold == decision.age_threshold
        )


@dataclass
class Bulletin:
    """Ce que le joueur voit à un point d'arrêt hebdomadaire."""

    day: int
    observations: pd.DataFrame
    new_observations: pd.DataFrame
    reveals: list[Any]
    tier: int
    allowed: list[str]


@dataclass
class GameScore:
    """Trois comptes qui permettent de comparer des stratégies."""

    days_hospital_saturated: int
    days_critical_saturated: int
    days_saturated: int
    deaths: int
    constraint_cost: float


def implementation_day(decision_day: int, config: GameConfig) -> int:
    """Jour où la décision agit. Le bulletin final est fixé au jour prévu."""
    if decision_day == config.final_decision_day:
        return config.final_implementation_day
    return decision_day + config.implementation_delay_days


def bulletin_days(config: GameConfig) -> list[int]:
    """Jours de bulletin, du premier multiple jusqu'à la fin de partie."""
    period = config.bulletin_period_days
    return list(range(period, config.end_day + 1, period))


def apply_decision(
    policies: list[ScheduledPolicy],
    decisions: list[Decision],
    *,
    decision_day: int,
    effect_day: int,
    min_policy_days: int,
) -> list[ScheduledPolicy]:
    """Remplace le calendrier. Lève une erreur si une mesure est retirée trop tôt."""
    new_by_name = {item.name: item for item in decisions}
    if len(new_by_name) != len(decisions):
        raise ValueError("La même intervention est demandée deux fois.")
    kept: list[ScheduledPolicy] = []
    still_open: set[str] = set()
    for policy in policies:
        open_now = policy.effect_end is None or policy.effect_end >= decision_day
        if not open_now:
            kept.append(policy)
            continue
        decision = new_by_name.get(policy.name)
        if decision is not None and policy.same_parameters(decision):
            kept.append(policy)
            still_open.add(policy.name)
            continue
        if policy.effect_start > decision_day:
            continue
        if decision_day < policy.min_end:
            label = LABELS.get(policy.name, policy.name)
            raise ValueError(
                f"{label} doit rester en place jusqu'au jour {policy.min_end} inclus."
            )
        kept.append(replace_end(policy, effect_day - 1))
    for decision in decisions:
        if decision.name in still_open:
            continue
        kept.append(
            ScheduledPolicy(
                name=decision.name,
                effect_start=effect_day,
                effect_end=None,
                min_end=effect_day + min_policy_days - 1,
                duration_days=decision.duration_days,
                age_threshold=decision.age_threshold,
                decision_day=decision_day,
            )
        )
    return kept


def replace_end(policy: ScheduledPolicy, effect_end: int) -> ScheduledPolicy:
    """Copie une mesure avec une nouvelle date de fin."""
    return ScheduledPolicy(
        name=policy.name,
        effect_start=policy.effect_start,
        effect_end=effect_end,
        min_end=policy.min_end,
        duration_days=policy.duration_days,
        age_threshold=policy.age_threshold,
        decision_day=policy.decision_day,
    )


class Game:
    """Moteur sans interface. Le modèle bayésien ne lit pas cet état."""

    def __init__(
        self,
        simulator: RealitySimulator,
        game_config: GameConfig,
        scenarios: ScenariosConfig,
        thresholds: ThresholdsConfig,
    ) -> None:
        self.simulator = simulator
        self.game_config = game_config
        self.scenarios = scenarios
        self.thresholds = thresholds
        self.policies: list[ScheduledPolicy] = []
        self._snapshots: dict[int, list[ScheduledPolicy]] = {}
        self.decision_log: dict[int, list[Decision]] = {}
        self.p_hospital: float | None = None
        self.p_saturation: float | None = None

    @classmethod
    def start(
        cls,
        *,
        seed: int | None = None,
        level: int | None = None,
        observation_source: str | None = None,
        end_day: int | None = None,
        output_dir: Path | None = None,
        write_secret: bool = False,
    ) -> "Game":
        """Ouvre une partie. Le fichier secret n'est écrit que sur demande."""
        simulator = RealitySimulator.from_configs(
            seed=seed,
            level=level,
            observation_source=observation_source,
        )
        game_config = load_game_config()
        if end_day is not None:
            game_config = game_config.model_copy(update={"end_day": end_day})
        game = cls(
            simulator,
            game_config,
            load_scenarios_config(),
            load_thresholds_config(),
        )
        if write_secret:
            directory = output_dir or resolve_path(simulator.config.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".truth.json").write_text(
                json.dumps(simulator.secret_payload(), indent=2),
                encoding="utf-8",
            )
        return game

    @property
    def day(self) -> int:
        """Jour courant. 0 avant la première publication."""
        return self.simulator.day

    @property
    def end_day(self) -> int:
        """Dernier jour de la partie."""
        return self.game_config.end_day

    def bulletin_day_list(self) -> list[int]:
        """Jours où une décision est possible."""
        return bulletin_days(self.game_config)

    def tier_level(self) -> int:
        """Palier débloqué, par la date ou par une alerte fournie de l'extérieur."""
        level = 0
        for tier in self.game_config.tiers:
            if self.day >= tier.from_day:
                level = max(level, tier.level)
        early_hospital = self.game_config.early_tier2
        if (
            self.p_hospital is not None
            and self.day >= early_hospital.earliest_day
            and self.p_hospital > early_hospital.probability
        ):
            level = max(level, 2)
        early_saturation = self.game_config.early_tier3
        if (
            self.p_saturation is not None
            and self.day >= early_saturation.earliest_day
            and self.p_saturation > early_saturation.probability
        ):
            level = max(level, 3)
        return level

    def allowed_interventions(self) -> list[str]:
        """Noms autorisés au palier courant."""
        level = self.tier_level()
        for tier in self.game_config.tiers:
            if tier.level == level:
                return list(tier.interventions)
        return []

    def revealed(self) -> list[Any]:
        """Informations publiques dont le jour est atteint."""
        return [item for item in self.game_config.reveals if item.day <= self.day]

    def submit_risk(self, p_hospital: float | None, p_saturation: float | None) -> None:
        """Reçoit des probabilités du futur modèle, pour débloquer un palier."""
        for value in (p_hospital, p_saturation):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("Une probabilité doit être entre 0 et 1.")
        self.p_hospital = p_hospital
        self.p_saturation = p_saturation

    def advance(self, days: int) -> pd.DataFrame:
        """Avance de quelques jours, sans dépasser la fin."""
        if days < 0:
            raise ValueError("Le nombre de jours doit être positif ou nul.")
        rows: list[dict[str, int]] = []
        for _ in range(days):
            if self.simulator.day >= self.end_day:
                break
            policies = self._active(self.simulator.day + 1)
            rows.append(self.simulator.step(policies))
        return pd.DataFrame(rows, columns=list(PUBLISHED_COLUMNS))

    def next_bulletin_day(self) -> int | None:
        """Prochain bulletin strictement après le jour courant."""
        for day in self.bulletin_day_list():
            if day > self.day:
                return day
        return None

    def next_bulletin(self) -> Bulletin:
        """Avance jusqu'au prochain bulletin et renvoie ce que le joueur peut voir."""
        target = self.next_bulletin_day()
        if target is None:
            raise ValueError("Aucun bulletin restant.")
        new_rows = self.advance(target - self.day)
        return Bulletin(
            day=self.day,
            observations=self.observations(),
            new_observations=new_rows,
            reveals=self.revealed(),
            tier=self.tier_level(),
            allowed=self.allowed_interventions(),
        )

    def decide(self, decisions: list[Decision]) -> None:
        """Valide une décision de bulletin et la programme avec son délai."""
        if self.day not in self.bulletin_day_list():
            raise ValueError("Une décision n'est possible que le jour d'un bulletin.")
        allowed = set(self.allowed_interventions())
        for decision in decisions:
            if decision.name not in allowed:
                label = LABELS.get(decision.name, decision.name)
                raise ValueError(
                    f"{label} n'est pas débloqué (palier {self.tier_level()})."
                )
        names = [decision.name for decision in decisions]
        assert_compatible(names, self.scenarios)
        assert_parameters(
            [
                AppliedPolicy(
                    decision.name, decision.duration_days, decision.age_threshold
                )
                for decision in decisions
            ],
            self.scenarios,
        )
        snapshot = self._snapshots.get(self.day)
        if snapshot is None:
            snapshot = copy.deepcopy(self.policies)
        updated = apply_decision(
            snapshot,
            decisions,
            decision_day=self.day,
            effect_day=implementation_day(self.day, self.game_config),
            min_policy_days=self.game_config.min_policy_days,
        )
        self._snapshots[self.day] = snapshot
        self.policies = updated
        self.decision_log[self.day] = list(decisions)

    def active_names(self, day: int | None = None) -> list[str]:
        """Noms des mesures actives un jour donné."""
        target = self.day if day is None else day
        return [policy.name for policy in self.policies if policy.active_on(target)]

    def observations(self) -> pd.DataFrame:
        """Toutes les lignes publiées depuis le jour 1."""
        return self.simulator.observations()

    def score(self) -> GameScore:
        """Jours de saturation, décès et coût cumulé des contraintes."""
        frame = self.observations()
        if frame.empty:
            return GameScore(0, 0, 0, 0, 0.0)
        hospital_limit = self.thresholds.max_hospitalized
        critical_limit = self.thresholds.max_critical
        hospital_days = int((frame["num_hospitalised"] > hospital_limit).sum())
        critical_days = int((frame["num_critical"] > critical_limit).sum())
        saturated = (frame["num_hospitalised"] > hospital_limit) | (
            frame["num_critical"] > critical_limit
        )
        cost = 0.0
        for day in frame["Day"].astype(int):
            for name in self.active_names(int(day)):
                cost += float(self.game_config.constraint_costs[name])
        return GameScore(
            days_hospital_saturated=hospital_days,
            days_critical_saturated=critical_days,
            days_saturated=int(saturated.sum()),
            deaths=int(frame["num_fatalities"].iloc[-1]),
            constraint_cost=cost,
        )

    def _active(self, day: int) -> list[AppliedPolicy]:
        return [
            AppliedPolicy(policy.name, policy.duration_days, policy.age_threshold)
            for policy in self.policies
            if policy.active_on(day)
        ]

    def save(self, path: Path) -> None:
        """Sauvegarde l'état et les générateurs, pour reprendre la partie."""
        payload = {
            "version": 1,
            "seed": self.simulator.config.seed,
            "level": self.simulator.config.stochasticity_level,
            "observation_source": self.simulator.config.observation_source,
            "end_day": self.end_day,
            "day": self.day,
            "truth": self.simulator.truth.to_dict(),
            "state": self.simulator.state.to_dict(),
            "latent": self.simulator.latent.to_dict(),
            "streams": {
                name: dump_rng(rng) for name, rng in self.simulator.streams.items()
            },
            "cumulative_hospitalizations": self.simulator.cumulative_hospitalizations,
            "published_rows": self.simulator.published_rows,
            "truth_rows": self.simulator.truth_rows,
            "policies": [asdict(policy) for policy in self.policies],
            "snapshots": {
                str(day): [asdict(policy) for policy in policies]
                for day, policies in self._snapshots.items()
            },
            "decision_log": {
                str(day): [asdict(decision) for decision in decisions]
                for day, decisions in self.decision_log.items()
            },
            "p_hospital": self.p_hospital,
            "p_saturation": self.p_saturation,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Game":
        """Recharge une partie. Les YAML servent aux règles, le fichier à l'état."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            raise ValueError("Version de sauvegarde inconnue.")
        from epidemic.simulator.parameters import TruthParameters
        from epidemic.simulator.state import EpidemicState, LatentNoise

        truth = TruthParameters.from_dict(raw["truth"])
        game = cls.start(
            seed=int(raw["seed"]),
            level=int(raw["level"]),
            observation_source=str(raw["observation_source"]),
            end_day=int(raw["end_day"]),
            write_secret=False,
        )
        game.simulator.truth = truth
        game.simulator.state = EpidemicState.from_dict(raw["state"])
        game.simulator.latent = LatentNoise.from_dict(raw["latent"])
        game.simulator.streams = {
            name: load_rng(state) for name, state in raw["streams"].items()
        }
        game.simulator.day = int(raw["day"])
        game.simulator.cumulative_hospitalizations = int(
            raw["cumulative_hospitalizations"]
        )
        game.simulator.published_rows = list(raw["published_rows"])
        game.simulator.truth_rows = list(raw["truth_rows"])
        game.policies = [_policy_from_dict(item) for item in raw["policies"]]
        game._snapshots = {
            int(day): [_policy_from_dict(item) for item in policies]
            for day, policies in raw["snapshots"].items()
        }
        game.decision_log = {
            int(day): [_decision_from_dict(item) for item in decisions]
            for day, decisions in raw["decision_log"].items()
        }
        game.p_hospital = raw["p_hospital"]
        game.p_saturation = raw["p_saturation"]
        return game


def _policy_from_dict(raw: dict[str, Any]) -> ScheduledPolicy:
    return ScheduledPolicy(
        name=str(raw["name"]),
        effect_start=int(raw["effect_start"]),
        effect_end=None if raw["effect_end"] is None else int(raw["effect_end"]),
        min_end=int(raw["min_end"]),
        duration_days=None
        if raw["duration_days"] is None
        else int(raw["duration_days"]),
        age_threshold=None
        if raw["age_threshold"] is None
        else int(raw["age_threshold"]),
        decision_day=int(raw["decision_day"]),
    )


def _decision_from_dict(raw: dict[str, Any]) -> Decision:
    return Decision(
        name=str(raw["name"]),
        duration_days=None
        if raw["duration_days"] is None
        else int(raw["duration_days"]),
        age_threshold=None
        if raw["age_threshold"] is None
        else int(raw["age_threshold"]),
    )
