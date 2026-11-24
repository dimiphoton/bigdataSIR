"""Tirage des paramètres secrets, à partir des plages des consignes et du YAML."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from epidemic.config import PriorsConfig, SimulatorConfig
from epidemic.simulator.contacts import ContactMatrices


@dataclass
class SourceFlags:
    """Sources d'aléa réellement actives pour une partie."""

    demographic: bool
    environmental: bool
    individual_contagion: bool
    individual_compliance: bool
    individual_durations: bool

    def to_dict(self) -> dict[str, bool]:
        """Sérialise les interrupteurs."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceFlags":
        """Recharge les interrupteurs."""
        return cls(
            demographic=bool(raw["demographic"]),
            environmental=bool(raw["environmental"]),
            individual_contagion=bool(raw["individual_contagion"]),
            individual_compliance=bool(raw["individual_compliance"]),
            individual_durations=bool(raw["individual_durations"]),
        )


def resolve_sources(config: SimulatorConfig) -> SourceFlags:
    """Le niveau ajoute les sources. Un true ou false du YAML l'emporte."""
    level = config.stochasticity_level
    defaults = {
        "demographic": level >= 1,
        "environmental": level >= 2,
        "individual_contagion": level >= 3,
        "individual_compliance": level >= 3,
        "individual_durations": level >= 3,
    }
    chosen: dict[str, bool] = {}
    for name, automatic in defaults.items():
        forced = getattr(config.sources, name)
        chosen[name] = automatic if forced is None else bool(forced)
    return SourceFlags(**chosen)


@dataclass
class TruthParameters:
    """Paramètres cachés d'une ville. Le modèle d'inférence ne doit pas les lire."""

    seed: int
    sources: SourceFlags
    symptomatic_fraction: float
    sensitivity: float
    specificity: float
    p_test: float
    p_test_low: float
    p_test_high: float
    incubation_days: float
    presymptomatic_days: float
    symptomatic_days: float
    asymptomatic_days: float
    hospital_days: float
    icu_days: float
    beta_per_contact: float
    initial_exposed: int
    flu_daily_mean: float
    asymptomatic_infectiousness: float
    heterogeneity_k: float
    p_hosp: tuple[float, ...]
    p_icu: tuple[float, ...]
    p_death_ward: tuple[float, ...]
    p_death_icu: tuple[float, ...]
    duration_shape: int
    extended_screening_rate: float
    isolate_on_symptoms: bool
    max_daily_tests: int
    ou_theta: float
    ou_sigma: float
    seasonality_amplitude: float
    p_test_drift_sigma: float
    flu_drift_sigma: float
    sensitivity_daily_sd: float
    compliance_daily_sd: float
    group_names: tuple[str, ...]

    @property
    def stage_count(self) -> int:
        """Nombre de stades. 1 tant que les durées gamma sont éteintes."""
        if self.sources.individual_durations:
            return self.duration_shape
        return 1

    def to_dict(self) -> dict[str, Any]:
        """Sérialise les paramètres secrets."""
        raw = asdict(self)
        raw["sources"] = self.sources.to_dict()
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TruthParameters":
        """Recharge des paramètres secrets."""
        payload = dict(raw)
        payload["sources"] = SourceFlags.from_dict(payload["sources"])
        for key in ("p_hosp", "p_icu", "p_death_ward", "p_death_icu", "group_names"):
            payload[key] = tuple(payload[key])
        return cls(**payload)


def draw_truth(
    config: SimulatorConfig,
    priors: PriorsConfig,
    rng: np.random.Generator,
) -> TruthParameters:
    """Tire les paramètres dans un ordre fixe, pour une graine stable."""
    if config.p_test.low < priors.min_fraction_symptomatic_tested:
        raise ValueError("p_test.low est sous le minimum des consignes.")
    symptomatic_fraction = float(
        rng.uniform(priors.symptomatic_fraction.low, priors.symptomatic_fraction.high)
    )
    sensitivity = float(
        rng.uniform(priors.test_sensitivity.low, priors.test_sensitivity.high)
    )
    p_test = float(rng.uniform(config.p_test.low, config.p_test.high))
    incubation_days = float(
        rng.uniform(priors.incubation_days.low, priors.incubation_days.high)
    )
    symptomatic_days = float(
        rng.uniform(
            priors.symptomatic_before_hospital_days.low,
            priors.symptomatic_before_hospital_days.high,
        )
    )
    # Hypothèse : les asymptomatiques restent infectieux sur la même plage de durées.
    asymptomatic_days = float(
        rng.uniform(
            priors.symptomatic_before_hospital_days.low,
            priors.symptomatic_before_hospital_days.high,
        )
    )
    presymptomatic_days = float(
        rng.uniform(config.presymptomatic_days.low, config.presymptomatic_days.high)
    )
    hospital_days = float(
        rng.uniform(config.hospital_days.low, config.hospital_days.high)
    )
    icu_days = float(rng.uniform(config.icu_days.low, config.icu_days.high))
    beta = float(rng.uniform(config.beta_per_contact.low, config.beta_per_contact.high))
    initial_exposed = int(
        rng.integers(config.initial_exposed.low, config.initial_exposed.high + 1)
    )
    flu_daily_mean = float(
        rng.uniform(config.flu_background_daily.low, config.flu_background_daily.high)
    )
    asymptomatic_infectiousness = float(
        rng.uniform(
            config.asymptomatic_infectiousness.low,
            config.asymptomatic_infectiousness.high,
        )
    )
    heterogeneity_k = float(
        rng.uniform(config.heterogeneity_k.low, config.heterogeneity_k.high)
    )
    names = tuple(age_bin.name for age_bin in config.age_bins)
    return TruthParameters(
        seed=config.seed,
        sources=resolve_sources(config),
        symptomatic_fraction=symptomatic_fraction,
        sensitivity=sensitivity,
        specificity=priors.test_specificity,
        p_test=p_test,
        p_test_low=config.p_test.low,
        p_test_high=config.p_test.high,
        incubation_days=incubation_days,
        presymptomatic_days=presymptomatic_days,
        symptomatic_days=symptomatic_days,
        asymptomatic_days=asymptomatic_days,
        hospital_days=hospital_days,
        icu_days=icu_days,
        beta_per_contact=beta,
        initial_exposed=initial_exposed,
        flu_daily_mean=flu_daily_mean,
        asymptomatic_infectiousness=asymptomatic_infectiousness,
        heterogeneity_k=heterogeneity_k,
        p_hosp=tuple(config.severity[name].p_hosp for name in names),
        p_icu=tuple(config.severity[name].p_icu for name in names),
        p_death_ward=tuple(config.severity[name].p_death_ward for name in names),
        p_death_icu=tuple(config.severity[name].p_death_icu for name in names),
        duration_shape=config.duration_shape,
        extended_screening_rate=config.extended_screening_rate,
        isolate_on_symptoms=config.isolate_on_symptoms,
        max_daily_tests=config.max_daily_tests,
        ou_theta=config.ou_theta,
        ou_sigma=config.ou_sigma,
        seasonality_amplitude=config.seasonality_amplitude,
        p_test_drift_sigma=config.p_test_drift_sigma,
        flu_drift_sigma=config.flu_drift_sigma,
        sensitivity_daily_sd=config.sensitivity_daily_sd,
        compliance_daily_sd=config.compliance_daily_sd,
        group_names=names,
    )


def approximate_r0(contacts: ContactMatrices, truth: TruthParameters) -> float:
    """Ordre de grandeur de R0, sans structure d'âge ni intervention."""
    infectious_days = (
        truth.symptomatic_fraction
        * (truth.presymptomatic_days + truth.symptomatic_days)
        + (1.0 - truth.symptomatic_fraction)
        * truth.asymptomatic_days
        * truth.asymptomatic_infectiousness
    )
    return truth.beta_per_contact * contacts.mean_weighted_contacts() * infectious_days
