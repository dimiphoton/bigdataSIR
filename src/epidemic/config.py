"""Chargement des fichiers YAML de configs/."""

from datetime import date
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Refuse les clés YAML inconnues."""

    model_config = ConfigDict(extra="forbid")


class FractionRange(StrictModel):
    """Borne basse et borne haute d'une proportion."""

    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError("low doit être inférieur ou égal à high.")
        return self


class DayRange(StrictModel):
    """Borne basse et borne haute d'une durée, en jours."""

    low: int = Field(ge=0)
    high: int = Field(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError("low doit être inférieur ou égal à high.")
        return self


class Interval(StrictModel):
    """Borne basse et borne haute d'un nombre réel."""

    low: float
    high: float

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError("low doit être inférieur ou égal à high.")
        return self


class IntBounds(StrictModel):
    """Borne basse et borne haute d'un entier, bornes incluses."""

    low: int
    high: int

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError("low doit être inférieur ou égal à high.")
        return self


class AgeBin(StrictModel):
    """Tranche d'âge, bornes incluses."""

    name: str
    low: int = Field(ge=0)
    high: int = Field(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"Tranche {self.name} : low > high.")
        return self


class AgeWindow(StrictModel):
    """Âges inclus dans un lieu (école ou travail)."""

    low: int = Field(ge=0)
    high: int = Field(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError("low doit être inférieur ou égal à high.")
        return self


class ContactCaps(StrictModel):
    """Nombre maximal de contacts effectifs par jour, hors domicile."""

    school: float = Field(gt=0)
    work: float = Field(gt=0)
    community: float = Field(gt=0)


class LocationWeights(StrictModel):
    """Poids de transmission d'un contact, par lieu."""

    home: float = Field(ge=0)
    school: float = Field(ge=0)
    work: float = Field(ge=0)
    community: float = Field(ge=0)


class SourceSwitches(StrictModel):
    """Active une source d'aléa. null : la source suit le niveau."""

    demographic: bool | None = None
    environmental: bool | None = None
    individual_contagion: bool | None = None
    individual_compliance: bool | None = None
    individual_durations: bool | None = None


class SeverityRates(StrictModel):
    """Probabilités de gravité d'une tranche d'âge, parmi les symptomatiques."""

    p_hosp: float = Field(ge=0, le=1)
    p_icu: float = Field(ge=0, le=1)
    p_death_ward: float = Field(ge=0, le=1)
    p_death_icu: float = Field(ge=0, le=1)


class SimulatorConfig(StrictModel):
    """Réglages du simulateur."""

    population: int = Field(gt=0)
    seed: int = Field(ge=0)
    stochasticity_level: int = Field(ge=0, le=3)
    day1_date: date
    observation_source: Literal["simulator", "course"] = "simulator"
    stats_dir: str = "data/external/stats"
    course_data: str = "data/external/data.csv"
    output_dir: str = "data/generated"
    sources: SourceSwitches = Field(default_factory=SourceSwitches)
    age_bins: list[AgeBin] = Field(default_factory=list)
    school_ages: AgeWindow = Field(default_factory=lambda: AgeWindow(low=3, high=17))
    work_ages: AgeWindow = Field(default_factory=lambda: AgeWindow(low=18, high=64))
    school_participation: float = Field(default=1.0, ge=0, le=1)
    work_participation: float = Field(default=1.0, ge=0, le=1)
    contact_caps: ContactCaps = Field(
        default_factory=lambda: ContactCaps(school=15, work=15, community=4)
    )
    location_weights: LocationWeights = Field(
        default_factory=lambda: LocationWeights(
            home=1, school=0.45, work=0.45, community=0.25
        )
    )
    beta_per_contact: Interval = Field(
        default_factory=lambda: Interval(low=0.02, high=0.045)
    )
    initial_exposed: IntBounds = Field(
        default_factory=lambda: IntBounds(low=25, high=70)
    )
    presymptomatic_days: Interval = Field(
        default_factory=lambda: Interval(low=1, high=3)
    )
    hospital_days: Interval = Field(default_factory=lambda: Interval(low=6, high=12))
    icu_days: Interval = Field(default_factory=lambda: Interval(low=7, high=14))
    asymptomatic_infectiousness: Interval = Field(
        default_factory=lambda: Interval(low=0.4, high=0.8)
    )
    p_test: Interval = Field(default_factory=lambda: Interval(low=0.5, high=0.85))
    flu_background_daily: Interval = Field(
        default_factory=lambda: Interval(low=100, high=500)
    )
    heterogeneity_k: Interval = Field(
        default_factory=lambda: Interval(low=0.1, high=0.5)
    )
    max_daily_tests: int = Field(default=20_000, gt=0)
    duration_shape: int = Field(default=3, ge=1, le=8)
    ou_theta: float = Field(default=0.1, ge=0)
    ou_sigma: float = Field(default=0.05, ge=0)
    seasonality_amplitude: float = Field(default=0.0, ge=0, lt=1)
    p_test_drift_sigma: float = Field(default=0.02, ge=0)
    flu_drift_sigma: float = Field(default=0.05, ge=0)
    sensitivity_daily_sd: float = Field(default=0.02, ge=0)
    compliance_daily_sd: float = Field(default=0.05, ge=0)
    extended_screening_rate: float = Field(default=0.0, ge=0, le=1)
    isolate_on_symptoms: bool = False
    severity: dict[str, SeverityRates] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_bins_and_severity(self) -> Self:
        """Les tranches ne se chevauchent pas, et chaque tranche a une gravité."""
        previous_high = -1
        seen: set[str] = set()
        for age_bin in self.age_bins:
            if age_bin.name in seen:
                raise ValueError(f"Tranche d'âge en double : {age_bin.name}.")
            seen.add(age_bin.name)
            if age_bin.low <= previous_high:
                raise ValueError(
                    "Les tranches d'âge doivent être croissantes et disjointes."
                )
            previous_high = age_bin.high
        missing = seen.symmetric_difference(self.severity)
        if self.age_bins and missing:
            raise ValueError(
                f"Gravité et tranches d'âge ne correspondent pas : {sorted(missing)}."
            )
        return self


class PriorsConfig(StrictModel):
    """Bornes connues des consignes. Les familles de lois sont dans le modèle."""

    population: int = Field(gt=0)
    symptomatic_fraction: FractionRange
    test_sensitivity: FractionRange
    test_specificity: float = Field(ge=0, le=1)
    min_fraction_symptomatic_tested: float = Field(ge=0, le=1)
    incubation_days: DayRange
    symptomatic_before_hospital_days: DayRange


class ContactReduction(StrictModel):
    """Part de contacts retirée dans chaque lieu. 0.25 signifie −25 %."""

    home: float = Field(ge=0, le=1)
    school: float = Field(ge=0, le=1)
    work: float = Field(ge=0, le=1)
    community: float = Field(ge=0, le=1)


class InterventionConfig(StrictModel):
    """Une intervention publique des consignes."""

    parameter: Literal["duration_days", "age_threshold"] | None
    infection_rate_reduction: float | None = Field(default=None, ge=0, le=1)
    contact_reduction: ContactReduction | None = None


class ScenariosConfig(StrictModel):
    """Interventions, respect des mesures et règles de combinaison."""

    compliance: float = Field(ge=0, le=1)
    combination_rule: Literal["strongest"]
    mask_multiplies: bool
    incompatible_pairs: list[tuple[str, str]]
    interventions: dict[str, InterventionConfig]

    @model_validator(mode="after")
    def check_pairs(self) -> Self:
        """Les paires incompatibles doivent nommer des interventions connues."""
        known = set(self.interventions)
        for left, right in self.incompatible_pairs:
            if left not in known or right not in known:
                raise ValueError(f"Paire inconnue : {left}, {right}.")
        return self


class ThresholdsConfig(StrictModel):
    """Plafonds d'hôpitaux et date de fin de l'objectif."""

    max_hospitalized: int = Field(gt=0)
    max_critical: int = Field(gt=0)
    horizon_end: date


class TierConfig(StrictModel):
    """Un palier de politiques et les interventions qu'il autorise."""

    level: int = Field(ge=0)
    from_day: int = Field(ge=1)
    interventions: list[str]


class EarlyHospitalUnlock(StrictModel):
    """Déblocage anticipé si la probabilité d'hôpitaux élevés est forte."""

    earliest_day: int = Field(ge=1)
    hospital_threshold: int = Field(gt=0)
    horizon_days: int = Field(gt=0)
    probability: float = Field(ge=0, le=1)


class EarlySaturationUnlock(StrictModel):
    """Déblocage anticipé si la probabilité de saturation est forte."""

    earliest_day: int = Field(ge=1)
    horizon_days: int = Field(gt=0)
    probability: float = Field(ge=0, le=1)


class RevealConfig(StrictModel):
    """Texte rendu public à partir d'un jour de jeu."""

    day: int = Field(ge=1)
    title: str
    body: str


class GameConfig(StrictModel):
    """Calendrier, paliers, révélations et coûts du jeu."""

    bulletin_period_days: int = Field(ge=1)
    end_day: int = Field(ge=1)
    implementation_delay_days: int = Field(ge=0)
    final_decision_day: int = Field(ge=1)
    final_implementation_day: int = Field(ge=1)
    min_policy_days: int = Field(ge=1)
    stats_public_day: int = Field(ge=1)
    tiers: list[TierConfig]
    early_tier2: EarlyHospitalUnlock
    early_tier3: EarlySaturationUnlock
    constraint_costs: dict[str, float]
    reveals: list[RevealConfig]

    @model_validator(mode="after")
    def check_calendar(self) -> Self:
        """Le jour de décision finale tombe sur un bulletin, avant sa mise en œuvre."""
        if (
            self.final_decision_day <= self.end_day
            and self.final_decision_day % self.bulletin_period_days != 0
        ):
            raise ValueError(
                "Le jour de décision finale doit être un jour de bulletin."
            )
        if self.final_implementation_day < self.final_decision_day:
            raise ValueError(
                "La mise en œuvre finale ne peut pas précéder la décision."
            )
        levels = [tier.level for tier in self.tiers]
        if levels != sorted(levels) or len(levels) != len(set(levels)):
            raise ValueError(
                "Les paliers doivent avoir des niveaux croissants et uniques."
            )
        return self


class NutsConfig(StrictModel):
    """Réglages de NUTS. Quatre chaînes, comme demandé pour l'inférence."""

    num_chains: int = Field(default=4, ge=1)
    num_warmup: int = Field(default=500, ge=1)
    num_samples: int = Field(default=500, ge=1)
    target_accept: float = Field(default=0.85, gt=0, lt=1)
    chain_method: Literal["parallel", "sequential", "vectorized"] = "vectorized"


class SviConfig(StrictModel):
    """Réglages de l'inférence variationnelle, pour les essais rapides."""

    num_steps: int = Field(default=2_000, ge=1)
    learning_rate: float = Field(default=0.005, gt=0)
    num_samples: int = Field(default=400, ge=1)


class DiagnosticsConfig(StrictModel):
    """Seuils. L'ajustement NUTS échoue si un seuil est dépassé."""

    max_rhat: float = Field(default=1.01, gt=1)
    min_ess: float = Field(default=400, gt=0)
    max_divergences: int = Field(default=0, ge=0)
    strict: bool = True


class LikelihoodConfig(StrictModel):
    """Lois d'observation. Une seule loi par série, pour ne pas compter deux fois."""

    cases: Literal["negative_binomial", "beta_binomial"] = "negative_binomial"
    hospital: Literal["negative_binomial", "lognormal"] = "negative_binomial"


class IdentifiabilityConfig(StrictModel):
    """Rapport largeur a posteriori / largeur a priori."""

    well_below: float = Field(default=0.5, gt=0, lt=1)
    partial_below: float = Field(default=0.85, gt=0, lt=1)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.well_below >= self.partial_below:
            raise ValueError("well_below doit être inférieur à partial_below.")
        return self


class InferenceConfig(StrictModel):
    """Réglages du modèle d'inférence. Il ne voit pas les paramètres secrets."""

    seed: int = Field(default=0, ge=0)
    age_groups: Literal[2, 3, 4] = 2
    variant: Literal["A", "B", "C"] = "A"
    beta_mode: Literal["weekly_rw", "piecewise"] = "weekly_rw"
    beta_breakpoints: list[int] = Field(default_factory=list)
    estimate_specificity: bool = False
    method: Literal["nuts", "svi"] = "nuts"
    nuts: NutsConfig = Field(default_factory=NutsConfig)
    svi: SviConfig = Field(default_factory=SviConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    fixed_concentration: float = Field(default=100.0, gt=0)
    output_dir: str = "data/generated/inference"
    course_data: str = "data/external/data.csv"
    likelihood: LikelihoodConfig = Field(default_factory=LikelihoodConfig)
    horizons: list[int] = Field(default_factory=lambda: [7, 14, 21])
    learning_curve_days: list[int] = Field(
        default_factory=lambda: [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    )
    identifiability: IdentifiabilityConfig = Field(
        default_factory=IdentifiabilityConfig
    )
    default_days: int = Field(default=40, ge=7)
    save_plots: bool = True

    @model_validator(mode="after")
    def check_breakpoints(self) -> Self:
        """Les dates de changement de β sont croissantes, à partir du jour 2."""
        if self.beta_breakpoints != sorted(set(self.beta_breakpoints)):
            raise ValueError("beta_breakpoints doit être strictement croissant.")
        if any(day < 2 for day in self.beta_breakpoints):
            raise ValueError(
                "Un changement de β ne peut pas commencer avant le jour 2."
            )
        if any(day < 1 for day in self.horizons):
            raise ValueError("Les horizons de prévision doivent être positifs.")
        if any(day < 2 for day in self.learning_curve_days):
            raise ValueError("La courbe d'apprentissage commence au jour 2 ou après.")
        return self


class IsolationReach(StrictModel):
    """Part des infectieux qu'un isolement retire, au point de référence."""

    symptomatic_fraction: float = Field(gt=0, lt=1)
    confirmation_probability: float = Field(gt=0, le=1)
    infectious_days: float = Field(gt=0)


class ReachSpread(StrictModel):
    """Écart-type du tirage autour du point de référence. 0 : valeur fixe."""

    symptomatic_fraction: float = Field(ge=0)
    confirmation_probability: float = Field(ge=0)
    infectious_days: float = Field(ge=0)


class PolicyGrid(StrictModel):
    """Grille des règles. Le seuil bas reste sous le seuil haut."""

    indicators: list[Literal["hospital", "icu", "cases_7d", "positivity"]]
    default_indicators: list[Literal["hospital", "icu", "cases_7d", "positivity"]]
    hospital_high: list[float]
    hospital_low: list[float]
    icu_high: list[float]
    icu_low: list[float]
    cases_high: list[float]
    cases_low: list[float]
    positivity_high: list[float]
    positivity_low: list[float]
    hold_days: list[int]

    @model_validator(mode="after")
    def check_grid(self) -> Self:
        if not self.indicators or not self.default_indicators:
            raise ValueError("La grille des règles doit citer au moins un indicateur.")
        unknown = set(self.default_indicators) - set(self.indicators)
        if unknown:
            raise ValueError(f"Indicateur hors grille : {sorted(unknown)}.")
        if any(day < 1 for day in self.hold_days):
            raise ValueError("L'hystérésis dure au moins un jour.")
        return self


class DecisionConfig(StrictModel):
    """Seuil de risque, grilles et incertitude utilisés pour la recommandation."""

    alpha: float = Field(gt=0, le=1)
    alphas: list[float]
    isolation_durations: list[int]
    age_thresholds: list[int]
    forecast_draws: int = Field(ge=1)
    future_seed: int = Field(ge=0)
    share_samples: int = Field(ge=1)
    share_concentration: float = Field(gt=0)
    interval_masses: list[float]
    isolation_reach: IsolationReach
    reach_spread: ReachSpread
    compliances: list[float]
    policy_grid: PolicyGrid

    @model_validator(mode="after")
    def check_decision(self) -> Self:
        if any(not 0 < value <= 1 for value in self.alphas):
            raise ValueError("Chaque alpha doit être entre 0 et 1.")
        if self.isolation_durations != sorted(set(self.isolation_durations)):
            raise ValueError(
                "Les durées d'isolement doivent être strictement croissantes."
            )
        if any(day < 1 for day in self.isolation_durations):
            raise ValueError("Une durée d'isolement vaut au moins un jour.")
        if self.age_thresholds != sorted(set(self.age_thresholds)):
            raise ValueError("Les âges seuils doivent être strictement croissants.")
        if any(not 0 < mass < 1 for mass in self.interval_masses):
            raise ValueError("Une masse d'intervalle est strictement entre 0 et 1.")
        if any(not 0 <= value <= 1 for value in self.compliances):
            raise ValueError("Un taux de respect est entre 0 et 1.")
        return self


class VerbalBand(StrictModel):
    """Un palier de l'échelle verbale. minimum est inclus."""

    minimum: float = Field(ge=0, le=1)
    label: str = Field(min_length=1)


class ReportingConfig(StrictModel):
    """Chemins, horizons et échelle verbale du rendu."""

    seed: int = Field(ge=0)
    docs_dir: str
    figures_dir: str
    bulletin_dir: str
    report_dir: str
    score_horizons: list[int]
    interval_masses: list[float]
    reliability_bins: int = Field(ge=2, le=20)
    ensemble_size: int = Field(ge=20)
    residual_window: int = Field(ge=2)
    min_history_days: int = Field(ge=2)
    verbal_scale: list[VerbalBand]
    bulletin_horizons: list[int]
    prior_draws: int = Field(ge=4, le=64)
    prior_days: int = Field(ge=14, le=120)
    figure_dpi: int = Field(ge=72, le=200)
    series: list[str]
    series_labels: dict[str, str]
    coverage_tolerance: float = Field(gt=0, lt=0.5)
    score_levels: list[int]
    score_level_days: int = Field(ge=21, le=80)
    trend_ratio: float = Field(gt=1, lt=2)
    event_horizon: int = Field(ge=1)

    @model_validator(mode="after")
    def check_reporting(self) -> Self:
        """Les listes du rendu restent ordonnées et complètes."""
        if self.score_horizons != sorted(set(self.score_horizons)):
            raise ValueError(
                "Les horizons de score doivent être strictement croissants."
            )
        if any(day < 1 for day in self.score_horizons):
            raise ValueError("Un horizon de score vaut au moins un jour.")
        if self.interval_masses != sorted(set(self.interval_masses)):
            raise ValueError(
                "Les masses d'intervalle doivent être strictement croissantes."
            )
        if len(self.interval_masses) != 2 or any(
            not 0 < mass < 1 for mass in self.interval_masses
        ):
            raise ValueError("Il faut deux masses d'intervalle, entre 0 et 1.")
        if self.bulletin_horizons != sorted(set(self.bulletin_horizons)):
            raise ValueError(
                "Les horizons du bulletin doivent être strictement croissants."
            )
        minima = [band.minimum for band in self.verbal_scale]
        if minima != sorted(minima, reverse=True) or minima[-1] != 0:
            raise ValueError(
                "L'échelle verbale va du plus haut palier au plus bas, et couvre 0."
            )
        if set(self.series_labels) != set(self.series):
            raise ValueError("Chaque série scorée a un libellé, et rien de plus.")
        if self.score_levels != sorted(set(self.score_levels)):
            raise ValueError(
                "Les niveaux de stochasticité scorés sont strictement croissants."
            )
        if any(level < 0 or level > 3 for level in self.score_levels):
            raise ValueError("Un niveau de stochasticité est entre 0 et 3.")
        if self.event_horizon not in self.score_horizons:
            raise ValueError(
                "L'horizon des événements de saturation est un horizon de score."
            )
        return self


def project_root() -> Path:
    """Remonte les dossiers jusqu'au dépôt, repéré par pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("pyproject.toml introuvable au-dessus du package.")


def load_yaml(path: Path) -> dict[str, Any]:
    """Lit un objet YAML. Échoue si le fichier n'est pas un dictionnaire."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} doit contenir un objet YAML.")
    return cast(dict[str, Any], raw)


def _config_path(name: str) -> Path:
    return project_root() / "configs" / name


def load_simulator_config() -> SimulatorConfig:
    """Charge configs/simulator.yaml."""
    return SimulatorConfig.model_validate(load_yaml(_config_path("simulator.yaml")))


def load_priors_config() -> PriorsConfig:
    """Charge configs/priors.yaml."""
    return PriorsConfig.model_validate(load_yaml(_config_path("priors.yaml")))


def load_scenarios_config() -> ScenariosConfig:
    """Charge configs/scenarios.yaml."""
    return ScenariosConfig.model_validate(load_yaml(_config_path("scenarios.yaml")))


def load_thresholds_config() -> ThresholdsConfig:
    """Charge configs/thresholds.yaml."""
    return ThresholdsConfig.model_validate(load_yaml(_config_path("thresholds.yaml")))


def load_game_config() -> GameConfig:
    """Charge configs/game.yaml."""
    return GameConfig.model_validate(load_yaml(_config_path("game.yaml")))


def load_inference_config() -> InferenceConfig:
    """Charge configs/inference.yaml."""
    return InferenceConfig.model_validate(load_yaml(_config_path("inference.yaml")))


def load_decision_config() -> DecisionConfig:
    """Charge configs/decision.yaml."""
    return DecisionConfig.model_validate(load_yaml(_config_path("decision.yaml")))


def load_reporting_config() -> ReportingConfig:
    """Charge configs/reporting.yaml."""
    return ReportingConfig.model_validate(load_yaml(_config_path("reporting.yaml")))


def resolve_path(relative: str) -> Path:
    """Résout un chemin du YAML par rapport à la racine du dépôt."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return project_root() / path
