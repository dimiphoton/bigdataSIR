"""Contexte Jinja : uniquement des valeurs lues ou calculées."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from epidemic.config import (
    ReportingConfig,
    load_decision_config,
    load_game_config,
    load_inference_config,
    load_priors_config,
    load_reporting_config,
    load_scenarios_config,
    load_simulator_config,
    load_thresholds_config,
    project_root,
    resolve_path,
)
from epidemic.forecast.scoring import (
    SeriesScore,
    interpret_coverage,
    reliability_table,
    score_observation_series,
    summary_rows,
)
from epidemic.inference.validation import load_course_series
from epidemic.reporting.priors_table import markdown_table, prior_rows
from epidemic.reporting.text import format_auto, format_fr, format_percent, french_date
from epidemic.series import ObservationSeries, observations_from_frame
from epidemic.simulator.engine import RealitySimulator
from epidemic.simulator.interventions import LABELS
from epidemic.viz.model_diagram import compartment_mermaid

_BETA_MODES = {
    "weekly_rw": "marche aléatoire hebdomadaire",
    "piecewise": "constante par morceaux entre des dates connues",
}


def build_context() -> dict[str, object]:
    """Assemble les chiffres du rendu. Aucune valeur n'est écrite à la main."""
    reporting = load_reporting_config()
    inference = load_inference_config()
    game = load_game_config()
    decision = load_decision_config()
    thresholds = load_thresholds_config()
    priors = load_priors_config()
    scenarios = load_scenarios_config()
    simulator = load_simulator_config()
    course = load_course_series()
    course_score = score_labeled(
        course, "données du cours", reporting, game.bulletin_period_days
    )
    level_scores = [
        score_labeled(
            simulate_public(level, reporting),
            f"niveau {level}",
            reporting,
            game.bulletin_period_days,
        )
        for level in reporting.score_levels
    ]
    rows = prior_rows(reporting.seed)
    costs = [
        {"label": LABELS.get(name, name), "cost": format_auto(float(cost))}
        for name, cost in sorted(
            game.constraint_costs.items(), key=lambda item: item[1]
        )
    ]
    low_mass, high_mass = reporting.interval_masses
    context: dict[str, object] = {
        "population": format_fr(simulator.population, 0),
        "day1": french_date(simulator.day1_date),
        "end_day": str(game.end_day),
        "end_date": french_date(thresholds.horizon_end),
        "max_hospitalized": format_fr(thresholds.max_hospitalized, 0),
        "max_critical": format_fr(thresholds.max_critical, 0),
        "alpha": format_percent(decision.alpha, 0),
        "alphas": ", ".join(format_percent(value, 0) for value in decision.alphas),
        "variant": inference.variant,
        "beta_mode": _BETA_MODES.get(inference.beta_mode, inference.beta_mode),
        "method": inference.method,
        "chains": str(inference.nuts.num_chains),
        "warmup": format_fr(inference.nuts.num_warmup, 0),
        "draws": format_fr(inference.nuts.num_samples, 0),
        "max_rhat": format_fr(inference.diagnostics.max_rhat, 2),
        "min_ess": format_fr(inference.diagnostics.min_ess, 0),
        "max_divergences": str(inference.diagnostics.max_divergences),
        "compliance": format_percent(scenarios.compliance, 0),
        "symptomatic_low": format_percent(priors.symptomatic_fraction.low, 0),
        "symptomatic_high": format_percent(priors.symptomatic_fraction.high, 0),
        "sensitivity_low": format_percent(priors.test_sensitivity.low, 0),
        "sensitivity_high": format_percent(priors.test_sensitivity.high, 0),
        "incubation_low": str(priors.incubation_days.low),
        "incubation_high": str(priors.incubation_days.high),
        "symptom_days_low": str(priors.symptomatic_before_hospital_days.low),
        "symptom_days_high": str(priors.symptomatic_before_hospital_days.high),
        "min_tested": format_percent(priors.min_fraction_symptomatic_tested, 0),
        "specificity": format_percent(priors.test_specificity, 0),
        "stochasticity": str(simulator.stochasticity_level),
        "age_groups": str(inference.age_groups),
        "fixed_concentration": format_auto(inference.fixed_concentration),
        "prior_draws": str(reporting.prior_draws),
        "prior_days": str(reporting.prior_days),
        "horizons": ", ".join(str(day) for day in reporting.score_horizons),
        "bulletin_horizons": ", ".join(str(day) for day in reporting.bulletin_horizons),
        "low_mass": format_percent(low_mass, 0),
        "high_mass": format_percent(high_mass, 0),
        "event_horizon": str(reporting.event_horizon),
        "isolation_durations": ", ".join(
            str(day) for day in decision.isolation_durations
        ),
        "age_thresholds": ", ".join(str(age) for age in decision.age_thresholds),
        "delay_days": str(game.implementation_delay_days),
        "min_policy_days": str(game.min_policy_days),
        "mask_reduction": format_percent(
            scenarios.interventions["mask"].infection_rate_reduction or 0.0, 0
        ),
        "seed": str(reporting.seed),
        "prior_table": markdown_table(rows),
        "cost_table": _cost_table(costs),
        "course_table": _summary_table(
            course_score, reporting.series_labels, low_mass, high_mass
        ),
        "level_table": _level_table(level_scores, high_mass),
        "course_sentence": _coverage_sentence(course_score, high_mass, low_mass),
        "course_interpretation": interpret_coverage(
            course_score.coverage(high=True),
            high_mass,
            reporting.coverage_tolerance,
        ),
        "course_brier": format_auto(course_score.brier()),
        "course_event_note": _event_note(course_score),
        "course_crps_persistence": format_auto(course_score.mean_crps(False)),
        "course_crps_exponential": format_auto(course_score.mean_crps(True)),
        "n_forecasts": str(len(course_score.hits)),
        "growth_sentence": _growth_sentence(course.num_positive),
        "hypotheses": (project_root() / "docs" / "hypotheses.md")
        .read_text(encoding="utf-8")
        .strip(),
        "mermaid_inference": compartment_mermaid("inference"),
        "mermaid_simulator": compartment_mermaid("simulator"),
        "posterior": _posterior_status(),
        "reliability_rows": _reliability_rows(course_score, reporting.reliability_bins),
        "cost_labels": [row["label"] for row in costs],
        "cost_values": [
            float(cost)
            for _, cost in sorted(
                game.constraint_costs.items(), key=lambda item: item[1]
            )
        ],
        "learning_days": ", ".join(str(day) for day in inference.learning_curve_days),
        "well_below": format_auto(inference.identifiability.well_below),
        "partial_below": format_auto(inference.identifiability.partial_below),
        "svi_steps": format_fr(inference.svi.num_steps, 0),
        "model_score_status": _model_score_status(),
        "early_tier2": format_percent(game.early_tier2.probability, 0),
        "early_hospital": format_fr(game.early_tier2.hospital_threshold, 0),
        "early_tier3": format_percent(game.early_tier3.probability, 0),
    }
    _assert_filled(context)
    return context


def score_labeled(
    series: ObservationSeries,
    label: str,
    reporting: ReportingConfig,
    period: int,
) -> SeriesScore:
    thresholds = load_thresholds_config()
    low_mass, high_mass = reporting.interval_masses
    return score_observation_series(
        series,
        label=label,
        period=period,
        min_history=reporting.min_history_days,
        horizons=list(reporting.score_horizons),
        low_mass=low_mass,
        high_mass=high_mass,
        ensemble_size=reporting.ensemble_size,
        seed=reporting.seed,
        residual_window=reporting.residual_window,
        series_names=list(reporting.series),
        max_hospitalized=float(thresholds.max_hospitalized),
        event_horizon=reporting.event_horizon,
    )


def simulate_public(level: int, reporting: ReportingConfig) -> ObservationSeries:
    """Lance la ville et ne garde que les colonnes publiées."""
    simulator = RealitySimulator.from_configs(seed=reporting.seed, level=level)
    simulator.run(reporting.score_level_days)
    return observations_from_frame(simulator.observations())


def _event_note(score: SeriesScore) -> str:
    """Précise un Brier bas quand l'événement ne s'est pas produit."""
    if not score.events:
        return "Aucun événement de saturation n'a pu être scoré."
    happened = float(np.mean([item.happened for item in score.events]))
    announced = float(np.mean([item.probability for item in score.events]))
    if happened == 0.0 and announced < 0.05:
        return (
            "Sur cette série, le plafond d'hospitalisés n'est pas franchi "
            "dans les fenêtres scorées, et la persistance ne l'annonce presque pas. "
            "Le Brier est donc bas sans mesurer un événement qui arrive vraiment."
        )
    return (
        "La fréquence observée de l'événement et la probabilité annoncée "
        f"valent {format_percent(happened, 0)} et {format_percent(announced, 0)}."
    )


def _coverage_sentence(score: SeriesScore, high_mass: float, low_mass: float) -> str:
    high = score.coverage(high=True)
    low = score.coverage(high=False)
    return (
        f"Intervalles à {format_percent(high_mass, 0)} : "
        f"{format_percent(high, 0)} de réussite sur {len(score.hits)} prévisions. "
        f"Intervalles à {format_percent(low_mass, 0)} : {format_percent(low, 0)}."
    )


def _summary_table(
    score: SeriesScore,
    labels: dict[str, str],
    low_mass: float,
    high_mass: float,
) -> str:
    low = format_percent(low_mass, 0)
    high = format_percent(high_mass, 0)
    header = (
        f"| Série | Horizon (jours) | Prévisions | Couverture {low} "
        f"| Couverture {high} | CRPS persistance | CRPS exponentielle |"
    )
    lines = [header, "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows(score, labels):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["series"]),
                    str(row["horizon"]),
                    str(row["n"]),
                    format_percent(float(row["coverage_low"]), 0),
                    format_percent(float(row["coverage_high"]), 0),
                    format_auto(float(row["crps_persistence"])),
                    format_auto(float(row["crps_exponential"])),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _level_table(scores: list[SeriesScore], high_mass: float) -> str:
    header = (
        f"| Trajectoire | Prévisions | Couverture {format_percent(high_mass, 0)} "
        "| CRPS persistance | CRPS exponentielle | Brier |"
    )
    lines = [header, "|---|---:|---:|---:|---:|---:|"]
    for score in scores:
        lines.append(
            "| "
            + " | ".join(
                (
                    score.label,
                    str(len(score.hits)),
                    format_percent(score.coverage(high=True), 0),
                    format_auto(score.mean_crps(False)),
                    format_auto(score.mean_crps(True)),
                    format_auto(score.brier()),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _cost_table(rows: list[dict[str, str]]) -> str:
    lines = ["| Mesure | Coût par jour |", "|---|---:|"]
    for row in rows:
        lines.append(f"| {row['label']} | {row['cost']} |")
    return "\n".join(lines)


def _reliability_rows(score: SeriesScore, n_bins: int) -> list[dict[str, float]]:
    if not score.events:
        raise ValueError("Le score du cours n'a pas d'événement de saturation.")
    frame = reliability_table(
        np.array([item.probability for item in score.events]),
        np.array([item.happened for item in score.events], dtype=float),
        n_bins,
    )
    return [
        {
            "announced_mean": float(row.announced_mean),
            "observed_rate": float(row.observed_rate),
            "count": float(row.count),
        }
        for row in frame.itertuples(index=False)
    ]


def _growth_sentence(positive: np.ndarray) -> str:
    window = np.asarray(positive, dtype=float)[-14:]
    if window.size < 2:
        return "La série de cas confirmés est trop courte pour une pente."
    time = np.arange(window.size, dtype=float)
    slope = float(np.polyfit(time, np.log1p(np.clip(window, 0.0, None)), deg=1)[0])
    if slope > 1e-4:
        doubling = math.log(2.0) / slope
        return (
            "Sur les 14 derniers jours de cas confirmés du fichier du cours, "
            f"la pente de log(1 + cas) vaut {format_auto(slope)} par jour, "
            f"soit un temps de doublement d'environ {format_auto(doubling)} jours."
        )
    if slope < -1e-4:
        halving = math.log(2.0) / -slope
        pace = format_auto(halving)
        return (
            "Sur les 14 derniers jours de cas confirmés du fichier du cours, "
            f"la pente de log(1 + cas) vaut {format_auto(slope)} par jour, "
            f"soit un temps de division par deux d'environ {pace} jours."
        )
    return (
        "Sur les 14 derniers jours de cas confirmés du fichier du cours, "
        "la pente de log(1 + cas) est presque nulle."
    )


def _posterior_status() -> dict[str, object]:
    directory = resolve_path(load_inference_config().output_dir)
    files = sorted(directory.glob("*.nc")) if directory.is_dir() else []
    empty = {
        "available": False,
        "status": (
            "Aucun fichier .nc dans le dossier d'inférence. "
            "Les comparaisons entre prior et posterior, le R̂ et l'ESS "
            "apparaissent après epidemic fit."
        ),
        "rhat": "non calculé",
        "ess": "non calculé",
        "divergences": "non calculé",
    }
    if not files:
        return empty
    try:
        import arviz as az

        from epidemic.inference.diagnostics import check_diagnostics

        settings = load_inference_config()
        report = check_diagnostics(
            az.from_netcdf(files[-1]),
            method="nuts",
            max_rhat=settings.diagnostics.max_rhat,
            min_ess=settings.diagnostics.min_ess,
            max_divergences=settings.diagnostics.max_divergences,
            strict=False,
        )
    except (OSError, ValueError, KeyError) as exc:
        empty["status"] = (
            f"Le posterior le plus récent n'a pas pu être lu ({type(exc).__name__})."
        )
        return empty
    return {
        "available": True,
        "status": report.message,
        "rhat": "non calculé"
        if report.max_rhat is None
        else format_auto(report.max_rhat),
        "ess": "non calculé" if report.min_ess is None else format_auto(report.min_ess),
        "divergences": (
            "non calculé" if report.divergences is None else str(report.divergences)
        ),
    }


def _model_score_status() -> str:
    path = resolve_path("data/generated/scoring/model.json")
    if path.is_file():
        return f"Un score de posterior est enregistré dans {path.name}."
    return (
        "Aucun score de posterior n'est enregistré. "
        "Les couvertures de cette section sont celles des références naïves, "
        "persistance et croissance exponentielle, pas encore celles du modèle."
    )


def _assert_filled(value: object, path: str = "contexte") -> None:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"Champ vide : {path}.")
        return
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"Dictionnaire vide : {path}.")
        for key, item in value.items():
            _assert_filled(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"Liste vide : {path}.")
        for index, item in enumerate(value):
            _assert_filled(item, f"{path}[{index}]")
        return
    if isinstance(value, (bool, int, float)):
        return
    if value is None:
        raise ValueError(f"Champ nul : {path}.")
    raise ValueError(f"Type inattendu pour {path} : {type(value).__name__}.")


def course_frame() -> pd.DataFrame:
    """Relit le CSV du cours, pour les bulletins."""
    path = resolve_path(load_inference_config().course_data)
    return pd.read_csv(path)


def output_base(destination: Path | None) -> Path:
    """Racine d'écriture : le dépôt, ou un dossier de test."""
    return project_root() if destination is None else destination
