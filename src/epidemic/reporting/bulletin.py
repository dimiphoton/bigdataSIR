"""Bulletin hebdomadaire : phrases par règles, gabarit Quarto, PDF de deux pages."""

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from epidemic.config import (
    GameConfig,
    ReportingConfig,
    load_game_config,
    load_reporting_config,
    load_thresholds_config,
)
from epidemic.forecast.project import LatentSegment
from epidemic.forecast.scoring import naive_ensemble
from epidemic.forecast.verbal import verbal_probability
from epidemic.reporting.context import course_frame
from epidemic.reporting.text import format_auto, format_fr, format_percent, french_date
from epidemic.series import ObservationSeries, observations_from_frame
from epidemic.simulator.interventions import LABELS

_TEMPLATE = "bulletin.qmd.j2"
_RESERVOIR_NOTE = (
    "Abscisse : réservoir infectieux estimé I(t). "
    "Ordonnée : nouvelles infections estimées par jour. "
    "Les deux échelles sont logarithmiques. "
    "La médiane est entourée des bandes à 50 % et 90 %. "
    "La droite de pente 1 traverse la phase exponentielle initiale, "
    "où dI/I = β S/N. "
    "Les marqueurs indiquent le début des interventions. "
    "Le second panneau montre dI/I au cours du temps, en échelle logarithmique."
)


def write_bulletin(
    day: int,
    *,
    frame: pd.DataFrame | None = None,
    destination: Path | None = None,
    intervention_days: list[int] | None = None,
) -> list[Path]:
    """Écrit le bulletin du jour demandé, en Quarto et en PDF."""
    settings = load_reporting_config()
    table = course_frame() if frame is None else frame
    payload = bulletin_payload(day, table)
    folder = destination or (Path(settings.bulletin_dir))
    if destination is None:
        from epidemic.config import resolve_path

        folder = resolve_path(settings.bulletin_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"bulletin-j{day:03d}"
    incidence = folder / f"{stem}-incidence.png"
    hospital = folder / f"{stem}-hopital.png"
    reservoir = folder / f"{stem}-reservoir.png"
    segment, reservoir_note = _reservoir_segment(day)
    _save_incidence(incidence, payload, settings.figure_dpi)
    _save_hospital(hospital, payload, settings.figure_dpi)
    _save_reservoir(reservoir, segment, intervention_days, settings.figure_dpi)
    payload["incidence_figure"] = incidence.name
    payload["hospital_figure"] = hospital.name
    payload["reservoir_figure"] = reservoir.name
    payload["reservoir_note"] = reservoir_note
    from epidemic.reporting.render import render_named

    qmd = folder / f"{stem}.qmd"
    qmd.write_text(render_named(_TEMPLATE, payload), encoding="utf-8")
    pdf = folder / f"{stem}.pdf"
    _write_pdf(pdf, payload, incidence, hospital, reservoir)
    return [qmd, incidence, hospital, reservoir, pdf]


def bulletin_payload(day: int, frame: pd.DataFrame) -> dict[str, str]:
    """Chiffres et phrases du bulletin. Toutes les chaînes sont remplies."""
    game = load_game_config()
    reporting = load_reporting_config()
    thresholds = load_thresholds_config()
    if day < 1 or day % game.bulletin_period_days != 0:
        raise ValueError(
            f"Le jour {day} n'est pas un jour de bulletin "
            f"(période {game.bulletin_period_days})."
        )
    if "Day" not in frame.columns or int(frame["Day"].max()) < day:
        raise ValueError(f"La série publiée ne va pas jusqu'au jour {day}.")
    kept = frame.loc[frame["Day"] <= day].copy()
    series = observations_from_frame(kept)
    if series.n_days < 2:
        raise ValueError("Il faut au moins deux jours pour un bulletin.")
    from epidemic.config import load_simulator_config

    level, allowed = _tier(day, game)
    allowed_text = (
        ", ".join(LABELS.get(name, name) for name in allowed)
        if allowed
        else "aucune, observation seulement"
    )
    current = _week(series, 0)
    previous = _week(series, 1) if series.n_days >= 14 else None
    date = french_date(load_simulator_config().day1_date + timedelta(days=day - 1))
    sentences = _situation(day, date, current, previous, reporting.trend_ratio)
    horizon_text, verbal, probability = _outlook(
        series, reporting, thresholds.max_hospitalized
    )
    ceiling = format_fr(thresholds.max_hospitalized, 0)
    chance = format_percent(probability, 0)
    sentences.append(
        "La probabilité estimée, selon la référence naïve de persistance, "
        f"que les hospitalisés confirmés dépassent {ceiling} "
        f"dans les {reporting.event_horizon} jours est {chance} "
        f"({verbal}). Ce n'est pas encore la probabilité du modèle bayésien."
    )
    recommendation = (
        f"Palier {level}. Mesures autorisées : {allowed_text}. "
        "La recommandation estimée, soit le coût le plus bas sous le seuil de "
        "saturation, sera jointe quand une loi a posteriori sera enregistrée."
    )
    payload = {
        "day": str(day),
        "date": date,
        "tier": str(level),
        "allowed": allowed_text,
        "situation": " ".join(sentences),
        "indicators": _indicator_table(current, previous),
        "outlook": horizon_text,
        "probability": format_percent(probability, 0),
        "verbal": verbal,
        "recommendation": recommendation,
        "max_hospitalized": format_fr(thresholds.max_hospitalized, 0),
        "max_critical": format_fr(thresholds.max_critical, 0),
        "positive": " ".join(str(int(value)) for value in series.num_positive),
        "hospital_series": " ".join(
            str(int(value)) for value in series.num_hospitalised
        ),
        "critical_series": " ".join(str(int(value)) for value in series.num_critical),
    }
    for key, value in payload.items():
        if not str(value).strip():
            raise ValueError(f"Bulletin incomplet : {key}.")
    return payload


def _situation(
    day: int,
    date: str,
    current: dict[str, float | str],
    previous: dict[str, float | str] | None,
    trend_ratio: float,
) -> list[str]:
    sentences = [
        (
            f"Au jour {day} ({date}), les cas confirmés des 7 derniers jours "
            f"sont {format_fr(float(current['cases']), 0)}"
            + _trend_clause(
                current["cases"],
                None if previous is None else previous["cases"],
                trend_ratio,
            )
            + "."
        ),
        (
            "Le taux de positivité confirmé sur 7 jours est "
            f"{current['positivity']}"
            + _trend_clause(
                current["positivity_value"],
                None if previous is None else previous["positivity_value"],
                trend_ratio,
            )
            + "."
        ),
        (
            "Les hospitalisés confirmés sont "
            f"{format_fr(float(current['hospital']), 0)}"
            + _trend_clause(
                current["hospital"],
                None if previous is None else previous["hospital"],
                trend_ratio,
            )
            + ", et les soins intensifs confirmés sont "
            + format_fr(float(current["critical"]), 0)
            + _trend_clause(
                current["critical"],
                None if previous is None else previous["critical"],
                trend_ratio,
            )
            + "."
        ),
        (
            "Les nouveaux décès confirmés de la semaine sont "
            f"{format_fr(float(current['deaths']), 0)}."
        ),
    ]
    return sentences


def _outlook(
    series: ObservationSeries,
    reporting: ReportingConfig,
    max_hospitalized: float,
) -> tuple[str, str, float]:
    horizons = list(reporting.bulletin_horizons)
    pieces = []
    probability = 0.0
    for index, horizon in enumerate(horizons):
        hospital = naive_ensemble(
            series.num_hospitalised,
            horizon,
            kind="persistence",
            n_draws=reporting.ensemble_size,
            seed=reporting.seed + horizon,
            window=reporting.residual_window,
        )
        cases = naive_ensemble(
            series.num_positive,
            horizon,
            kind="persistence",
            n_draws=reporting.ensemble_size,
            seed=reporting.seed + 17 + horizon,
            window=reporting.residual_window,
        )
        pieces.append(
            f"À {horizon} jours, la médiane estimée de la référence naïve est "
            f"{format_fr(float(np.median(cases[:, -1])), 0)} cas confirmés et "
            f"{format_fr(float(np.median(hospital[:, -1])), 0)} hospitalisés."
        )
        if horizon == reporting.event_horizon or index == len(horizons) - 1:
            probability = float(np.mean(np.any(hospital > max_hospitalized, axis=1)))
    verbal = verbal_probability(probability)
    return " ".join(pieces), verbal, probability


def _week(series: ObservationSeries, shift: int) -> dict[str, float | str]:
    end = series.n_days - shift * 7
    start = end - 7
    if start < 0:
        raise ValueError("La semaine demandée sort de la série.")
    cases = float(series.num_positive[start:end].sum())
    tested = float(series.num_tested[start:end].sum())
    positivity = cases / tested if tested > 0 else 0.0
    return {
        "cases": cases,
        "tested": tested,
        "positivity_value": positivity,
        "positivity": "non calculé" if tested <= 0 else format_percent(positivity, 1),
        "hospital": float(series.num_hospitalised[end - 1]),
        "critical": float(series.num_critical[end - 1]),
        "deaths": float(series.death_increments[start:end].sum()),
    }


def _trend_clause(
    current: float | str,
    previous: float | str | None,
    trend_ratio: float,
) -> str:
    if previous is None:
        return ", sans semaine précédente complète"
    return ", " + _trend_word(float(current), float(previous), trend_ratio)


def _trend_word(current: float, previous: float, trend_ratio: float) -> str:
    if previous <= 0:
        if current > 0:
            return "en hausse par rapport à la semaine précédente"
        return "stable par rapport à la semaine précédente"
    ratio = current / previous
    if ratio >= trend_ratio:
        return "en hausse par rapport à la semaine précédente"
    if ratio <= 1.0 / trend_ratio:
        return "en baisse par rapport à la semaine précédente"
    return "stable par rapport à la semaine précédente"


def _indicator_table(
    current: dict[str, float | str],
    previous: dict[str, float | str] | None,
) -> str:
    rows = [
        (
            "Cas confirmés, 7 jours",
            current["cases"],
            None if previous is None else previous["cases"],
        ),
        (
            "Positivité confirmée, 7 jours",
            current["positivity_value"],
            None if previous is None else previous["positivity_value"],
        ),
        (
            "Hospitalisés confirmés",
            current["hospital"],
            None if previous is None else previous["hospital"],
        ),
        (
            "Soins intensifs confirmés",
            current["critical"],
            None if previous is None else previous["critical"],
        ),
        (
            "Nouveaux décès confirmés",
            current["deaths"],
            None if previous is None else previous["deaths"],
        ),
    ]
    lines = ["| Indicateur | Valeur | Semaine précédente |", "|---|---:|---:|"]
    for label, value, old in rows:
        shown = (
            current["positivity"]
            if "Positivité" in label
            else format_auto(float(value))
        )
        if "Positivité" in label and old is not None:
            previous_text = format_percent(float(old), 1)
        elif old is None:
            previous_text = "—"
        else:
            previous_text = format_auto(float(old))
        lines.append(f"| {label} | {shown} | {previous_text} |")
    return "\n".join(lines)


def _tier(day: int, game: GameConfig) -> tuple[int, list[str]]:
    level = 0
    names: list[str] = []
    for tier in game.tiers:
        if day >= tier.from_day and tier.level >= level:
            level = tier.level
            names = list(tier.interventions)
    return level, names


def _reservoir_segment(day: int) -> tuple[LatentSegment, str]:
    """Segment latent du jour : posterior s'il couvre le bulletin, sinon grille de β."""
    posterior = _posterior_reservoir(day)
    if posterior is not None:
        return posterior, _RESERVOIR_NOTE
    return _demo_reservoir(day), (
        "Aucun posterior n'est enregistré pour ce jour. "
        "La courbe est la trajectoire latente du modèle d'inférence "
        "sur une grille de β. " + _RESERVOIR_NOTE
    )


def _posterior_reservoir(day: int) -> LatentSegment | None:
    """Projection du dernier posterior si sa fenêtre est exactement ce bulletin."""
    import arviz as az

    from epidemic.config import load_inference_config, resolve_path
    from epidemic.forecast.idata import draws_from_idata
    from epidemic.forecast.project import (
        as_latent_segment,
        project_scenarios,
        subsample,
    )
    from epidemic.model.age_structure import load_age_structure

    directory = resolve_path(load_inference_config().output_dir)
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.nc"), key=lambda item: item.stat().st_mtime)
    if not files:
        return None
    draws = draws_from_idata(az.from_netcdf(files[-1]))
    if draws.n_past != day:
        return None
    if draws.n_draws > 24:
        draws = subsample(draws, 24, seed=load_reporting_config().seed)
    structure = load_age_structure(draws.n_groups)
    projection = project_scenarios(
        draws,
        structure,
        [np.asarray(structure.contacts, dtype=float)],
        n_days=day,
        switch_index=day,
        scales=np.ones((1, draws.n_draws)),
        seed=load_reporting_config().seed,
    )[0]
    return as_latent_segment(projection)


def _demo_reservoir(day: int) -> LatentSegment:
    """Trajectoire du modèle à β fixé sur une petite grille, sans ajustement."""
    from epidemic.forecast.project import (
        as_latent_segment,
        constant_draws,
        project_scenarios,
    )
    from epidemic.model.age_structure import load_age_structure

    structure = load_age_structure(2)
    n_draws = 6
    draws = constant_draws(
        n_draws,
        day,
        structure.n_groups,
        beta=0.06,
        e0=50.0,
        process_sigma=0.0,
    )
    grid = np.linspace(0.045, 0.08, n_draws)
    draws.beta_t = grid[:, None] * np.ones(day)
    projection = project_scenarios(
        draws,
        structure,
        [np.asarray(structure.contacts, dtype=float)],
        n_days=day,
        switch_index=day,
        scales=np.ones((1, n_draws)),
        seed=load_reporting_config().seed,
    )[0]
    return as_latent_segment(projection)


def _save_reservoir(
    path: Path,
    segment: LatentSegment,
    intervention_days: list[int] | None,
    dpi: int,
) -> None:
    from epidemic.viz.tracking import plot_reservoir

    figure = plot_reservoir(segment, intervention_days=intervention_days)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _save_incidence(path: Path, payload: dict[str, str], dpi: int) -> None:
    values = np.array([float(item) for item in payload["positive"].split()])
    day = np.arange(1, values.size + 1)
    smooth = np.convolve(values, np.ones(7) / 7, mode="same")
    figure, axis = plt.subplots(figsize=(7, 3.2))
    axis.semilogy(
        day, np.clip(values, 0.1, None), color="#9bb8d3", label="Cas confirmés"
    )
    axis.semilogy(
        day, np.clip(smooth, 0.1, None), color="#1f4e79", label="Moyenne 7 jours"
    )
    axis.set_xlabel("Jour")
    axis.set_ylabel("Cas confirmés")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _save_hospital(path: Path, payload: dict[str, str], dpi: int) -> None:
    hospital = np.array([float(item) for item in payload["hospital_series"].split()])
    critical = np.array([float(item) for item in payload["critical_series"].split()])
    day = np.arange(1, hospital.size + 1)
    figure, axis = plt.subplots(figsize=(7, 3.2))
    axis.plot(day, hospital, color="#1f4e79", label="Hospitalisés confirmés")
    axis.plot(day, critical, color="#b85c38", label="Soins intensifs confirmés")
    axis.axhline(
        float(payload["max_hospitalized"].replace(" ", "")),
        color="#1f4e79",
        ls="--",
        lw=0.8,
    )
    axis.axhline(
        float(payload["max_critical"].replace(" ", "")),
        color="#b85c38",
        ls="--",
        lw=0.8,
    )
    axis.set_xlabel("Jour")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _write_pdf(
    path: Path,
    payload: dict[str, str],
    incidence: Path,
    hospital: Path,
    reservoir: Path,
) -> None:
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.suptitle(
            f"Bulletin du jour {payload['day']} — {payload['date']}",
            fontsize=14,
            y=0.97,
        )
        figure.text(0.08, 0.90, "Situation en bref", fontsize=12, fontweight="bold")
        figure.text(
            0.08, 0.86, _wrap(payload["situation"]), fontsize=9, va="top", wrap=True
        )
        figure.text(0.08, 0.62, "Indicateurs confirmés", fontsize=12, fontweight="bold")
        figure.text(
            0.08, 0.58, payload["indicators"], fontsize=8, va="top", family="monospace"
        )
        image_axis = figure.add_axes((0.12, 0.08, 0.76, 0.28))
        image_axis.imshow(plt.imread(incidence))
        image_axis.axis("off")
        pdf.savefig(figure)
        plt.close(figure)

        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.08, 0.96, "Hôpital et capacité", fontsize=12, fontweight="bold")
        image_axis = figure.add_axes((0.12, 0.80, 0.76, 0.14))
        image_axis.imshow(plt.imread(hospital))
        image_axis.axis("off")
        figure.text(0.08, 0.77, "Réservoir infectieux", fontsize=12, fontweight="bold")
        image_axis = figure.add_axes((0.10, 0.46, 0.80, 0.29))
        image_axis.imshow(plt.imread(reservoir))
        image_axis.axis("off")
        figure.text(0.08, 0.43, "Projections", fontsize=12, fontweight="bold")
        figure.text(0.08, 0.40, _wrap(payload["outlook"]), fontsize=8, va="top")
        figure.text(
            0.08,
            0.30,
            _wrap(
                "P(saturation hospitalière, référence naïve) estimée à "
                f"{payload['probability']} ({payload['verbal']})."
            ),
            fontsize=8,
            va="top",
        )
        figure.text(0.08, 0.22, "Recommandation", fontsize=12, fontweight="bold")
        figure.text(0.08, 0.19, _wrap(payload["recommendation"]), fontsize=8, va="top")
        figure.text(
            0.08,
            0.06,
            _wrap(
                f"Plafonds : {payload['max_hospitalized']} hospitalisés, "
                f"{payload['max_critical']} soins intensifs."
            ),
            fontsize=8,
        )
        pdf.savefig(figure)
        plt.close(figure)


def _wrap(text: str, width: int = 95) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))
