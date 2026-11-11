"""Ligne de commande du projet."""

from pathlib import Path
from typing import Annotated

import typer

from epidemic.config import (
    load_game_config,
    load_inference_config,
    load_simulator_config,
    resolve_path,
)
from epidemic.simulator.engine import RealitySimulator
from epidemic.simulator.game import Decision, Game
from epidemic.simulator.interventions import LABELS

app = typer.Typer(
    help="Projet SEIR bayésien du cours PROJ0016.",
    no_args_is_help=True,
)


def _parse_decision(line: str) -> Decision:
    parts = line.split()
    if not parts:
        raise ValueError("Ligne vide.")
    name = parts[0]
    if name in ("mask", "lockdown"):
        if len(parts) != 1:
            raise ValueError(f"{LABELS[name]} ne prend pas de paramètre.")
        return Decision(name)
    if name in ("case_isolation", "household_quarantine"):
        if len(parts) != 2:
            raise ValueError(f"{LABELS[name]} : indiquez la durée en jours.")
        return Decision(name, duration_days=int(parts[1]))
    if name == "social_distancing":
        if len(parts) != 2:
            raise ValueError("Distanciation sociale : indiquez l'âge seuil.")
        return Decision(name, age_threshold=int(parts[1]))
    raise ValueError(f"Intervention inconnue : {name}.")


def _read_decision_lines() -> list[str]:
    typer.echo(
        "Une mesure par ligne, puis « fin ». « fin » seul : aucune mesure nouvelle."
    )
    typer.echo("Exemples : mask | case_isolation 10 | social_distancing 60 | lockdown")
    lines: list[str] = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line == "" or line.lower() == "fin":
            break
        lines.append(line)
    return lines


@app.command()
def simulate(
    days: Annotated[
        int | None,
        typer.Option(
            "--days", help="Nombre de jours. Par défaut, jusqu'à la fin de partie."
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Graine. Par défaut, celle du YAML.")
    ] = None,
    level: Annotated[
        int | None,
        typer.Option(
            "--level", min=0, max=3, help="Niveau de stochasticité, de 0 à 3."
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Dossier des fichiers produits."),
    ] = None,
) -> None:
    """Lance le simulateur sans décision et écrit les séries."""
    game_config = load_game_config()
    horizon = game_config.end_day if days is None else days
    if horizon < 1:
        raise typer.BadParameter("Le nombre de jours doit être au moins 1.")
    simulator = RealitySimulator.from_configs(seed=seed, level=level)
    simulator.run(horizon)
    directory = output_dir or resolve_path(load_simulator_config().output_dir)
    simulator.write_outputs(directory)
    frame = simulator.observations()
    last = frame.iloc[-1]
    typer.echo(
        "Jour "
        f"{int(last['Day'])}. Positifs du jour : {int(last['num_positive'])}. "
        f"Tests : {int(last['num_tested'])}. "
        f"Hospitalisés : {int(last['num_hospitalised'])}. "
        f"Soins intensifs : {int(last['num_critical'])}. "
        f"Décès : {int(last['num_fatalities'])}."
    )
    typer.echo(f"Observations : {directory / 'observations.csv'}")
    typer.echo(f"Paramètres secrets : {directory / '.truth.json'}")


@app.command()
def play(
    seed: Annotated[
        int | None, typer.Option("--seed", help="Graine de la partie.")
    ] = None,
    level: Annotated[
        int | None,
        typer.Option(
            "--level", min=0, max=3, help="Niveau de stochasticité, de 0 à 3."
        ),
    ] = None,
    end_day: Annotated[
        int | None, typer.Option("--end-day", help="Dernier jour, pour un essai court.")
    ] = None,
    course: Annotated[
        bool,
        typer.Option(
            "--course", help="Utilise data.csv comme réalité des 71 premiers jours."
        ),
    ] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Joue une partie : un bulletin par semaine, puis une décision."""
    source = "course" if course else None
    directory = output_dir or resolve_path(load_simulator_config().output_dir)
    game = Game.start(
        seed=seed,
        level=level,
        observation_source=source,
        end_day=end_day,
        output_dir=directory,
        write_secret=True,
    )
    typer.echo(
        "Les paramètres de la ville sont cachés. Seules les séries publiées sont là."
    )
    while game.day < game.end_day and game.next_bulletin_day() is not None:
        game.next_bulletin()
        _print_bulletin(game)
        if game.day >= game.end_day:
            break
        while True:
            try:
                decisions = [_parse_decision(line) for line in _read_decision_lines()]
                game.decide(decisions)
            except ValueError as exc:
                typer.echo(str(exc))
                typer.echo(
                    "La décision n'est pas enregistrée. Recommencez, ou « fin »."
                )
                continue
            else:
                break
        if decisions:
            labels = ", ".join(LABELS.get(item.name, item.name) for item in decisions)
            typer.echo(f"Décision enregistrée : {labels}.")
        else:
            typer.echo("Aucune mesure ajoutée.")
        active = game.active_names()
        if active:
            typer.echo(
                "Déjà en vigueur : "
                + ", ".join(LABELS.get(name, name) for name in active)
            )
    if game.day < game.end_day:
        game.advance(game.end_day - game.day)
    game.simulator.write_outputs(directory)
    game.save(directory / "game.json")
    _print_score(game)
    _print_calibration(game)


def _print_bulletin(game: Game) -> None:
    frame = game.observations()
    typer.echo("")
    typer.echo(f"--- Bulletin du jour {game.day} | palier {game.tier_level()} ---")
    shown = frame.tail(7)
    typer.echo(shown.to_string(index=False))
    for item in game.revealed():
        typer.echo(f"Information (J{item.day}) — {item.title} : {item.body.strip()}")
    allowed = game.allowed_interventions()
    if allowed:
        typer.echo(
            "Mesures autorisées : "
            + ", ".join(LABELS.get(name, name) for name in allowed)
        )
    else:
        typer.echo("Aucune mesure autorisée : observation seulement.")
    costs = game.game_config.constraint_costs
    typer.echo(
        "Coûts par jour : "
        + ", ".join(f"{LABELS.get(name, name)} {costs[name]:g}" for name in costs)
    )


def _print_score(game: Game) -> None:
    result = game.score()
    typer.echo("")
    typer.echo(
        f"Fin de partie, jour {game.day}. "
        f"Jours en saturation : {result.days_saturated}. "
        f"Décès : {result.deaths}. "
        f"Coût des contraintes : {result.constraint_cost:g}."
    )


def _print_calibration(game: Game) -> None:
    """Couverture des références naïves sur les séries publiées de la partie."""
    from epidemic.config import load_reporting_config
    from epidemic.forecast.scoring import interpret_coverage
    from epidemic.reporting.context import score_labeled
    from epidemic.reporting.text import format_percent
    from epidemic.series import observations_from_frame

    reporting = load_reporting_config()
    try:
        result = score_labeled(
            observations_from_frame(game.observations()),
            "partie",
            reporting,
            game.game_config.bulletin_period_days,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        return
    high = reporting.interval_masses[1]
    typer.echo(
        f"Calibration naïve, couverture {format_percent(high, 0)} : "
        f"{format_percent(result.coverage(high=True), 0)} "
        f"sur {len(result.hits)} prévisions."
    )
    typer.echo(
        interpret_coverage(
            result.coverage(high=True),
            high,
            reporting.coverage_tolerance,
        )
    )


@app.command()
def fit(
    source: Annotated[
        str,
        typer.Option("--source", help="simulator : ville simulée. course : data.csv."),
    ] = "simulator",
    days: Annotated[
        int | None,
        typer.Option(
            "--days", help="Nombre de jours. Par défaut : 40, ou tout le fichier cours."
        ),
    ] = None,
    method: Annotated[
        str | None,
        typer.Option("--method", help="nuts (défaut du YAML) ou svi, plus rapide."),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option("--variant", help="A, B ou C."),
    ] = None,
    beta_mode: Annotated[
        str | None,
        typer.Option("--beta-mode", help="weekly_rw ou piecewise."),
    ] = None,
    level: Annotated[
        int,
        typer.Option(
            "--level",
            min=0,
            max=3,
            help="Stochasticité du simulateur. 0 pour retrouver des paramètres.",
        ),
    ] = 0,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    strict: Annotated[
        bool | None,
        typer.Option(
            "--strict/--no-strict",
            help="Échoue si R̂, l'ESS ou les divergences sont mauvais.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Fichier posterior.nc."),
    ] = None,
) -> None:
    """Ajuste le SEIRH étendu sur des observations publiques."""
    from epidemic.inference.diagnostics import DiagnosticsError
    from epidemic.inference.fit import configured, fit_observations
    from epidemic.inference.validation import load_course_series, simulate_public_series
    from epidemic.series import ObservationSeries

    if source not in ("simulator", "course"):
        raise typer.BadParameter("source doit être simulator ou course.")
    settings = configured(
        load_inference_config(),
        method=method,
        variant=variant,
        beta_mode=beta_mode,
        seed=seed,
        strict=strict,
    )
    series: ObservationSeries
    if source == "course":
        series = load_course_series(days)
    else:
        horizon = settings.default_days if days is None else days
        series = simulate_public_series(level, horizon, settings.seed)
    typer.echo(
        f"Ajustement {settings.method}, variante {settings.variant}, "
        f"β {settings.beta_mode}, {series.n_days} jours."
    )
    if settings.method == "nuts":
        typer.echo(
            f"{settings.nuts.num_chains} chaînes, "
            f"{settings.nuts.num_warmup} pas d'échauffement, "
            f"{settings.nuts.num_samples} tirages."
        )
    try:
        result = fit_observations(
            series,
            settings,
            output_path=output,
            show_progress=True,
        )
    except DiagnosticsError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    report = result.diagnostics
    typer.echo(report.message)
    if result.elbo is not None:
        typer.echo(f"ELBO final : {result.elbo:.1f}.")
    if result.output_path is not None:
        typer.echo(f"Posterior : {result.output_path}")


@app.command()
def docs(
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Racine d'écriture. Par défaut, le dépôt."),
    ] = None,
) -> None:
    """Génère la documentation bayésienne et les diapositives Marp."""
    from epidemic.reporting.render import render_docs

    for path in render_docs(output_dir):
        typer.echo(path)


@app.command()
def slides() -> None:
    """Exporte la présentation Marp en HTML et en PDF."""
    from epidemic.reporting.render import export_slides

    written = export_slides()
    if not written:
        typer.echo("marp n'a rien écrit. La source reste docs/presentation.md.")
        raise typer.Exit(code=1)
    for path in written:
        typer.echo(path)


@app.command()
def bulletin(
    day: Annotated[int, typer.Option("--day", help="Jour de bulletin, multiple de 7.")],
    source: Annotated[
        str,
        typer.Option("--source", help="course : data.csv. csv : fichier --input."),
    ] = "course",
    csv_path: Annotated[
        Path | None,
        typer.Option("--input", help="CSV au format du cours, si --source csv."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Écrit le bulletin d'un jour : phrases, graphes, Quarto et PDF."""
    import pandas as pd

    from epidemic.reporting.bulletin import write_bulletin

    frame = None
    if source == "csv":
        if csv_path is None:
            raise typer.BadParameter("Avec --source csv, indiquez --input.")
        frame = pd.read_csv(csv_path)
    elif source != "course":
        raise typer.BadParameter("source vaut course ou csv.")
    try:
        written = write_bulletin(day, frame=frame, destination=output_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for path in written:
        typer.echo(path)


@app.command()
def score(
    level: Annotated[
        int | None,
        typer.Option(
            "--level", min=0, max=3, help="Un niveau du simulateur. Défaut : le cours."
        ),
    ] = None,
) -> None:
    """Affiche la calibration des références naïves."""
    from epidemic.config import load_game_config, load_reporting_config
    from epidemic.forecast.scoring import interpret_coverage
    from epidemic.inference.validation import load_course_series
    from epidemic.reporting.context import score_labeled, simulate_public
    from epidemic.reporting.text import format_percent

    reporting = load_reporting_config()
    if level is None:
        series = load_course_series()
        label = "données du cours"
    else:
        series = simulate_public(level, reporting)
        label = f"niveau {level}"
    result = score_labeled(
        series, label, reporting, load_game_config().bulletin_period_days
    )
    high = reporting.interval_masses[1]
    typer.echo(
        f"{label}. Couverture {format_percent(high, 0)} : "
        f"{format_percent(result.coverage(high=True), 0)} "
        f"sur {len(result.hits)} prévisions."
    )
    typer.echo(
        interpret_coverage(
            result.coverage(high=True),
            high,
            reporting.coverage_tolerance,
        )
    )


@app.command()
def report(
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    pdf: Annotated[
        bool,
        typer.Option("--pdf/--no-pdf", help="Tente aussi un PDF via Edge ou Chrome."),
    ] = True,
) -> None:
    """Produit la documentation, le rapport Quarto, le HTML et le PDF si possible."""
    from epidemic.reporting.render import render_report

    for path in render_report(output_dir, pdf=pdf):
        typer.echo(str(path))
