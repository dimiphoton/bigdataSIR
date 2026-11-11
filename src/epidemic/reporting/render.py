"""Écrit la documentation, le rapport et les diapositives depuis les gabarits."""

import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from epidemic.config import load_reporting_config, project_root
from epidemic.reporting.context import build_context, output_base
from epidemic.reporting.figures import (
    prior_trajectory_bundle,
    save_compartments,
    save_costs,
    save_prior_densities,
    save_prior_paths,
    save_reliability,
)
from epidemic.reporting.html_export import markdown_to_html, write_pdf

_FIGURES = {
    "priors": "generated/figures/priors.png",
    "compartments": "generated/figures/compartiments.png",
    "prior_predictive": "generated/figures/prior-predictive.png",
    "reliability": "generated/figures/fiabilite.png",
    "costs": "generated/figures/couts.png",
}


def render_docs(destination: Path | None = None) -> list[Path]:
    """Génère les deux markdown et leurs figures."""
    return _write(destination, report=False, pdf=False)


def render_report(destination: Path | None = None, *, pdf: bool = True) -> list[Path]:
    """Génère la documentation, le rapport Quarto, le HTML et, si possible, le PDF."""
    return _write(destination, report=True, pdf=pdf)


def render_named(template_name: str, context: Mapping[str, object]) -> str:
    """Rend un gabarit déjà complet, utilisé par le bulletin."""
    text = _environment().get_template(template_name).render(**context)
    _reject_leftovers(text, template_name)
    return text


def export_slides(markdown: Path | None = None) -> list[Path]:
    """Exporte la présentation Marp en HTML et en PDF si la commande marp existe."""
    source = markdown or (project_root() / "docs" / "presentation.md")
    if not source.is_file():
        raise FileNotFoundError("presentation.md manque. Lancez d'abord epidemic docs.")
    written: list[Path] = []
    for suffix in (".html", ".pdf"):
        target = source.parent / "generated" / f"presentation{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    *_marp_executable(),
                    str(source),
                    "-o",
                    str(target),
                    "--allow-local-files",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return written
        if result.returncode == 0 and target.is_file():
            written.append(target)
    return written


def _marp_pdf(source: Path, target: Path) -> None:
    """Deuxième essai pour le PDF du rapport, avec Marp si Edge est absent."""
    try:
        subprocess.run(
            [
                *_marp_executable(),
                str(source),
                "-o",
                str(target),
                "--allow-local-files",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return


def _marp_executable() -> list[str]:
    """Trouve marp. Sous Windows, `which` renvoie souvent le script PowerShell."""
    found = shutil.which("marp")
    if found is None:
        return ["marp"]
    path = Path(found)
    command = path.with_suffix(".cmd")
    if path.suffix.lower() == ".ps1" and command.is_file():
        return [str(command)]
    return [str(path)]


def _write(destination: Path | None, *, report: bool, pdf: bool) -> list[Path]:
    base = output_base(destination)
    settings = load_reporting_config()
    context = build_context()
    figure_dir = base / settings.figures_dir
    bundle = prior_trajectory_bundle(settings)
    save_prior_densities(figure_dir / "priors.png", settings)
    save_compartments(figure_dir / "compartiments.png", settings)
    save_prior_paths(figure_dir / "prior-predictive.png", bundle, settings)
    rows = context["reliability_rows"]
    labels = context["cost_labels"]
    values = context["cost_values"]
    if (
        not isinstance(rows, list)
        or not isinstance(labels, list)
        or not isinstance(values, list)
    ):
        raise TypeError("Le contexte des figures n'a pas la forme attendue.")
    save_reliability(
        figure_dir / "fiabilite.png",
        cast(list[dict[str, float]], rows),
        settings,
    )
    save_costs(
        figure_dir / "couts.png",
        cast(list[str], labels),
        cast(list[float], values),
        settings,
    )
    context["figure"] = dict(_FIGURES)
    docs_dir = base / settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _emit(
            docs_dir / "framework-bayesien.md",
            "framework-bayesien.md.j2",
            context,
            docs_dir,
        ),
        _emit(docs_dir / "presentation.md", "presentation.md.j2", context, docs_dir),
    ]
    if report:
        report_context = dict(context)
        report_context["figure"] = {
            name: f"../docs/{path}" for name, path in _FIGURES.items()
        }
        report_dir = base / settings.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        qmd = _emit(
            report_dir / "rapport.qmd", "rapport.qmd.j2", report_context, report_dir
        )
        html_text = markdown_to_html(
            qmd.read_text(encoding="utf-8"), "Rapport SEIR bayésien"
        )
        html_path = report_dir / "rapport.html"
        html_path.write_text(html_text, encoding="utf-8")
        written.extend([qmd, html_path])
        pdf_path = report_dir / "rapport.pdf"
        if pdf and not write_pdf(html_path, pdf_path):
            _marp_pdf(qmd, pdf_path)
        if pdf and pdf_path.is_file():
            written.append(pdf_path)
    for name in _FIGURES.values():
        written.append(docs_dir / name)
    return written


def _emit(
    path: Path, template_name: str, context: dict[str, object], base: Path
) -> Path:
    text = render_named(template_name, context)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    missing = _missing_figures(text, base)
    if missing:
        raise FileNotFoundError(
            f"Figures absentes dans {path.name} : {', '.join(missing)}."
        )
    return path


def _missing_figures(markdown: str, base: Path) -> list[str]:
    missing = []
    for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown):
        if match.startswith("http"):
            continue
        if not (base / match).is_file():
            missing.append(match)
    return missing


def _reject_leftovers(text: str, name: str) -> None:
    if "{{" in text or "{%" in text or text.strip() == "":
        raise ValueError(f"Le gabarit {name} a laissé un champ vide ou une balise.")


def _environment() -> Environment:
    root = project_root()
    return Environment(
        loader=FileSystemLoader(
            [root / "docs" / "templates", root / "reports" / "templates"]
        ),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
