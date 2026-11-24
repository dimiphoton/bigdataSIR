"""DAG du modèle NumPyro et tableau des priors.

``numpyro.render_model`` est tenté. Sans Graphviz, on garde le tableau et un
rappel du schéma de compartiments.
"""

from pathlib import Path

from epidemic.config import load_inference_config, load_priors_config
from epidemic.model.age_structure import load_age_structure
from epidemic.model.priors import prior_catalogue
from epidemic.model.seirh import model_kwargs, seirh_model
from epidemic.viz.model_diagram import compartment_mermaid


def prior_table_markdown() -> str:
    """Tableau des priors, une ligne par loi du catalogue."""
    header = "| Paramètre | Loi | Paramètres | Source | Justification |"
    rule = "|---|---|---|---|---|"
    rows = [header, rule]
    for item in prior_catalogue():
        rows.append(
            "| "
            + " | ".join(
                (
                    item.label,
                    item.family,
                    item.parameters,
                    item.source,
                    item.justification,
                )
            )
            + " |"
        )
    if len(rows) < 3:
        raise ValueError("Le catalogue des priors est vide.")
    return "\n".join(rows)


def render_numpyro_dag(path: Path, n_days: int = 4) -> bool:
    """Écrit le DAG si Graphviz est disponible. Renvoie False sinon."""
    try:
        import graphviz  # type: ignore[import-not-found]  # noqa: F401
        from numpyro.render import render_model
    except ImportError:
        return False
    structure = load_age_structure(2)
    settings = load_inference_config()
    priors = load_priors_config()
    kwargs = model_kwargs(
        structure,
        n_days,
        settings,
        fixed_specificity=priors.test_specificity,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_model(
            seirh_model,
            model_kwargs=kwargs,
            render_distributions=True,
            filename=str(path.with_suffix("")),
        )
    except Exception:
        return False
    return (
        path.with_suffix(".pdf").is_file()
        or path.with_suffix(".svg").is_file()
        or path.is_file()
    )


def write_model_notes(directory: Path) -> list[Path]:
    """Écrit les deux schémas Mermaid, le tableau des priors, et le DAG si possible."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for kind in ("simulator", "inference"):
        path = directory / f"compartiments-{kind}.mmd"
        path.write_text(compartment_mermaid(kind) + "\n", encoding="utf-8")
        written.append(path)
    table = directory / "priors.md"
    table.write_text(prior_table_markdown() + "\n", encoding="utf-8")
    written.append(table)
    dag = directory / "dag-numpyro"
    if render_numpyro_dag(dag):
        written.append(dag)
    return written
