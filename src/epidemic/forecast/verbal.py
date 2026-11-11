"""Échelle verbale des probabilités, celle du GIEC, lue dans le YAML."""

from epidemic.config import VerbalBand, load_reporting_config


def verbal_probability(
    probability: float,
    bands: list[VerbalBand] | None = None,
) -> str:
    """Traduit une probabilité en formulation verbale. Le chiffre reste à côté."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Une probabilité est entre 0 et 1.")
    chosen = load_reporting_config().verbal_scale if bands is None else bands
    for band in chosen:
        if probability >= band.minimum:
            return band.label
    raise ValueError("L'échelle verbale ne couvre pas cette probabilité.")
