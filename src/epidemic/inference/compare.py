"""Comparaison LOO et WAIC. Les deux demandent un ajustement NUTS."""

import arviz as az
import pandas as pd


def compare_criteria(fits: dict[str, az.InferenceData]) -> dict[str, pd.DataFrame]:
    """Compare des modèles déjà ajustés. Échoue si la log-vraisemblance manque."""
    if len(fits) < 2:
        raise ValueError("Il faut au moins deux ajustements à comparer.")
    for name, idata in fits.items():
        if "log_likelihood" not in idata.groups():
            raise ValueError(
                f"{name} n'a pas de log_likelihood. "
                "Relancez cet ajustement avec method=nuts."
            )
    return {
        "loo": az.compare(fits, ic="loo"),
        "waic": az.compare(fits, ic="waic"),
    }
