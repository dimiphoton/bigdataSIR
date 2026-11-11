"""Diagnostics MCMC. Un NUTS mal convergé échoue explicitement."""

from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np

from epidemic.model.seirh import OBSERVED_SITES, TIME_SERIES_SITES


class DiagnosticsError(RuntimeError):
    """Les chaînes ne sont pas utilisables pour une décision."""


@dataclass(frozen=True)
class DiagnosticsReport:
    """Résumé chiffré d'un ajustement."""

    ok: bool
    method: str
    max_rhat: float | None
    min_ess: float | None
    divergences: int | None
    message: str


def parameter_names(idata: az.InferenceData) -> list[str]:
    """Paramètres scalaires ou par groupe, sans les séries quotidiennes."""
    skipped = set(TIME_SERIES_SITES) | set(OBSERVED_SITES)
    names = []
    loose: Any = idata
    posterior = loose.posterior
    for name, values in posterior.data_vars.items():
        if str(name) in skipped or "time" in values.dims or values.ndim > 3:
            continue
        names.append(str(name))
    return names


def _finite_values(data: Any, names: list[str]) -> list[np.ndarray]:
    found = []
    for name in names:
        if name not in data:
            continue
        flat = np.asarray(data[name], dtype=float).ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size:
            found.append(finite)
    return found


def check_diagnostics(
    idata: az.InferenceData,
    *,
    method: str,
    max_rhat: float,
    min_ess: float,
    max_divergences: int,
    strict: bool,
    elbo: float | None = None,
) -> DiagnosticsReport:
    """Contrôle R̂, l'ESS et les divergences. SVI n'a pas ces quantités."""
    if method == "svi":
        finite = elbo is not None and _isfinite(elbo)
        message = (
            "SVI terminé. R̂ et l'ESS ne sont pas définis : "
            "utilisez method=nuts pour les diagnostics officiels."
            if finite
            else "L'ELBO du SVI n'est pas fini."
        )
        report = DiagnosticsReport(finite, method, None, None, None, message)
        if strict and not report.ok:
            raise DiagnosticsError(report.message)
        return report

    names = parameter_names(idata)
    if not names:
        raise DiagnosticsError("Le posterior ne contient aucun paramètre à contrôler.")
    rhat_values = _finite_values(az.rhat(idata, var_names=names, method="rank"), names)
    ess_values = _finite_values(az.ess(idata, var_names=names, method="bulk"), names)
    if not rhat_values or not ess_values:
        report = DiagnosticsReport(
            False,
            method,
            None,
            None,
            None,
            "R̂ ou l'ESS n'a pas pu être calculé.",
        )
        if strict:
            raise DiagnosticsError(report.message)
        return report
    worst_rhat = float(max(np.max(item) for item in rhat_values))
    smallest_ess = float(min(np.min(item) for item in ess_values))
    divergences = 0
    if "sample_stats" in idata.groups():
        loose: Any = idata
        sample_stats = loose.sample_stats
        if "diverging" in sample_stats:
            divergences = int(np.asarray(sample_stats["diverging"]).sum())
    ok = (
        worst_rhat < max_rhat
        and smallest_ess >= min_ess
        and divergences <= max_divergences
    )
    message = (
        f"R̂ max = {worst_rhat:.3f} (seuil {max_rhat}), "
        f"ESS min = {smallest_ess:.0f} (seuil {min_ess}), "
        f"divergences = {divergences} (seuil {max_divergences})."
    )
    report = DiagnosticsReport(
        ok, method, worst_rhat, smallest_ess, divergences, message
    )
    if strict and not ok:
        raise DiagnosticsError(message)
    return report


def _isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
