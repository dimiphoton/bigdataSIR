"""MCMC, SVI et diagnostics."""

from epidemic.inference.diagnostics import DiagnosticsError
from epidemic.inference.fit import FitResult, fit_observations

__all__ = ["DiagnosticsError", "FitResult", "fit_observations"]
