"""Ajustement NUTS ou SVI, puis sauvegarde ArviZ."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, log_likelihood
from numpyro.infer.autoguide import AutoNormal

from epidemic.config import InferenceConfig, load_inference_config, resolve_path
from epidemic.inference.diagnostics import (
    DiagnosticsError,
    DiagnosticsReport,
    check_diagnostics,
    parameter_names,
)
from epidemic.inference.plots import save_prior_posterior_plot, save_trace_plot
from epidemic.model.age_structure import AgeStructure, load_age_structure
from epidemic.model.seirh import (
    OBSERVED_SITES,
    model_kwargs,
    seirh_model,
    specificity_from_config,
)
from epidemic.series import ObservationSeries

__all__ = ["DiagnosticsError", "FitResult", "fit_observations"]


@dataclass
class FitResult:
    """Loi a posteriori et son diagnostic."""

    idata: az.InferenceData
    settings: InferenceConfig
    n_days: int
    output_path: Path | None
    diagnostics: DiagnosticsReport
    elbo: float | None


def fit_observations(
    observations: ObservationSeries,
    settings: InferenceConfig | None = None,
    *,
    n_days: int | None = None,
    seed: int | None = None,
    output_path: Path | None = None,
    show_progress: bool = False,
    included_series: tuple[str, ...] | None = None,
    write: bool = True,
) -> FitResult:
    """Ajuste le SEIRH étendu sur les séries publiées. N'ouvre aucun fichier secret."""
    chosen = settings or load_inference_config()
    used_days = observations.n_days if n_days is None else n_days
    used_seed = chosen.seed if seed is None else seed
    structure = load_age_structure(chosen.age_groups)
    kwargs = model_kwargs(
        structure,
        used_days,
        chosen,
        observations,
        fixed_specificity=specificity_from_config(),
        included_series=OBSERVED_SITES if included_series is None else included_series,
    )
    key = jax.random.PRNGKey(used_seed)
    if chosen.method == "nuts":
        idata, elbo = _run_nuts(chosen, kwargs, key, show_progress)
    elif chosen.method == "svi":
        idata, elbo = _run_svi(chosen, kwargs, key, show_progress)
    else:
        raise ValueError("La méthode doit être nuts ou svi.")
    destination = _destination(chosen, output_path) if write else None
    if destination is not None:
        _save_idata(idata, destination)
        _save_summary(idata, destination.with_name("summary.csv"))
        if chosen.save_plots:
            _save_figures(
                idata, structure, chosen, kwargs, destination.parent, used_seed
            )
    report = check_diagnostics(
        idata,
        method=chosen.method,
        max_rhat=chosen.diagnostics.max_rhat,
        min_ess=chosen.diagnostics.min_ess,
        max_divergences=chosen.diagnostics.max_divergences,
        strict=chosen.diagnostics.strict,
        elbo=elbo,
    )
    return FitResult(idata, chosen, used_days, destination, report, elbo)


def _run_nuts(
    settings: InferenceConfig,
    kwargs: dict[str, Any],
    key: jax.Array,
    show_progress: bool,
) -> tuple[az.InferenceData, None]:
    nuts = settings.nuts
    kernel = NUTS(seirh_model, target_accept_prob=nuts.target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=nuts.num_warmup,
        num_samples=nuts.num_samples,
        num_chains=nuts.num_chains,
        chain_method=nuts.chain_method,
        progress_bar=show_progress,
    )
    mcmc.run(key, **kwargs)
    try:
        idata = az.from_numpyro(mcmc)
    except (ValueError, RuntimeError, KeyError):
        idata = az.from_numpyro(mcmc, log_likelihood=False)
    return idata, None


def _run_svi(
    settings: InferenceConfig,
    kwargs: dict[str, Any],
    key: jax.Array,
    show_progress: bool,
) -> tuple[az.InferenceData, float]:
    guide = AutoNormal(seirh_model)
    optimizer = numpyro.optim.Adam(step_size=settings.svi.learning_rate)
    svi = SVI(seirh_model, guide, optimizer, loss=Trace_ELBO())
    run_key, sample_key, predictive_key = jax.random.split(key, 3)
    result = svi.run(
        run_key,
        settings.svi.num_steps,
        progress_bar=show_progress,
        **kwargs,
    )
    losses = np.asarray(result.losses, dtype=float)
    elbo = float(losses[-1])
    latent = guide.sample_posterior(
        sample_key, result.params, sample_shape=(settings.svi.num_samples,)
    )
    generated = Predictive(seirh_model, posterior_samples=latent)(
        predictive_key, **kwargs
    )
    merged: dict[str, np.ndarray] = {}
    for name, value in {**latent, **generated}.items():
        if name in OBSERVED_SITES:
            continue
        merged[name] = np.asarray(value)
    posterior = {name: values[None, ...] for name, values in merged.items()}
    observed = {
        name: np.asarray(kwargs[name])
        for name in OBSERVED_SITES
        if kwargs.get(name) is not None
    }
    log_lik = _safe_log_likelihood(latent, kwargs)
    if log_lik is None:
        idata = az.from_dict(posterior=posterior, observed_data=observed)
    else:
        idata = az.from_dict(
            posterior=posterior,
            observed_data=observed,
            log_likelihood=log_lik,
        )
    return idata, elbo


def _safe_log_likelihood(
    latent: dict[str, jax.Array], kwargs: dict[str, Any]
) -> dict[str, np.ndarray] | None:
    try:
        pointwise = log_likelihood(seirh_model, latent, **kwargs)
    except (ValueError, RuntimeError, TypeError, KeyError):
        return None
    return {name: np.asarray(value)[None, ...] for name, value in pointwise.items()}


def _destination(settings: InferenceConfig, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return resolve_path(settings.output_dir) / "posterior.nc"


def _save_idata(idata: az.InferenceData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for engine in (None, "h5netcdf", "scipy", "netcdf4"):
        try:
            if engine is None:
                idata.to_netcdf(str(path))
            else:
                idata.to_netcdf(str(path), engine=engine)
            return
        except (ImportError, ValueError, OSError, KeyError, RuntimeError) as exc:
            errors.append(f"{engine}: {exc}")
    raise RuntimeError("Impossible d'écrire le netCDF.\n" + "\n".join(errors))


def _save_summary(idata: az.InferenceData, path: Path) -> None:
    names = parameter_names(idata)
    if not names:
        return
    summary = az.summary(idata, var_names=names, round_to=4)
    summary.to_csv(path)


def _save_figures(
    idata: az.InferenceData,
    structure: AgeStructure,
    settings: InferenceConfig,
    kwargs: dict[str, Any],
    directory: Path,
    seed: int,
) -> None:
    del structure
    save_trace_plot(idata, directory / "traces.png")
    prior_kwargs = dict(kwargs)
    for name in OBSERVED_SITES:
        prior_kwargs[name] = None
    prior = Predictive(seirh_model, num_samples=200)(
        jax.random.PRNGKey(seed + 1), **prior_kwargs
    )
    arrays = {name: np.asarray(value) for name, value in prior.items()}
    save_prior_posterior_plot(idata, arrays, directory / "prior_posterior.png")


def configured(
    settings: InferenceConfig,
    *,
    method: str | None = None,
    variant: str | None = None,
    beta_mode: str | None = None,
    seed: int | None = None,
    strict: bool | None = None,
    save_plots: bool | None = None,
) -> InferenceConfig:
    """Copie les réglages en ne changeant que les options demandées."""
    updates: dict[str, Any] = {}
    if method is not None:
        updates["method"] = method
    if variant is not None:
        updates["variant"] = variant
    if beta_mode is not None:
        updates["beta_mode"] = beta_mode
    if seed is not None:
        updates["seed"] = seed
    if save_plots is not None:
        updates["save_plots"] = save_plots
    if strict is not None:
        updates["diagnostics"] = settings.diagnostics.model_copy(
            update={"strict": strict}
        )
    if not updates:
        return settings
    return settings.model_copy(update=updates)
