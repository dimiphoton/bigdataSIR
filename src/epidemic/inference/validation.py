"""Validation du modèle. Ce module a le droit de lancer le simulateur.

Il n'en garde que les séries publiées. `.truth.json` et l'état interne
ne sont pas lus par l'ajustement.
"""

from typing import Any, cast

import jax
import numpy as np
import pandas as pd
from numpyro.infer import Predictive

from epidemic.config import InferenceConfig, load_inference_config, resolve_path
from epidemic.inference.baselines import exponential_forecast, persistence_forecast
from epidemic.inference.fit import FitResult, configured, fit_observations
from epidemic.inference.metrics import (
    FRENCH_STATUS,
    classify_width_ratio,
    coverage_rate,
    crps_ensemble,
    hdi_width,
    interval_covers,
    log_score,
)
from epidemic.inference.predict import posterior_draws, posterior_predictive
from epidemic.model.age_structure import load_age_structure
from epidemic.model.seirh import (
    OBSERVED_SITES,
    TIME_SERIES_SITES,
    model_kwargs,
    seirh_model,
    specificity_from_config,
)
from epidemic.series import ObservationSeries, observations_from_frame
from epidemic.simulator.engine import RealitySimulator

SERIES = (
    "num_positive",
    "num_hospitalised",
    "num_critical",
    "death_increments",
)
CASES_ONLY = ("num_positive",)
WITH_HOSPITAL = (
    "num_positive",
    "num_hospitalised",
    "num_critical",
    "death_increments",
)


def load_course_series(max_days: int | None = None) -> ObservationSeries:
    """Charge le data.csv du cours, le cas réel de 71 jours."""
    path = resolve_path(load_inference_config().course_data)
    frame = pd.read_csv(path)
    series = observations_from_frame(frame)
    if max_days is None or max_days >= series.n_days:
        return series
    return series.cut(max_days)


def simulate_public_series(level: int, days: int, seed: int) -> ObservationSeries:
    """Fait tourner la ville et ne renvoie que les colonnes publiques."""
    simulator = RealitySimulator.from_configs(seed=seed, level=level)
    simulator.run(days)
    return observations_from_frame(simulator.observations())


def prior_predictive_arrays(
    n_days: int,
    num_samples: int,
    seed: int,
    settings: InferenceConfig | None = None,
) -> dict[str, np.ndarray]:
    """Tirages de la loi a priori, avant d'avoir vu les données."""
    chosen = settings or load_inference_config()
    structure = load_age_structure(chosen.age_groups)
    kwargs = model_kwargs(
        structure,
        n_days,
        chosen,
        None,
        fixed_specificity=specificity_from_config(),
    )
    predictive = Predictive(seirh_model, num_samples=num_samples)
    draws = predictive(jax.random.PRNGKey(seed), **kwargs)
    return {name: np.asarray(value) for name, value in draws.items()}


def assert_prior_plausible(draws: dict[str, np.ndarray], population: float) -> None:
    """Les trajectoires a priori restent des effectifs finis, dans la population."""
    for name in ("expected_confirmed", "hospitalized", "compartment_total"):
        values = draws[name]
        if not np.isfinite(values).all():
            raise AssertionError(f"{name} a priori n'est pas fini.")
    totals = draws["compartment_total"]
    if np.max(np.abs(totals - population)) > 5.0:
        raise AssertionError("La population a priori n'est pas conservée.")
    if np.min(draws["hospitalized"]) < -1.0:
        raise AssertionError("Des hospitalisations a priori sont négatives.")


def draw_prior_dataset(
    n_days: int,
    seed: int,
    settings: InferenceConfig | None = None,
) -> tuple[dict[str, np.ndarray], ObservationSeries]:
    """Un jeu simulé par le modèle lui-même, avec les vrais paramètres du tirage."""
    draws = prior_predictive_arrays(n_days, 1, seed, settings)
    series = ObservationSeries(
        num_positive=np.asarray(draws["num_positive"][0], dtype=float),
        num_tested=np.asarray(draws["num_tested"][0], dtype=float),
        num_hospitalised=np.asarray(draws["num_hospitalised"][0], dtype=float),
        num_critical=np.asarray(draws["num_critical"][0], dtype=float),
        death_increments=np.asarray(draws["death_increments"][0], dtype=float),
    )
    hidden = set(OBSERVED_SITES) | set(TIME_SERIES_SITES)
    truth = {
        name: np.asarray(value)[0]
        for name, value in draws.items()
        if name not in hidden
    }
    return truth, series


def scalar_coverage(
    result: FitResult,
    truth: dict[str, np.ndarray],
    level: float = 0.9,
) -> float:
    """Part des scalaires dont l'intervalle a posteriori couvre la valeur connue."""
    truths: dict[str, float] = {}
    samples: dict[str, np.ndarray] = {}
    for name, value in truth.items():
        array = np.asarray(value)
        posterior = cast(Any, result.idata).posterior
        if array.size != 1 or name not in posterior:
            continue
        drawn = posterior_draws(result.idata, name)
        if drawn is None:
            continue
        truths[name] = float(array)
        samples[name] = drawn.reshape(-1)
    return coverage_rate(truths, samples, level)


def rolling_backtest(
    series: ObservationSeries,
    origins: list[int],
    settings: InferenceConfig | None = None,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Ajuste jusqu'au jour t, prévoit 7, 14 et 21 jours, et score la prévision."""
    chosen = configured(
        settings or load_inference_config(), save_plots=False, strict=False
    )
    structure = load_age_structure(chosen.age_groups)
    rows: list[dict[str, Any]] = []
    for origin in origins:
        if origin < 7 or origin >= series.n_days:
            continue
        horizon = min(max(chosen.horizons), series.n_days - origin)
        fitted = fit_observations(
            series.cut(origin),
            chosen,
            seed=seed,
            write=False,
            show_progress=False,
        )
        predictive = posterior_predictive(
            fitted.idata,
            structure,
            chosen,
            n_future=horizon,
            seed=seed + origin,
        )
        for step in chosen.horizons:
            if step > horizon:
                continue
            index = origin + step - 1
            for name in SERIES:
                samples = np.asarray(getattr(predictive, name)[:, index], dtype=float)
                truth = float(getattr(series, name)[index])
                history = np.asarray(getattr(series, name)[:origin], dtype=float)
                rows.append(
                    {
                        "origin": origin,
                        "horizon": step,
                        "series": name,
                        "crps": crps_ensemble(samples, truth),
                        "covered_50": interval_covers(samples, truth, 0.5),
                        "covered_90": interval_covers(samples, truth, 0.9),
                        "log_score": log_score(samples, truth),
                        "persistence_abs_error": abs(
                            float(persistence_forecast(history, step)[step - 1]) - truth
                        ),
                        "exponential_abs_error": abs(
                            float(exponential_forecast(history, step)[step - 1]) - truth
                        ),
                    }
                )
    return pd.DataFrame(rows)


def learning_curve(
    series: ObservationSeries,
    settings: InferenceConfig | None = None,
    *,
    seed: int = 0,
    num_prior: int = 200,
) -> pd.DataFrame:
    """Largeur des intervalles à chaque bulletin, rapportée à la largeur a priori."""
    chosen = configured(
        settings or load_inference_config(), save_plots=False, strict=False
    )
    prior = prior_predictive_arrays(series.n_days, num_prior, seed, chosen)
    rows: list[dict[str, Any]] = []
    for day in chosen.learning_curve_days:
        if day > series.n_days:
            continue
        fitted = fit_observations(series.cut(day), chosen, seed=seed, write=False)
        for name, prior_values in prior.items():
            if name in OBSERVED_SITES or name in TIME_SERIES_SITES:
                continue
            posterior = posterior_draws(fitted.idata, name)
            if posterior is None:
                continue
            prior_width = _mean_width(np.asarray(prior_values))
            posterior_width = _mean_width(posterior)
            ratio = posterior_width / max(prior_width, 1e-8)
            rows.append(
                {
                    "day": day,
                    "parameter": name,
                    "prior_width": prior_width,
                    "posterior_width": posterior_width,
                    "width_ratio": ratio,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["lecture"] = [
        FRENCH_STATUS[
            classify_width_ratio(
                float(ratio),
                chosen.identifiability.well_below,
                chosen.identifiability.partial_below,
            )
        ]
        for ratio in table["width_ratio"]
    ]
    return table


def ablation_widths(
    series: ObservationSeries,
    settings: InferenceConfig | None = None,
    *,
    seed: int = 0,
    parameters: tuple[str, ...] = ("p_test", "beta0", "symptomatic_fraction"),
) -> dict[str, dict[str, float]]:
    """Largeurs après les positifs, puis l'hôpital, puis le taux de positivité."""
    chosen = configured(
        settings or load_inference_config(), save_plots=False, strict=False
    )
    stages = {
        "cases_only": CASES_ONLY,
        "with_hospital": WITH_HOSPITAL,
        "with_positivity": OBSERVED_SITES,
    }
    widths: dict[str, dict[str, float]] = {}
    for label, included in stages.items():
        fitted = fit_observations(
            series,
            chosen,
            seed=seed,
            write=False,
            included_series=included,
        )
        widths[label] = {}
        for name in parameters:
            drawn = posterior_draws(fitted.idata, name)
            if drawn is None:
                continue
            widths[label][name] = _mean_width(drawn)
    return widths


def _mean_width(samples: np.ndarray, level: float = 0.9) -> float:
    values = np.asarray(samples, dtype=float)
    if values.ndim <= 1 or values.shape[-1] == 1:
        return hdi_width(values.reshape(-1), level)
    widths = [hdi_width(values[..., index], level) for index in range(values.shape[-1])]
    return float(np.mean(widths))
