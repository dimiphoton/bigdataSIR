"""Lit un InferenceData et en fait des tirages prêts à projeter."""

import numpy as np

from epidemic.config import InferenceConfig, load_inference_config
from epidemic.forecast.project import ParameterDraws, subsample
from epidemic.inference.predict import posterior_draws


def draws_from_idata(
    idata: object,
    settings: InferenceConfig | None = None,
    *,
    n_draws: int | None = None,
    seed: int = 0,
) -> ParameterDraws:
    """Empile chaînes et tirages. Lève une erreur si β ou la gravité manquent."""
    used = settings or load_inference_config()
    beta = _required(idata, "beta_t")
    if beta.ndim != 2:
        raise ValueError("beta_t a la forme (tirages, jours).")
    n = beta.shape[0]
    p_hosp = _grouped(idata, "p_hosp", n)
    shock = posterior_draws(idata, "infection_shock")
    draws = ParameterDraws(
        beta_t=beta,
        beta_sigma=_vector_or(idata, "beta_sigma", n, 0.0),
        incubation_days=_vector(idata, "incubation_days", n),
        symptomatic_days=_vector(idata, "symptomatic_days", n),
        hospital_days=_vector(idata, "hospital_days", n),
        icu_days=_vector(idata, "icu_days", n),
        symptomatic_fraction=_vector(idata, "symptomatic_fraction", n),
        asymptomatic_infectiousness=_vector(idata, "asymptomatic_infectiousness", n),
        sensitivity=_vector(idata, "sensitivity", n),
        specificity=_vector_or(idata, "specificity", n, 1.0),
        p_test=_testing_rate(idata, n, beta.shape[1]),
        flu_mean=_vector_or(idata, "flu_mean", n, 300.0),
        e0=_vector(idata, "e0", n),
        i0_share=_vector(idata, "i0_share", n),
        p_hosp=p_hosp,
        p_icu=_grouped(idata, "p_icu", n),
        p_death_ward=_grouped(idata, "p_death_ward", n),
        p_death_icu=_grouped(idata, "p_death_icu", n),
        process_sigma=_vector_or(idata, "process_sigma", n, 0.0),
        concentration=_vector_or(
            idata, "concentration_positive", n, used.fixed_concentration
        ),
        infection_shock=None if shock is None else np.asarray(shock, dtype=float),
    )
    if n_draws is not None:
        return subsample(draws, n_draws, seed)
    return draws


def _required(idata: object, name: str) -> np.ndarray:
    values = posterior_draws(idata, name)
    if values is None:
        raise ValueError(f"Le posterior ne contient pas {name}.")
    return np.asarray(values, dtype=float)


def _vector(idata: object, name: str, n_draws: int) -> np.ndarray:
    values = _required(idata, name)
    flat = values.reshape(n_draws, -1)
    if flat.shape[1] != 1:
        raise ValueError(f"{name} n'est pas un scalaire par tirage.")
    return flat[:, 0]


def _vector_or(idata: object, name: str, n_draws: int, default: float) -> np.ndarray:
    if posterior_draws(idata, name) is None:
        return np.full(n_draws, default, dtype=float)
    return _vector(idata, name, n_draws)


def _grouped(idata: object, name: str, n_draws: int) -> np.ndarray:
    values = _required(idata, name)
    if values.ndim != 2 or values.shape[0] != n_draws:
        raise ValueError(f"{name} a la forme (tirages, groupes).")
    return values


def _testing_rate(idata: object, n_draws: int, n_past: int) -> np.ndarray:
    series = posterior_draws(idata, "p_test_t")
    if series is not None:
        array = np.asarray(series, dtype=float)
        if array.shape[0] != n_draws:
            raise ValueError("p_test_t n'a pas un tirage par ligne.")
        if array.ndim == 1:
            return array
        if array.shape[1] < 1:
            raise ValueError("p_test_t ne contient aucun jour.")
        if array.shape[1] != n_past:
            raise ValueError("p_test_t ne couvre pas les mêmes jours que beta_t.")
        return array[:, -1]
    return _vector(idata, "p_test", n_draws)
