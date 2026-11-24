"""Prolonge une loi a posteriori au-delà des jours déjà ajustés.

La dynamique est rejouée à partir des paramètres tirés. Les semaines futures
reçoivent de nouveaux chocs : le modèle ne revoit pas les données futures.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from epidemic.config import InferenceConfig
from epidemic.model.age_structure import AgeStructure
from epidemic.model.dynamics import IA, ICU, IS, E, H, initial_state, run_dynamics
from epidemic.model.observation import expected_observations
from epidemic.model.seirh import HOSPITAL_LOG_SCALE


@dataclass(frozen=True)
class PredictiveDraws:
    """Tirages prédictifs : stocks latents et observations bruitées."""

    num_positive: np.ndarray
    num_tested: np.ndarray
    num_hospitalised: np.ndarray
    num_critical: np.ndarray
    death_increments: np.ndarray
    new_infections: np.ndarray
    expected_confirmed: np.ndarray
    false_negatives: np.ndarray
    prevalence: np.ndarray
    hospitalized: np.ndarray
    critical: np.ndarray


def extend_weekly(
    past: jax.Array, sigma: jax.Array, future_shock: jax.Array
) -> jax.Array:
    """Prolonge une marche hebdomadaire. sigma = 0 ne change rien."""
    if future_shock.shape[0] == 0:
        return past
    n_past = past.shape[0]

    def body(
        carry: tuple[jax.Array, jax.Array], step: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        current, previous_week = carry
        day_index = n_past + step
        week = day_index // 7
        updated = jnp.where(
            week != previous_week,
            current * jnp.exp(sigma * future_shock[step]),
            current,
        )
        return (updated, week), updated

    start = (past[-1], jnp.asarray((n_past - 1) // 7, dtype=day_dtype()))
    _carry, series = jax.lax.scan(body, start, jnp.arange(future_shock.shape[0]))
    return jnp.concatenate([past, series])


def extend_probability(
    past: jax.Array, sigma: jax.Array, future_shock: jax.Array
) -> jax.Array:
    """Même calendrier, choc ajouté au logit au-dessus de 50 %."""
    if future_shock.shape[0] == 0:
        return past
    n_past = past.shape[0]
    unit0 = jnp.clip((past[-1] - 0.5) / 0.5, 1e-4, 1.0 - 1e-4)

    def body(
        carry: tuple[jax.Array, jax.Array], step: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        unit, previous_week = carry
        day_index = n_past + step
        week = day_index // 7
        logits = jnp.log(unit) - jnp.log1p(-unit)
        moved = jax.nn.sigmoid(logits + sigma * future_shock[step])
        updated = jnp.where(week != previous_week, moved, unit)
        return (updated, week), 0.5 + 0.5 * updated

    start = (unit0, jnp.asarray((n_past - 1) // 7, dtype=day_dtype()))
    _carry, series = jax.lax.scan(body, start, jnp.arange(future_shock.shape[0]))
    return jnp.concatenate([past, series])


def extend_piecewise(
    past: jax.Array,
    sigma: jax.Array,
    future_shock: jax.Array,
    breakpoints: jax.Array,
) -> jax.Array:
    """Garde le palier en cours, et saute aux dates de changement futures."""
    if future_shock.shape[0] == 0:
        return past
    n_past = past.shape[0]

    def body(current: jax.Array, step: jax.Array) -> tuple[jax.Array, jax.Array]:
        day_number = n_past + step + 1
        jump = jnp.any(breakpoints == day_number)
        updated = jnp.where(
            jump, current * jnp.exp(sigma * future_shock[step]), current
        )
        return updated, updated

    _final, series = jax.lax.scan(body, past[-1], jnp.arange(future_shock.shape[0]))
    return jnp.concatenate([past, series])


def day_dtype() -> jnp.dtype:
    return jnp.int32


def posterior_draws(idata: object, name: str) -> np.ndarray | None:
    """Empile chaînes et tirages. None si le site n'est pas dans le posterior."""
    posterior = getattr(idata, "posterior", None)
    if posterior is None or name not in posterior:
        return None
    values = np.asarray(posterior[name])
    if values.ndim < 2:
        raise ValueError(f"{name} n'a pas les dimensions chaîne et tirage.")
    return values.reshape((-1, *values.shape[2:]))


def posterior_predictive(
    idata: object,
    structure: AgeStructure,
    settings: InferenceConfig,
    *,
    n_future: int,
    seed: int,
) -> PredictiveDraws:
    """Simule le passé rejoué et n_future jours de plus, avec le bruit d'observation."""
    if n_future < 0:
        raise ValueError("n_future doit être positif ou nul.")
    beta = _required(idata, "beta_t")
    n_draws, n_past = beta.shape
    sigma = _optional_vector(idata, "beta_sigma", n_draws)
    p_test = _required(idata, "p_test_t")
    p_sigma = _optional_vector(idata, "p_test_sigma", n_draws)
    key = jax.random.PRNGKey(seed)
    beta_key, test_key, shock_key, noise_key = jax.random.split(key, 4)
    future_beta = _future_noise(beta_key, n_draws, n_future)
    future_test = _future_noise(test_key, n_draws, n_future)
    beta_full, p_test_full = _extend_paths(
        beta, sigma, future_beta, p_test, p_sigma, future_test, settings
    )
    shocks = _shocks(idata, shock_key, n_draws, n_past, n_future, structure.n_groups)
    latent = _latent_paths(
        idata,
        structure,
        beta_full,
        p_test_full,
        shocks,
        _optional_vector(idata, "process_sigma", n_draws),
    )
    return _with_noise(latent, idata, settings, n_draws, noise_key)


def _required(idata: object, name: str) -> np.ndarray:
    values = posterior_draws(idata, name)
    if values is None:
        raise ValueError(f"Le posterior ne contient pas {name}.")
    return values


def _optional_vector(idata: object, name: str, n_draws: int) -> np.ndarray:
    values = posterior_draws(idata, name)
    if values is None:
        return np.zeros(n_draws, dtype=float)
    return np.asarray(values, dtype=float).reshape(n_draws)


def _future_noise(key: jax.Array, n_draws: int, n_future: int) -> jax.Array:
    if n_future == 0:
        return jnp.zeros((n_draws, 0))
    return jax.random.normal(key, (n_draws, n_future))


def _extend_paths(
    beta: np.ndarray,
    sigma: np.ndarray,
    future_beta: jax.Array,
    p_test: np.ndarray,
    p_sigma: np.ndarray,
    future_test: jax.Array,
    settings: InferenceConfig,
) -> tuple[jax.Array, jax.Array]:
    breakpoints = jnp.asarray(settings.beta_breakpoints, dtype=jnp.int32)

    def one_beta(past: jax.Array, scale: jax.Array, shock: jax.Array) -> jax.Array:
        # Même borne que dans le modèle : le tirage brut peut dépasser le clip.
        if settings.beta_mode == "weekly_rw":
            return extend_weekly(past, jnp.clip(scale, 0.0, 1.0), shock)
        return extend_piecewise(past, jnp.clip(scale, 0.0, 2.0), shock, breakpoints)

    beta_full = jax.vmap(one_beta)(jnp.asarray(beta), jnp.asarray(sigma), future_beta)
    p_full = jax.vmap(extend_probability)(
        jnp.asarray(p_test), jnp.asarray(p_sigma), future_test
    )
    return beta_full, p_full


def _shocks(
    idata: object,
    key: jax.Array,
    n_draws: int,
    n_past: int,
    n_future: int,
    n_groups: int,
) -> jax.Array:
    past = posterior_draws(idata, "infection_shock")
    if past is None:
        past_array = jnp.zeros((n_draws, n_past, n_groups))
    else:
        past_array = jnp.asarray(past)
    if n_future == 0:
        return past_array
    future = jax.random.normal(key, (n_draws, n_future, n_groups))
    return jnp.concatenate([past_array, future], axis=1)


def _latent_paths(
    idata: object,
    structure: AgeStructure,
    beta: jax.Array,
    p_test: jax.Array,
    shocks: jax.Array,
    process_sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    population = jnp.asarray(structure.populations, dtype=jnp.float32)
    contacts = jnp.asarray(structure.contacts, dtype=jnp.float32)

    def one(
        beta_t: jax.Array,
        p_test_t: jax.Array,
        shock_t: jax.Array,
        sigma_proc: jax.Array,
        incubation: jax.Array,
        symptomatic_days: jax.Array,
        hospital_days: jax.Array,
        icu_days: jax.Array,
        symptomatic_fraction: jax.Array,
        infectiousness: jax.Array,
        sensitivity: jax.Array,
        specificity: jax.Array,
        flu_mean: jax.Array,
        e0: jax.Array,
        i0_share: jax.Array,
        p_hosp: jax.Array,
        p_icu: jax.Array,
        p_death_ward: jax.Array,
        p_death_icu: jax.Array,
    ) -> tuple[jax.Array, ...]:
        state0 = initial_state(population, e0, e0 * i0_share, symptomatic_fraction)
        trajectory = run_dynamics(
            state0,
            beta_t,
            shock_t,
            population=population,
            contacts=contacts,
            sigma=1.0 / incubation,
            gamma=1.0 / symptomatic_days,
            gamma_h=1.0 / hospital_days,
            gamma_icu=1.0 / icu_days,
            symptomatic_fraction=symptomatic_fraction,
            p_hosp=p_hosp,
            p_icu=p_icu,
            p_death_ward=p_death_ward,
            p_death_icu=p_death_icu,
            asymptomatic_infectiousness=infectiousness,
            process_sigma=sigma_proc,
        )
        positive, tested, false_negatives = expected_observations(
            trajectory["new_symptomatic"],
            trajectory["new_hospital"],
            p_test_t,
            sensitivity,
            specificity,
            flu_mean,
        )
        state = trajectory["state"]
        hospitalized = state[:, :, H].sum(axis=1)
        critical = state[:, :, ICU].sum(axis=1)
        infected = (
            state[:, :, E]
            + state[:, :, IA]
            + state[:, :, IS]
            + state[:, :, H]
            + state[:, :, ICU]
        ).sum(axis=1)
        return (
            positive,
            tested,
            false_negatives,
            hospitalized,
            critical,
            trajectory["new_deaths"].sum(axis=1),
            trajectory["new_infections"].sum(axis=1),
            infected / jnp.sum(population),
        )

    outputs = jax.vmap(one)(
        beta,
        p_test,
        shocks,
        jnp.asarray(process_sigma),
        *[
            _vector(idata, name)
            for name in (
                "incubation_days",
                "symptomatic_days",
                "hospital_days",
                "icu_days",
                "symptomatic_fraction",
                "asymptomatic_infectiousness",
                "sensitivity",
                "specificity",
                "flu_mean",
                "e0",
                "i0_share",
            )
        ],
        *[
            _vector(idata, name)
            for name in (
                "p_hosp",
                "p_icu",
                "p_death_ward",
                "p_death_icu",
            )
        ],
    )
    keys = (
        "expected_confirmed",
        "expected_tested",
        "false_negatives",
        "hospitalized",
        "critical",
        "death_increments",
        "new_infections",
        "prevalence",
    )
    return {key: np.asarray(value) for key, value in zip(keys, outputs, strict=True)}


def _vector(idata: object, name: str) -> jax.Array:
    return jnp.asarray(_required(idata, name))


def _with_noise(
    latent: dict[str, np.ndarray],
    idata: object,
    settings: InferenceConfig,
    n_draws: int,
    key: jax.Array,
) -> PredictiveDraws:
    positive_key, tested_key, hospital_key, critical_key, death_key = jax.random.split(
        key, 5
    )
    conc_positive = _concentration(idata, "concentration_positive", n_draws, settings)
    conc_tested = _concentration(idata, "concentration_tested", n_draws, settings)
    conc_hospital = _concentration(idata, "concentration_hospital", n_draws, settings)
    conc_critical = _concentration(idata, "concentration_critical", n_draws, settings)
    conc_deaths = _concentration(idata, "concentration_deaths", n_draws, settings)
    tested = _sample_nb(tested_key, latent["expected_tested"], conc_tested)
    if settings.likelihood.cases == "negative_binomial":
        positive = _sample_nb(positive_key, latent["expected_confirmed"], conc_positive)
    else:
        positive = _sample_beta_binomial(
            positive_key,
            latent["expected_confirmed"],
            latent["expected_tested"],
            conc_positive,
            tested,
        )
    if settings.likelihood.hospital == "negative_binomial":
        hospital = _sample_nb(hospital_key, latent["hospitalized"], conc_hospital)
        critical = _sample_nb(critical_key, latent["critical"], conc_critical)
    else:
        hospital = _sample_log1p(hospital_key, latent["hospitalized"])
        critical = _sample_log1p(critical_key, latent["critical"])
    deaths = _sample_nb(death_key, latent["death_increments"], conc_deaths)
    return PredictiveDraws(
        num_positive=positive,
        num_tested=tested,
        num_hospitalised=hospital,
        num_critical=critical,
        death_increments=deaths,
        new_infections=latent["new_infections"],
        expected_confirmed=latent["expected_confirmed"],
        false_negatives=latent["false_negatives"],
        prevalence=latent["prevalence"],
        hospitalized=latent["hospitalized"],
        critical=latent["critical"],
    )


def _concentration(
    idata: object, name: str, n_draws: int, settings: InferenceConfig
) -> np.ndarray:
    values = posterior_draws(idata, name)
    if values is None:
        return np.full(n_draws, settings.fixed_concentration, dtype=float)
    return np.asarray(values, dtype=float).reshape(n_draws)


def _sample_nb(
    key: jax.Array, mean: np.ndarray, concentration: np.ndarray
) -> np.ndarray:
    """Mélange gamma-Poisson : variance = moyenne + moyenne² / concentration."""
    safe_mean = jnp.clip(jnp.asarray(mean), 1e-6, None)
    safe_conc = jnp.broadcast_to(
        jnp.clip(jnp.asarray(concentration)[:, None], 1e-6, None),
        safe_mean.shape,
    )
    rate = jax.random.gamma(key, safe_conc) * (safe_mean / safe_conc)
    return np.asarray(jax.random.poisson(jax.random.fold_in(key, 1), rate))


def _sample_log1p(key: jax.Array, mean: np.ndarray) -> np.ndarray:
    noisy = jax.random.normal(key, np.shape(mean)) * HOSPITAL_LOG_SCALE
    return np.asarray(
        jnp.expm1(jnp.log1p(jnp.clip(jnp.asarray(mean), 0.0, None)) + noisy)
    ).clip(0)


# Niveau du jour d'horizon : cas du jour, lits occupés.
_LEVEL_SERIES = ("num_positive", "num_hospitalised", "num_critical")
# Somme sur la fenêtre : cas, infections réelles, décès.
_SUM_SERIES = ("num_positive", "new_infections", "death_increments")


def projection_tables(
    draws: PredictiveDraws,
    *,
    origin_day: int,
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table hebdomadaire et table journalière de la projection.

    L'origine est le dernier jour observé. L'horizon 7 est le bulletin suivant.
    """
    if origin_day < 1:
        raise ValueError("Le jour d'origine commence à 1.")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("Les horizons doivent être des nombres de jours positifs.")
    n_days = _day_count(draws)
    if origin_day > n_days:
        raise ValueError("La projection ne couvre pas le jour d'origine.")
    daily = _daily_frame(draws, origin_day, n_days)
    weekly_rows: list[dict[str, float | int | str]] = []
    for horizon in horizons:
        target = origin_day + horizon
        if target > n_days:
            raise ValueError(f"L'horizon {horizon} dépasse la projection.")
        target_index = target - 1
        window = slice(origin_day, target_index + 1)
        for name in _LEVEL_SERIES:
            values = _series(draws, name)[:, target_index]
            weekly_rows.append(
                _weekly_row(origin_day, horizon, target, name, "niveau", values)
            )
        for name in _SUM_SERIES:
            values = _series(draws, name)[:, window].sum(axis=1)
            weekly_rows.append(
                _weekly_row(origin_day, horizon, target, name, "somme", values)
            )
    weekly = pd.DataFrame(weekly_rows)
    return weekly, daily


def _day_count(draws: PredictiveDraws) -> int:
    count = int(draws.num_positive.shape[1])
    names = _LEVEL_SERIES + _SUM_SERIES
    for name in names:
        if _series(draws, name).shape[1] != count:
            raise ValueError(f"{name} n'a pas le même nombre de jours.")
    return count


def _series(draws: PredictiveDraws, name: str) -> np.ndarray:
    values = np.asarray(getattr(draws, name), dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"{name} a la forme (tirages, jours).")
    return values


def _daily_frame(draws: PredictiveDraws, origin_day: int, n_days: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    day = np.arange(1, n_days + 1)
    for name in dict.fromkeys(_LEVEL_SERIES + _SUM_SERIES):
        samples = _series(draws, name)
        low90, low50, median, high50, high90 = _quantiles(samples)
        for index in range(n_days):
            rows.append(
                {
                    "day": int(day[index]),
                    "role": "observe" if day[index] <= origin_day else "projection",
                    "series": name,
                    "median": float(median[index]),
                    "low50": float(low50[index]),
                    "high50": float(high50[index]),
                    "low90": float(low90[index]),
                    "high90": float(high90[index]),
                }
            )
    return pd.DataFrame(rows)


def _weekly_row(
    origin_day: int,
    horizon: int,
    target: int,
    series: str,
    kind: str,
    values: np.ndarray,
) -> dict[str, float | int | str]:
    low90, _low50, median, _high50, high90 = _quantiles(values[:, None])
    return {
        "origin_day": origin_day,
        "horizon": horizon,
        "target_day": target,
        "series": series,
        "kind": kind,
        "median": float(median[0]),
        "low90": float(low90[0]),
        "high90": float(high90[0]),
    }


def _quantiles(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bandes à 50 % et 90 %, plus la médiane. Axe 0 : les tirages."""
    low90, low50, median, high50, high90 = np.quantile(
        samples, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0
    )
    return low90, low50, median, high50, high90


def _sample_beta_binomial(
    key: jax.Array,
    positive_mean: np.ndarray,
    tested_mean: np.ndarray,
    concentration: np.ndarray,
    tested: np.ndarray,
) -> np.ndarray:
    ratio = np.clip(positive_mean / np.clip(tested_mean, 1e-6, None), 1e-4, 1.0 - 1e-4)
    alpha = ratio * concentration[:, None]
    beta = (1.0 - ratio) * concentration[:, None]
    probabilities = np.asarray(
        jax.random.beta(key, jnp.asarray(alpha), jnp.asarray(beta))
    )
    return np.asarray(
        jax.random.binomial(
            jax.random.fold_in(key, 1),
            jnp.asarray(tested, dtype=jnp.int32),
            jnp.asarray(probabilities),
        )
    )
