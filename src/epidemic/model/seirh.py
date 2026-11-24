"""Modèle NumPyro SEIRH étendu.

Variante A : seul β(t) varie, les autres paramètres sont fixes et inconnus.
Variante B : le taux de dépistage varie aussi, et la surdispersion est estimée.
Variante C : les nouvelles infections ont un bruit normal d'échelle binomiale.
Le filtre particulaire (PMMH) n'est pas dans cette variante : il ne sert que si
A et B sont mal calibrées, ce qui se mesure après un NUTS complet.

Deux lois pour β(t) : marche aléatoire hebdomadaire, ou palier entre des dates
connues. La comparaison LOO / WAIC se fait sur des ajustements NUTS.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from epidemic.config import InferenceConfig, load_priors_config
from epidemic.model.age_structure import AgeStructure
from epidemic.model.dynamics import (
    IA,
    ICU,
    IS,
    E,
    H,
    S,
    Trajectory,
    initial_state,
    run_dynamics,
)
from epidemic.model.observation import expected_observations
from epidemic.model.priors import (
    AGE_SLOPE_MU,
    AGE_SLOPE_SIGMA,
    BETA_MU,
    BETA_SIGMA,
    CONCENTRATION_MU,
    CONCENTRATION_SIGMA,
    DEATH_ICU_MU_LOC,
    DEATH_WARD_MU_LOC,
    E0_MU,
    E0_SIGMA,
    FLU_MU,
    FLU_SIGMA,
    HOSP_MU_LOC,
    HOSPITAL_MU,
    HOSPITAL_SIGMA,
    ICU_MU,
    ICU_MU_LOC,
    ICU_SIGMA,
    INCUBATION_MU,
    INCUBATION_SIGMA,
    PIECEWISE_BETA_SCALE,
    PROCESS_SIGMA_SCALE,
    SENSITIVITY_ALPHA,
    SENSITIVITY_BETA,
    SEVERITY_MU_SCALE,
    SEVERITY_TAU_SCALE,
    SYMPTOM_ALPHA,
    SYMPTOM_BETA,
    SYMPTOMATIC_MU,
    SYMPTOMATIC_SIGMA,
    WEEKLY_BETA_SCALE,
)
from epidemic.series import ObservationSeries

# Échelle de la loi normale sur log(1 + stock), option hospital: lognormal.
HOSPITAL_LOG_SCALE = 0.15

OBSERVED_SITES = (
    "num_positive",
    "num_tested",
    "num_hospitalised",
    "num_critical",
    "death_increments",
)

TIME_SERIES_SITES = (
    "beta_t",
    "p_test_t",
    "new_infections",
    "new_symptomatic",
    "new_hospital",
    "new_deaths",
    "expected_confirmed",
    "expected_tested",
    "false_negatives",
    "hospitalized",
    "critical",
    "prevalence",
    "underreporting",
    "rt_proxy",
    "infection_shock",
    "compartment_total",
)


def _sample(name: str, distribution: dist.Distribution) -> jax.Array:
    return cast(jax.Array, numpyro.sample(name, distribution))


def _record(name: str, value: jax.Array) -> jax.Array:
    return cast(jax.Array, numpyro.deterministic(name, value))


def _exp_between(raw: jax.Array, low: float, high: float) -> jax.Array:
    """Exp borné. Un tirage extrême du guide reste un nombre fini."""
    return jnp.exp(jnp.clip(raw, jnp.log(low), jnp.log(high)))


def _logit(values: jax.Array) -> jax.Array:
    clipped = jnp.clip(values, 1e-4, 1.0 - 1e-4)
    return jnp.log(clipped) - jnp.log1p(-clipped)


def _observe(
    name: str,
    distribution: dist.Distribution,
    obs: jax.Array | None,
) -> jax.Array:
    if obs is None:
        return _sample(name, distribution)
    return cast(jax.Array, numpyro.sample(name, distribution, obs=obs))


def _hierarchical(name: str, age_score: jax.Array, mu_loc: float) -> jax.Array:
    """Probabilité par groupe. La pente d'âge est positive : non centrée."""
    mu = _sample(f"{name}_mu", dist.Normal(mu_loc, SEVERITY_MU_SCALE))
    log_slope = _sample(f"{name}_log_slope", dist.Normal(AGE_SLOPE_MU, AGE_SLOPE_SIGMA))
    slope = _exp_between(log_slope, 1e-3, 8.0)
    tau = _sample(f"{name}_tau", dist.HalfNormal(SEVERITY_TAU_SCALE))
    tau = jnp.clip(tau, 0.0, 3.0)
    raw = _sample(f"{name}_z", dist.Normal(0.0, 1.0).expand([age_score.shape[0]]))
    logits = jnp.clip(mu + slope * age_score + tau * raw, -12.0, 12.0)
    return _record(name, jax.nn.sigmoid(logits))


def _concentration(name: str, variant: str, fixed: float) -> jax.Array:
    if variant == "B":
        log_value = _sample(
            f"log_concentration_{name}",
            dist.Normal(CONCENTRATION_MU, CONCENTRATION_SIGMA),
        )
        value = _exp_between(log_value, 0.2, 100_000.0)
    else:
        value = jnp.asarray(fixed, dtype=jnp.float32)
    return _record(f"concentration_{name}", value)


def _beta_path(
    n_days: int,
    beta_mode: str,
    breakpoints: tuple[int, ...],
) -> jax.Array:
    log_beta0 = _sample("log_beta0", dist.Normal(BETA_MU, BETA_SIGMA))
    beta0 = _record("beta0", _exp_between(log_beta0, 1e-4, 0.5))
    if beta_mode == "weekly_rw":
        # σ est tiré même sur une seule semaine : la prévision des semaines
        # suivantes s'en sert, avec la loi a posteriori (ou le prior, si la
        # semaine n'a pas bougé).
        sigma = jnp.clip(
            _sample("beta_sigma", dist.HalfNormal(WEEKLY_BETA_SCALE)), 0.0, 1.0
        )
        n_weeks = (n_days + 6) // 7
        if n_weeks == 1:
            log_week = jnp.zeros((1,))
        else:
            shocks = _sample("beta_week_z", dist.Normal(0.0, 1.0).expand([n_weeks - 1]))
            log_week = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(sigma * shocks)])
        log_week = jnp.clip(log_week, -2.0, 2.0)
        week_index = jnp.arange(n_days) // 7
        return _record("beta_t", beta0 * jnp.exp(log_week[week_index]))
    if beta_mode != "piecewise":
        raise ValueError("beta_mode doit être weekly_rw ou piecewise.")
    sigma = jnp.clip(
        _sample("beta_sigma", dist.HalfNormal(PIECEWISE_BETA_SCALE)), 0.0, 2.0
    )
    n_pieces = len(breakpoints) + 1
    if n_pieces == 1:
        levels = jnp.ones((1,))
    else:
        shocks = _sample("beta_piece_z", dist.Normal(0.0, 1.0).expand([n_pieces - 1]))
        # Chaque palier est indépendant. Le premier vaut β₀.
        jumped = jnp.exp(jnp.clip(sigma * shocks, -3.0, 3.0))
        levels = jnp.concatenate([jnp.ones((1,)), jumped])
    starts = jnp.asarray(breakpoints, dtype=jnp.int32)
    day_number = jnp.arange(1, n_days + 1)
    piece = jnp.searchsorted(starts, day_number, side="right")
    return _record("beta_t", beta0 * levels[piece])


def _ptest_path(n_days: int, variant: str) -> jax.Array:
    unit = _sample("p_test_unit", dist.Beta(2.0, 2.0))
    if variant == "B":
        sigma = _sample("p_test_sigma", dist.HalfNormal(0.25))
        n_weeks = (n_days + 6) // 7
        if n_weeks == 1:
            deviation = jnp.zeros((n_days,))
        else:
            shocks = _sample(
                "p_test_week_z", dist.Normal(0.0, 1.0).expand([n_weeks - 1])
            )
            weekly = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(sigma * shocks)])
            deviation = weekly[jnp.arange(n_days) // 7]
        unit = jax.nn.sigmoid(_logit(unit) + deviation)
    # Plancher de 50 %. Variante A : un taux constant, répété chaque jour.
    level = (0.5 + 0.5 * unit) * jnp.ones(n_days)
    p_test = _record("p_test_t", level)
    _record("p_test", p_test[0])
    return p_test


def _infection_shocks(
    n_days: int, n_groups: int, variant: str
) -> tuple[jax.Array, jax.Array]:
    if variant == "C":
        shocks = _sample(
            "infection_shock",
            dist.Normal(0.0, 1.0).expand([n_days, n_groups]),
        )
        sigma = jnp.clip(
            _sample("process_sigma", dist.HalfNormal(PROCESS_SIGMA_SCALE)), 0.0, 5.0
        )
        return shocks, sigma
    zeros = jnp.zeros((n_days, n_groups))
    return zeros, jnp.asarray(0.0, dtype=jnp.float32)


def seirh_model(
    population: jax.Array,
    contacts: jax.Array,
    age_score: jax.Array,
    n_days: int,
    *,
    variant: str,
    beta_mode: str,
    breakpoints: tuple[int, ...],
    fixed_specificity: float,
    estimate_specificity: bool,
    fixed_concentration: float,
    cases_likelihood: str,
    hospital_likelihood: str,
    included_series: tuple[str, ...] = OBSERVED_SITES,
    num_positive: jax.Array | None = None,
    num_tested: jax.Array | None = None,
    num_hospitalised: jax.Array | None = None,
    num_critical: jax.Array | None = None,
    death_increments: jax.Array | None = None,
) -> None:
    """Modèle génératif. ``None`` sur une série : on la simule, on ne l'observe pas."""
    if variant not in ("A", "B", "C"):
        raise ValueError("La variante doit être A, B ou C.")
    beta_t = _beta_path(n_days, beta_mode, breakpoints)
    p_test_t = _ptest_path(n_days, variant)

    incubation = _record(
        "incubation_days",
        _exp_between(
            _sample("log_incubation", dist.Normal(INCUBATION_MU, INCUBATION_SIGMA)),
            0.5,
            30.0,
        ),
    )
    symptomatic_days = _record(
        "symptomatic_days",
        _exp_between(
            _sample("log_symptomatic", dist.Normal(SYMPTOMATIC_MU, SYMPTOMATIC_SIGMA)),
            1.0,
            40.0,
        ),
    )
    hospital_days = _record(
        "hospital_days",
        _exp_between(
            _sample("log_hospital_days", dist.Normal(HOSPITAL_MU, HOSPITAL_SIGMA)),
            1.0,
            60.0,
        ),
    )
    icu_days = _record(
        "icu_days",
        _exp_between(
            _sample("log_icu_days", dist.Normal(ICU_MU, ICU_SIGMA)),
            1.0,
            60.0,
        ),
    )
    symptomatic_fraction = _sample(
        "symptomatic_fraction", dist.Beta(SYMPTOM_ALPHA, SYMPTOM_BETA)
    )
    sensitivity = _sample("sensitivity", dist.Beta(SENSITIVITY_ALPHA, SENSITIVITY_BETA))
    if estimate_specificity:
        specificity = _sample("specificity", dist.Beta(199.0, 1.0))
    else:
        specificity = _record("specificity", jnp.asarray(fixed_specificity))
    asymptomatic_infectiousness = _sample(
        "asymptomatic_infectiousness", dist.Beta(2.0, 2.0)
    )
    flu_mean = _record(
        "flu_mean",
        _exp_between(_sample("log_flu", dist.Normal(FLU_MU, FLU_SIGMA)), 1.0, 50_000.0),
    )
    e0 = _record(
        "e0",
        _exp_between(_sample("log_e0", dist.Normal(E0_MU, E0_SIGMA)), 1.0, 50_000.0),
    )
    i0_share = _sample("i0_share", dist.Beta(2.0, 5.0))
    p_hosp = _hierarchical("p_hosp", age_score, HOSP_MU_LOC)
    p_icu = _hierarchical("p_icu", age_score, ICU_MU_LOC)
    p_death_ward = _hierarchical("p_death_ward", age_score, DEATH_WARD_MU_LOC)
    p_death_icu = _hierarchical("p_death_icu", age_score, DEATH_ICU_MU_LOC)

    state0 = initial_state(population, e0, e0 * i0_share, symptomatic_fraction)
    shocks, process_sigma = _infection_shocks(n_days, population.shape[0], variant)
    trajectory = run_dynamics(
        state0,
        beta_t,
        shocks,
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
        asymptomatic_infectiousness=asymptomatic_infectiousness,
        process_sigma=process_sigma,
    )
    _publish_latent(
        trajectory,
        beta_t,
        population,
        contacts,
        symptomatic_days,
        symptomatic_fraction,
        asymptomatic_infectiousness,
    )
    expected_positive, expected_tested, false_negatives = expected_observations(
        trajectory["new_symptomatic"],
        trajectory["new_hospital"],
        p_test_t,
        sensitivity,
        specificity,
        flu_mean,
    )
    _record("expected_confirmed", expected_positive)
    _record("expected_tested", expected_tested)
    _record("false_negatives", false_negatives)
    _record(
        "underreporting",
        trajectory["new_infections"].sum(axis=1)
        / jnp.clip(expected_positive, 1e-3, None),
    )
    _observe_series(
        expected_positive,
        expected_tested,
        trajectory["state"][:, :, H].sum(axis=1),
        trajectory["state"][:, :, ICU].sum(axis=1),
        trajectory["new_deaths"].sum(axis=1),
        population.sum(),
        variant=variant,
        fixed_concentration=fixed_concentration,
        cases_likelihood=cases_likelihood,
        hospital_likelihood=hospital_likelihood,
        included_series=included_series,
        num_positive=num_positive,
        num_tested=num_tested,
        num_hospitalised=num_hospitalised,
        num_critical=num_critical,
        death_increments=death_increments,
    )


def _publish_latent(
    trajectory: Trajectory,
    beta_t: jax.Array,
    population: jax.Array,
    contacts: jax.Array,
    symptomatic_days: jax.Array,
    symptomatic_fraction: jax.Array,
    asymptomatic_infectiousness: jax.Array,
) -> None:
    state = trajectory["state"]
    _record("new_infections", trajectory["new_infections"])
    _record("new_symptomatic", trajectory["new_symptomatic"])
    _record("new_hospital", trajectory["new_hospital"])
    _record("new_deaths", trajectory["new_deaths"].sum(axis=1))
    hospitalized = state[:, :, H].sum(axis=1)
    critical = state[:, :, ICU].sum(axis=1)
    _record("hospitalized", hospitalized)
    _record("critical", critical)
    infected = (
        state[:, :, E]
        + state[:, :, IA]
        + state[:, :, IS]
        + state[:, :, H]
        + state[:, :, ICU]
    ).sum(axis=1)
    _record("prevalence", infected / jnp.sum(population))
    row_sums = contacts.sum(axis=1)
    mean_contacts = jnp.sum(row_sums * population) / jnp.sum(population)
    infectious_weight = symptomatic_days * (
        symptomatic_fraction
        + (1.0 - symptomatic_fraction) * asymptomatic_infectiousness
    )
    susceptible_fraction = state[:, :, S].sum(axis=1) / jnp.sum(population)
    # Proxy : β × contacts moyens × durée infectieuse pondérée × S/N.
    # Ce n'est pas le rayon spectral de la matrice de prochaine génération.
    _record(
        "rt_proxy", beta_t * mean_contacts * infectious_weight * susceptible_fraction
    )
    # Les compartiments servent à vérifier la conservation dans les tests.
    _record("compartment_total", state.sum(axis=(1, 2)))


def _observe_series(
    expected_positive: jax.Array,
    expected_tested: jax.Array,
    hospitalized: jax.Array,
    critical: jax.Array,
    death_mean: jax.Array,
    population: jax.Array,
    *,
    variant: str,
    fixed_concentration: float,
    cases_likelihood: str,
    hospital_likelihood: str,
    included_series: tuple[str, ...],
    num_positive: jax.Array | None,
    num_tested: jax.Array | None,
    num_hospitalised: jax.Array | None,
    num_critical: jax.Array | None,
    death_increments: jax.Array | None,
) -> None:
    cap = jnp.asarray(population, dtype=expected_positive.dtype)
    conc_positive = _concentration("positive", variant, fixed_concentration)
    conc_tested = _concentration("tested", variant, fixed_concentration)
    conc_hospital = _concentration("hospital", variant, fixed_concentration)
    conc_critical = _concentration("critical", variant, fixed_concentration)
    conc_deaths = _concentration("deaths", variant, fixed_concentration)
    tested_mean = jnp.clip(expected_tested, 1e-4, cap)
    positive_mean = jnp.clip(expected_positive, 1e-4, cap)
    tested_value = None
    if "num_tested" in included_series:
        tested_value = _observe(
            "num_tested",
            dist.NegativeBinomial2(mean=tested_mean, concentration=conc_tested),
            None if num_tested is None else jnp.asarray(num_tested, dtype=jnp.float32),
        )
    if "num_positive" not in included_series:
        pass
    elif cases_likelihood == "negative_binomial":
        _observe(
            "num_positive",
            dist.NegativeBinomial2(mean=positive_mean, concentration=conc_positive),
            None
            if num_positive is None
            else jnp.asarray(num_positive, dtype=jnp.float32),
        )
    elif cases_likelihood == "beta_binomial":
        if tested_value is None:
            raise ValueError("La bêta-binomiale exige aussi la série des tests.")
        # p = positifs attendus / tests attendus, conditionnel aux tests du jour.
        ratio = jnp.clip(positive_mean / tested_mean, 1e-4, 1.0 - 1e-4)
        total = jnp.rint(jnp.clip(tested_value, 0.0, None)).astype(jnp.int32)
        _observe(
            "num_positive",
            dist.BetaBinomial(
                concentration1=ratio * conc_positive,
                concentration0=(1.0 - ratio) * conc_positive,
                total_count=total,
            ),
            None
            if num_positive is None
            else jnp.asarray(num_positive, dtype=jnp.int32),
        )
    else:
        raise ValueError("cases doit être negative_binomial ou beta_binomial.")
    if "num_hospitalised" in included_series:
        _observe_stock(
            "num_hospitalised",
            hospitalized,
            conc_hospital,
            hospital_likelihood,
            num_hospitalised,
            cap,
        )
    if "num_critical" in included_series:
        _observe_stock(
            "num_critical",
            critical,
            conc_critical,
            hospital_likelihood,
            num_critical,
            cap,
        )
    if "death_increments" in included_series:
        _observe(
            "death_increments",
            dist.NegativeBinomial2(
                mean=jnp.clip(death_mean, 1e-4, cap),
                concentration=conc_deaths,
            ),
            None
            if death_increments is None
            else jnp.asarray(death_increments, dtype=jnp.float32),
        )


def _observe_stock(
    name: str,
    mean: jax.Array,
    concentration: jax.Array,
    kind: str,
    obs: jax.Array | None,
    cap: jax.Array,
) -> None:
    if kind == "negative_binomial":
        _observe(
            name,
            dist.NegativeBinomial2(
                mean=jnp.clip(mean, 1e-4, cap),
                concentration=concentration,
            ),
            None if obs is None else jnp.asarray(obs, dtype=jnp.float32),
        )
        return
    if kind != "lognormal":
        raise ValueError("hospital doit être negative_binomial ou lognormal.")
    predicted = jnp.log1p(jnp.clip(mean, 0.0, None))
    observed = None if obs is None else jnp.log1p(jnp.asarray(obs, dtype=jnp.float32))
    _observe(name, dist.Normal(predicted, HOSPITAL_LOG_SCALE), observed)


def model_kwargs(
    structure: AgeStructure,
    n_days: int,
    settings: InferenceConfig,
    observations: ObservationSeries | None = None,
    *,
    fixed_specificity: float,
    included_series: tuple[str, ...] = OBSERVED_SITES,
) -> dict[str, Any]:
    """Arguments du modèle. Les observations, si présentes, sont tronquées à n_days."""
    if n_days < 1:
        raise ValueError("Il faut au moins un jour.")
    observed: dict[str, jax.Array | None] = {
        "num_positive": None,
        "num_tested": None,
        "num_hospitalised": None,
        "num_critical": None,
        "death_increments": None,
    }
    if observations is not None:
        if observations.n_days < n_days:
            raise ValueError("La série observée est plus courte que n_days.")
        cut = observations.cut(n_days)
        observed = {
            "num_positive": jnp.asarray(cut.num_positive, dtype=jnp.float32),
            "num_tested": jnp.asarray(cut.num_tested, dtype=jnp.float32),
            "num_hospitalised": jnp.asarray(cut.num_hospitalised, dtype=jnp.float32),
            "num_critical": jnp.asarray(cut.num_critical, dtype=jnp.float32),
            "death_increments": jnp.asarray(cut.death_increments, dtype=jnp.float32),
        }
    return {
        "population": jnp.asarray(structure.populations, dtype=jnp.float32),
        "contacts": jnp.asarray(structure.contacts, dtype=jnp.float32),
        "age_score": jnp.asarray(structure.age_score, dtype=jnp.float32),
        "n_days": n_days,
        "variant": settings.variant,
        "beta_mode": settings.beta_mode,
        "breakpoints": tuple(settings.beta_breakpoints),
        "fixed_specificity": fixed_specificity,
        "estimate_specificity": settings.estimate_specificity,
        "fixed_concentration": settings.fixed_concentration,
        "cases_likelihood": settings.likelihood.cases,
        "hospital_likelihood": settings.likelihood.hospital,
        "included_series": included_series,
        **observed,
    }


def specificity_from_config() -> float:
    """Spécificité fixée : celle des priors, 1 dans les consignes."""
    return float(load_priors_config().test_specificity)
