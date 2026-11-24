"""Projections : on rejoue la dynamique avec le bruit du processus, puis l'observation.

Les jours déjà publiés gardent les chocs inférés (nuls si la variante n'en a pas).
Les jours futurs reçoivent de nouveaux chocs. Un scénario change la matrice de
contact à partir d'un jour d'effet, et peut échelonner β tirage par tirage.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from epidemic.forecast.noise import (
    expected_counts,
    sample_beta_binomial,
    sample_log1p,
    sample_negative_binomial,
)
from epidemic.inference.predict import extend_weekly
from epidemic.model.age_structure import AgeStructure
from epidemic.model.dynamics import (
    IA,
    ICU,
    IS,
    H,
    S,
    initial_state,
    run_dynamics,
)
from epidemic.model.seirh import HOSPITAL_LOG_SCALE


@dataclass
class ParameterDraws:
    """Tirages a posteriori déjà empilés, une ligne par tirage."""

    beta_t: np.ndarray
    beta_sigma: np.ndarray
    incubation_days: np.ndarray
    symptomatic_days: np.ndarray
    hospital_days: np.ndarray
    icu_days: np.ndarray
    symptomatic_fraction: np.ndarray
    asymptomatic_infectiousness: np.ndarray
    sensitivity: np.ndarray
    specificity: np.ndarray
    p_test: np.ndarray
    flu_mean: np.ndarray
    e0: np.ndarray
    i0_share: np.ndarray
    p_hosp: np.ndarray
    p_icu: np.ndarray
    p_death_ward: np.ndarray
    p_death_icu: np.ndarray
    process_sigma: np.ndarray
    concentration: np.ndarray
    infection_shock: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.beta_t.ndim != 2 or self.beta_t.shape[0] < 1:
            raise ValueError("beta_t a la forme (tirages, jours).")
        if self.p_hosp.ndim != 2 or self.p_hosp.shape[0] != self.n_draws:
            raise ValueError("p_hosp a la forme (tirages, groupes).")
        vectors = (
            self.beta_sigma,
            self.incubation_days,
            self.symptomatic_days,
            self.hospital_days,
            self.icu_days,
            self.symptomatic_fraction,
            self.asymptomatic_infectiousness,
            self.sensitivity,
            self.specificity,
            self.p_test,
            self.flu_mean,
            self.e0,
            self.i0_share,
            self.process_sigma,
            self.concentration,
        )
        for array in vectors:
            if array.shape != (self.n_draws,):
                raise ValueError("Un paramètre scalaire n'a pas une valeur par tirage.")
        for array in (self.p_icu, self.p_death_ward, self.p_death_icu):
            if array.shape != self.p_hosp.shape:
                raise ValueError("Les probabilités de gravité n'ont pas la même forme.")
        if self.infection_shock is not None and self.infection_shock.shape != (
            self.n_draws,
            self.n_past,
            self.n_groups,
        ):
            raise ValueError(
                "infection_shock n'a pas la forme (tirages, jours, groupes)."
            )

    @property
    def n_draws(self) -> int:
        return int(self.beta_t.shape[0])

    @property
    def n_past(self) -> int:
        return int(self.beta_t.shape[1])

    @property
    def n_groups(self) -> int:
        return int(self.p_hosp.shape[1])


@dataclass
class Projection:
    """Une trajectoire par tirage, du jour 1 à la fin de l'horizon."""

    day: np.ndarray
    num_positive: np.ndarray
    num_tested: np.ndarray
    num_hospitalised: np.ndarray
    num_critical: np.ndarray
    death_increments: np.ndarray
    new_infections: np.ndarray
    hospitalized: np.ndarray
    critical: np.ndarray
    susceptible: np.ndarray
    infectious: np.ndarray
    rt: np.ndarray
    beta_effective: np.ndarray
    false_negatives: np.ndarray
    expected_confirmed: np.ndarray


def constant_draws(
    n_draws: int,
    n_days: int,
    n_groups: int,
    *,
    beta: float = 0.05,
    beta_sigma: float = 0.0,
    incubation_days: float = 3.0,
    symptomatic_days: float = 6.0,
    hospital_days: float = 8.0,
    icu_days: float = 10.0,
    symptomatic_fraction: float = 0.6,
    asymptomatic_infectiousness: float = 0.5,
    sensitivity: float = 0.8,
    specificity: float = 1.0,
    p_test: float = 0.7,
    flu_mean: float = 0.0,
    e0: float = 40.0,
    i0_share: float = 0.25,
    p_hosp: float = 0.05,
    p_icu: float = 0.15,
    p_death_ward: float = 0.02,
    p_death_icu: float = 0.2,
    process_sigma: float = 0.0,
    concentration: float = 100.0,
) -> ParameterDraws:
    """Répète les mêmes paramètres. Sert aux essais courts."""
    if n_draws < 1 or n_days < 1 or n_groups < 1:
        raise ValueError("Il faut au moins un tirage, un jour et un groupe.")

    def column(value: float) -> np.ndarray:
        return np.full(n_draws, value, dtype=float)

    def by_group(value: float) -> np.ndarray:
        return np.full((n_draws, n_groups), value, dtype=float)

    return ParameterDraws(
        beta_t=np.full((n_draws, n_days), beta, dtype=float),
        beta_sigma=column(beta_sigma),
        incubation_days=column(incubation_days),
        symptomatic_days=column(symptomatic_days),
        hospital_days=column(hospital_days),
        icu_days=column(icu_days),
        symptomatic_fraction=column(symptomatic_fraction),
        asymptomatic_infectiousness=column(asymptomatic_infectiousness),
        sensitivity=column(sensitivity),
        specificity=column(specificity),
        p_test=column(p_test),
        flu_mean=column(flu_mean),
        e0=column(e0),
        i0_share=column(i0_share),
        p_hosp=by_group(p_hosp),
        p_icu=by_group(p_icu),
        p_death_ward=by_group(p_death_ward),
        p_death_icu=by_group(p_death_icu),
        process_sigma=column(process_sigma),
        concentration=column(concentration),
    )


def subsample(draws: ParameterDraws, n_draws: int, seed: int) -> ParameterDraws:
    """Garde n_draws lignes, sans remise. L'ordre reste celui du tirage."""
    if n_draws < 1:
        raise ValueError("On garde au moins un tirage.")
    if n_draws >= draws.n_draws:
        return draws
    pick = np.sort(
        np.random.default_rng(seed).choice(draws.n_draws, size=n_draws, replace=False)
    )
    shock = None if draws.infection_shock is None else draws.infection_shock[pick]
    return ParameterDraws(
        beta_t=draws.beta_t[pick],
        beta_sigma=draws.beta_sigma[pick],
        incubation_days=draws.incubation_days[pick],
        symptomatic_days=draws.symptomatic_days[pick],
        hospital_days=draws.hospital_days[pick],
        icu_days=draws.icu_days[pick],
        symptomatic_fraction=draws.symptomatic_fraction[pick],
        asymptomatic_infectiousness=draws.asymptomatic_infectiousness[pick],
        sensitivity=draws.sensitivity[pick],
        specificity=draws.specificity[pick],
        p_test=draws.p_test[pick],
        flu_mean=draws.flu_mean[pick],
        e0=draws.e0[pick],
        i0_share=draws.i0_share[pick],
        p_hosp=draws.p_hosp[pick],
        p_icu=draws.p_icu[pick],
        p_death_ward=draws.p_death_ward[pick],
        p_death_icu=draws.p_death_icu[pick],
        process_sigma=draws.process_sigma[pick],
        concentration=draws.concentration[pick],
        infection_shock=shock,
    )


def project_scenarios(
    draws: ParameterDraws,
    structure: AgeStructure,
    matrices: list[np.ndarray],
    *,
    n_days: int,
    switch_index: int,
    scales: np.ndarray,
    seed: int,
    cases_likelihood: str = "negative_binomial",
    hospital_likelihood: str = "negative_binomial",
) -> list[Projection]:
    """Simule chaque matrice après ``switch_index``. Avant, la matrice du modèle.

    ``scales`` a la forme (scénarios, tirages) et multiplie β après le basculement.
    """
    n_scenarios = len(matrices)
    if n_scenarios < 1:
        raise ValueError("Il faut au moins un scénario.")
    if draws.n_groups != structure.n_groups:
        raise ValueError(
            "Les tirages et la structure d'âge n'ont pas les mêmes groupes."
        )
    if n_days < draws.n_past:
        raise ValueError("L'horizon est plus court que les jours déjà tirés.")
    if switch_index < 0 or switch_index > n_days:
        raise ValueError("Le jour d'effet est hors de l'horizon.")
    if scales.shape != (n_scenarios, draws.n_draws):
        raise ValueError("Les échelles ont la forme (scénarios, tirages).")
    for matrix in matrices:
        if matrix.shape != (structure.n_groups, structure.n_groups):
            raise ValueError("Une matrice de contact n'a pas la bonne taille.")

    rng = np.random.default_rng(seed)
    beta = _extend_beta(draws, n_days, rng)
    shocks, sigma = _shocks(draws, n_days, rng)
    population = np.asarray(structure.populations, dtype=float)
    baseline = np.asarray(structure.contacts, dtype=float)
    past, state = _segment(
        draws,
        population,
        beta[:, :switch_index],
        shocks[:, :switch_index],
        sigma,
        _repeat_matrix(baseline, draws.n_draws),
        _initial_states(draws, population) if switch_index > 0 else None,
    )
    future_days = n_days - switch_index
    day = np.arange(1, n_days + 1)
    if future_days == 0:
        if past is None:
            raise ValueError("L'horizon ne contient aucun jour.")
        shared = _with_observation(
            past,
            draws,
            day,
            rng,
            cases_likelihood=cases_likelihood,
            hospital_likelihood=hospital_likelihood,
        )
        return [shared for _ in matrices]
    segments = _future_segments(
        draws,
        population,
        beta[:, switch_index:],
        shocks[:, switch_index:],
        sigma,
        matrices,
        scales,
        state if state is not None else _initial_states(draws, population),
    )
    projections: list[Projection] = []
    for segment, _final_state in segments:
        merged = _concat(past, segment)
        projections.append(
            _with_observation(
                merged,
                draws,
                day,
                rng,
                cases_likelihood=cases_likelihood,
                hospital_likelihood=hospital_likelihood,
            )
        )
    return projections


@dataclass
class Horizon:
    """β, chocs et état au dernier jour déjà observé, prêts pour la suite."""

    beta: np.ndarray
    shocks: np.ndarray
    sigma: np.ndarray
    state: np.ndarray


def prepare_horizon(
    draws: ParameterDraws,
    structure: AgeStructure,
    n_days: int,
    seed: int,
) -> Horizon:
    """Prolonge β et les chocs, et rejoue les jours déjà tirés."""
    if n_days < draws.n_past:
        raise ValueError("L'horizon est plus court que les jours déjà tirés.")
    if draws.n_past < 1:
        raise ValueError("La prévision part d'au moins un jour déjà tiré.")
    rng = np.random.default_rng(seed)
    beta = _extend_beta(draws, n_days, rng)
    shocks, sigma = _shocks(draws, n_days, rng)
    population = np.asarray(structure.populations, dtype=float)
    _past, state = _segment(
        draws,
        population,
        beta[:, : draws.n_past],
        shocks[:, : draws.n_past],
        sigma,
        _repeat_matrix(np.asarray(structure.contacts, dtype=float), draws.n_draws),
        None,
    )
    if state is None:
        raise ValueError("L'état au jour observé n'a pas été calculé.")
    return Horizon(beta=beta, shocks=shocks, sigma=sigma, state=state)


@dataclass
class LatentSegment:
    """Séries latentes d'un tronçon, sommes sur les groupes."""

    new_infections: np.ndarray
    new_symptomatic: np.ndarray
    new_hospital: np.ndarray
    new_deaths: np.ndarray
    hospitalized: np.ndarray
    critical: np.ndarray
    susceptible: np.ndarray
    infectious: np.ndarray
    rt: np.ndarray
    beta_effective: np.ndarray
    beta: np.ndarray


def as_latent_segment(projection: Projection) -> LatentSegment:
    """Reprend les séries latentes déjà portées par une projection.

    ``new_infections`` et ``infectious`` sont copiées. Les flux que la
    projection ne garde pas (symptomatiques, admissions, décès latents, β nu)
    restent à zéro.
    """
    new_infections = np.asarray(projection.new_infections, dtype=float).copy()
    infectious = np.asarray(projection.infectious, dtype=float).copy()
    blank = np.zeros_like(new_infections)

    def copy(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=float).copy()

    return LatentSegment(
        new_infections=new_infections,
        new_symptomatic=blank.copy(),
        new_hospital=blank.copy(),
        new_deaths=blank.copy(),
        hospitalized=copy(projection.hospitalized),
        critical=copy(projection.critical),
        susceptible=copy(projection.susceptible),
        infectious=infectious,
        rt=copy(projection.rt),
        beta_effective=copy(projection.beta_effective),
        beta=blank.copy(),
    )


def step_one_day(
    state: np.ndarray,
    beta: np.ndarray,
    shock: np.ndarray,
    sigma: np.ndarray,
    contacts: np.ndarray,
    draws: ParameterDraws,
    population: np.ndarray,
) -> tuple[np.ndarray, LatentSegment]:
    """Avance tous les tirages d'un jour. ``beta`` a une valeur par tirage."""
    scanned_state, infections, symptomatic, hospital, deaths = _scan(
        state,
        beta[:, None],
        shock[:, None, :],
        sigma,
        contacts,
        draws,
        population,
    )
    segment = _collapse(
        scanned_state,
        infections,
        symptomatic,
        hospital,
        deaths,
        beta[:, None],
        contacts,
        draws,
        population,
    )
    return scanned_state[:, -1], segment


def _extend_beta(
    draws: ParameterDraws, n_days: int, rng: np.random.Generator
) -> np.ndarray:
    if n_days == draws.n_past:
        return np.asarray(draws.beta_t, dtype=float)
    future = rng.normal(size=(draws.n_draws, n_days - draws.n_past))
    extended = jax.vmap(extend_weekly)(
        jnp.asarray(draws.beta_t),
        jnp.asarray(draws.beta_sigma),
        jnp.asarray(future),
    )
    return np.asarray(extended, dtype=float)


def _shocks(
    draws: ParameterDraws, n_days: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Chocs passés connus, chocs futurs neufs. σ = 1 si le posterior n'en a pas."""
    past = (
        np.zeros((draws.n_draws, draws.n_past, draws.n_groups), dtype=float)
        if draws.infection_shock is None
        else np.asarray(draws.infection_shock, dtype=float)
    )
    n_future = n_days - draws.n_past
    if n_future:
        future = rng.normal(size=(draws.n_draws, n_future, draws.n_groups))
        shocks = np.concatenate([past, future], axis=1)
    else:
        shocks = past
    sigma = np.asarray(draws.process_sigma, dtype=float)
    if float(np.max(sigma)) == 0.0:
        sigma = np.ones(draws.n_draws, dtype=float)
    return shocks, sigma


def _initial_states(draws: ParameterDraws, population: np.ndarray) -> np.ndarray:
    pop = jnp.asarray(population)

    def one(exposed: jax.Array, share: jax.Array, fraction: jax.Array) -> jax.Array:
        return initial_state(pop, exposed, exposed * share, fraction)

    states = jax.vmap(one)(
        jnp.asarray(draws.e0),
        jnp.asarray(draws.i0_share),
        jnp.asarray(draws.symptomatic_fraction),
    )
    return np.asarray(states)


def _repeat_matrix(matrix: np.ndarray, n_draws: int) -> np.ndarray:
    return np.broadcast_to(matrix, (n_draws, *matrix.shape)).copy()


def _tile_draws(array: np.ndarray, n_scenarios: int) -> np.ndarray:
    """Répète le bloc des tirages, une fois par scénario."""
    return np.tile(array, (n_scenarios, *([1] * (array.ndim - 1))))


def _segment(
    draws: ParameterDraws,
    population: np.ndarray,
    beta: np.ndarray,
    shocks: np.ndarray,
    sigma: np.ndarray,
    contacts: np.ndarray,
    state0: np.ndarray | None,
) -> tuple[LatentSegment | None, np.ndarray | None]:
    """Rejoue un tronçon. state0 None : on part du germe initial."""
    if beta.shape[1] == 0:
        return None, state0
    if state0 is None:
        state0 = _initial_states(draws, population)
    state, infections, symptomatic, hospital, deaths = _scan(
        state0,
        beta,
        shocks,
        sigma,
        contacts,
        draws,
        population,
    )
    segment = _collapse(
        state,
        infections,
        symptomatic,
        hospital,
        deaths,
        beta,
        contacts,
        draws,
        population,
    )
    return segment, state[:, -1]


def _future_segments(
    draws: ParameterDraws,
    population: np.ndarray,
    beta: np.ndarray,
    shocks: np.ndarray,
    sigma: np.ndarray,
    matrices: list[np.ndarray],
    scales: np.ndarray,
    state0: np.ndarray,
) -> list[tuple[LatentSegment, np.ndarray]]:
    n_scenarios = len(matrices)
    n_draws = draws.n_draws
    beta_batch = _tile_draws(beta, n_scenarios) * scales.reshape(-1)[:, None]
    shock_batch = _tile_draws(shocks, n_scenarios)
    sigma_batch = _tile_draws(sigma, n_scenarios)
    state_batch = _tile_draws(state0, n_scenarios)
    contacts = np.repeat(np.stack(matrices, axis=0), n_draws, axis=0)
    tiled = _tile_parameters(draws, n_scenarios)
    state, infections, symptomatic, hospital, deaths = _scan(
        state_batch,
        beta_batch,
        shock_batch,
        sigma_batch,
        contacts,
        tiled,
        population,
    )
    segment = _collapse(
        state,
        infections,
        symptomatic,
        hospital,
        deaths,
        beta_batch,
        contacts,
        tiled,
        population,
    )
    out: list[tuple[LatentSegment, np.ndarray]] = []
    for index in range(n_scenarios):
        sl = slice(index * n_draws, (index + 1) * n_draws)
        out.append((_slice_segment(segment, sl), state[sl, -1]))
    return out


def _tile_parameters(draws: ParameterDraws, n_scenarios: int) -> ParameterDraws:
    def tile(array: np.ndarray) -> np.ndarray:
        return _tile_draws(array, n_scenarios)

    return ParameterDraws(
        beta_t=tile(draws.beta_t),
        beta_sigma=tile(draws.beta_sigma),
        incubation_days=tile(draws.incubation_days),
        symptomatic_days=tile(draws.symptomatic_days),
        hospital_days=tile(draws.hospital_days),
        icu_days=tile(draws.icu_days),
        symptomatic_fraction=tile(draws.symptomatic_fraction),
        asymptomatic_infectiousness=tile(draws.asymptomatic_infectiousness),
        sensitivity=tile(draws.sensitivity),
        specificity=tile(draws.specificity),
        p_test=tile(draws.p_test),
        flu_mean=tile(draws.flu_mean),
        e0=tile(draws.e0),
        i0_share=tile(draws.i0_share),
        p_hosp=tile(draws.p_hosp),
        p_icu=tile(draws.p_icu),
        p_death_ward=tile(draws.p_death_ward),
        p_death_icu=tile(draws.p_death_icu),
        process_sigma=tile(draws.process_sigma),
        concentration=tile(draws.concentration),
    )


def _scan(
    state0: np.ndarray,
    beta: np.ndarray,
    shocks: np.ndarray,
    sigma: np.ndarray,
    contacts: np.ndarray,
    draws: ParameterDraws,
    population: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pop = jnp.asarray(population)

    def one(
        state: jax.Array,
        beta_t: jax.Array,
        shock_t: jax.Array,
        sigma_i: jax.Array,
        contact: jax.Array,
        incubation: jax.Array,
        symptomatic_days: jax.Array,
        hospital_days: jax.Array,
        icu_days: jax.Array,
        fraction: jax.Array,
        infectiousness: jax.Array,
        p_hosp: jax.Array,
        p_icu: jax.Array,
        p_death_ward: jax.Array,
        p_death_icu: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        trajectory = run_dynamics(
            state,
            beta_t,
            shock_t,
            population=pop,
            contacts=contact,
            sigma=1.0 / incubation,
            gamma=1.0 / symptomatic_days,
            gamma_h=1.0 / hospital_days,
            gamma_icu=1.0 / icu_days,
            symptomatic_fraction=fraction,
            p_hosp=p_hosp,
            p_icu=p_icu,
            p_death_ward=p_death_ward,
            p_death_icu=p_death_icu,
            asymptomatic_infectiousness=infectiousness,
            process_sigma=sigma_i,
        )
        return (
            trajectory["state"],
            trajectory["new_infections"],
            trajectory["new_symptomatic"],
            trajectory["new_hospital"],
            trajectory["new_deaths"],
        )

    state, infections, symptomatic, hospital, deaths = jax.vmap(one)(
        jnp.asarray(state0),
        jnp.asarray(beta),
        jnp.asarray(shocks),
        jnp.asarray(sigma),
        jnp.asarray(contacts),
        jnp.asarray(draws.incubation_days),
        jnp.asarray(draws.symptomatic_days),
        jnp.asarray(draws.hospital_days),
        jnp.asarray(draws.icu_days),
        jnp.asarray(draws.symptomatic_fraction),
        jnp.asarray(draws.asymptomatic_infectiousness),
        jnp.asarray(draws.p_hosp),
        jnp.asarray(draws.p_icu),
        jnp.asarray(draws.p_death_ward),
        jnp.asarray(draws.p_death_icu),
    )
    return (
        np.asarray(state),
        np.asarray(infections),
        np.asarray(symptomatic),
        np.asarray(hospital),
        np.asarray(deaths),
    )


def _collapse(
    state: np.ndarray,
    infections: np.ndarray,
    symptomatic: np.ndarray,
    hospital_in: np.ndarray,
    deaths: np.ndarray,
    beta: np.ndarray,
    contacts: np.ndarray,
    draws: ParameterDraws,
    population: np.ndarray,
) -> LatentSegment:
    total_population = float(population.sum())
    hospitalized = state[:, :, :, H].sum(axis=2)
    critical = state[:, :, :, ICU].sum(axis=2)
    susceptible = state[:, :, :, S].sum(axis=2)
    infectious = (
        draws.asymptomatic_infectiousness[:, None, None] * state[:, :, :, IA]
        + state[:, :, :, IS]
    ).sum(axis=2)
    new_infections = infections.sum(axis=2)
    new_symptomatic = symptomatic.sum(axis=2)
    new_hospital = hospital_in.sum(axis=2)
    new_deaths = deaths.sum(axis=2)
    row_mass = contacts.sum(axis=2)
    mean_contacts = (row_mass * population[None, :]).sum(axis=1) / total_population
    weight = draws.symptomatic_days * (
        draws.symptomatic_fraction
        + (1.0 - draws.symptomatic_fraction) * draws.asymptomatic_infectiousness
    )
    rt = (
        beta * mean_contacts[:, None] * weight[:, None] * susceptible / total_population
    )
    denominator = susceptible * infectious / total_population
    beta_effective = np.divide(
        new_infections,
        denominator,
        out=np.full_like(new_infections, np.nan),
        where=denominator > 1.0,
    )
    return LatentSegment(
        new_infections=new_infections,
        new_symptomatic=new_symptomatic,
        new_hospital=new_hospital,
        new_deaths=new_deaths,
        hospitalized=hospitalized,
        critical=critical,
        susceptible=susceptible,
        infectious=infectious,
        rt=rt,
        beta_effective=beta_effective,
        beta=beta,
    )


def _slice_segment(segment: LatentSegment, sl: slice) -> LatentSegment:
    return LatentSegment(
        new_infections=segment.new_infections[sl],
        new_symptomatic=segment.new_symptomatic[sl],
        new_hospital=segment.new_hospital[sl],
        new_deaths=segment.new_deaths[sl],
        hospitalized=segment.hospitalized[sl],
        critical=segment.critical[sl],
        susceptible=segment.susceptible[sl],
        infectious=segment.infectious[sl],
        rt=segment.rt[sl],
        beta_effective=segment.beta_effective[sl],
        beta=segment.beta[sl],
    )


def _concat(past: LatentSegment | None, future: LatentSegment) -> LatentSegment:
    if past is None:
        return future

    def join(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.concatenate([left, right], axis=1)

    return LatentSegment(
        new_infections=join(past.new_infections, future.new_infections),
        new_symptomatic=join(past.new_symptomatic, future.new_symptomatic),
        new_hospital=join(past.new_hospital, future.new_hospital),
        new_deaths=join(past.new_deaths, future.new_deaths),
        hospitalized=join(past.hospitalized, future.hospitalized),
        critical=join(past.critical, future.critical),
        susceptible=join(past.susceptible, future.susceptible),
        infectious=join(past.infectious, future.infectious),
        rt=join(past.rt, future.rt),
        beta_effective=join(past.beta_effective, future.beta_effective),
        beta=join(past.beta, future.beta),
    )


def _with_observation(
    segment: LatentSegment,
    draws: ParameterDraws,
    day: np.ndarray,
    rng: np.random.Generator,
    *,
    cases_likelihood: str,
    hospital_likelihood: str,
) -> Projection:
    positive_mean, tested_mean, false_negatives = expected_counts(
        segment.new_symptomatic,
        segment.new_hospital,
        draws.p_test,
        draws.sensitivity,
        draws.specificity,
        draws.flu_mean,
    )
    tested = sample_negative_binomial(rng, tested_mean, draws.concentration)
    if cases_likelihood == "negative_binomial":
        positive = sample_negative_binomial(rng, positive_mean, draws.concentration)
    elif cases_likelihood == "beta_binomial":
        positive = sample_beta_binomial(
            rng, positive_mean, tested_mean, draws.concentration, tested
        )
    else:
        raise ValueError("La vraisemblance des cas est inconnue.")
    if hospital_likelihood == "negative_binomial":
        hospital = sample_negative_binomial(
            rng, segment.hospitalized, draws.concentration
        )
        critical = sample_negative_binomial(rng, segment.critical, draws.concentration)
    elif hospital_likelihood == "lognormal":
        hospital = sample_log1p(rng, segment.hospitalized, HOSPITAL_LOG_SCALE)
        critical = sample_log1p(rng, segment.critical, HOSPITAL_LOG_SCALE)
    else:
        raise ValueError("La vraisemblance de l'hôpital est inconnue.")
    deaths = sample_negative_binomial(rng, segment.new_deaths, draws.concentration)
    return Projection(
        day=day,
        num_positive=positive,
        num_tested=tested,
        num_hospitalised=hospital,
        num_critical=critical,
        death_increments=deaths,
        new_infections=segment.new_infections,
        hospitalized=segment.hospitalized,
        critical=segment.critical,
        susceptible=segment.susceptible,
        infectious=segment.infectious,
        rt=segment.rt,
        beta_effective=segment.beta_effective,
        false_negatives=false_negatives,
        expected_confirmed=positive_mean,
    )
