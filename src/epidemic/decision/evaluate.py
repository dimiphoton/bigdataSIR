"""Relie un posterior, les scénarios et les probabilités de saturation."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from epidemic.config import (
    DecisionConfig,
    GameConfig,
    ScenariosConfig,
    load_decision_config,
    load_game_config,
    load_inference_config,
    load_scenarios_config,
    load_thresholds_config,
)
from epidemic.decision.catalog import (
    Scenario,
    enumerate_scenarios,
    scenario_key,
    scenario_label,
)
from epidemic.decision.policies import iter_rules
from epidemic.decision.recommend import Recommendation, ScoredOption, recommend
from epidemic.decision.reduction import (
    ReachDraw,
    beta_scales,
    factor_samples,
    isolation_reach,
    point_factor,
    relative_contacts,
)
from epidemic.decision.sensitivity import (
    by_alpha,
    factors_by_compliance,
    isolation_factor,
)
from epidemic.forecast.bands import Fan, fan_from_draws
from epidemic.forecast.closed_loop import rollout_rule
from epidemic.forecast.probabilities import (
    RiskSummary,
    death_interval,
    probability_over,
    summarize,
    weekly_probability_below,
)
from epidemic.forecast.project import (
    ParameterDraws,
    Projection,
    project_scenarios,
    subsample,
)
from epidemic.model.age_structure import (
    AgeStructure,
    load_age_structure,
    load_contact_matrices,
)
from epidemic.series import ObservationSeries
from epidemic.simulator.contacts import ContactMatrices
from epidemic.simulator.game import implementation_day
from epidemic.simulator.interventions import AppliedPolicy
from epidemic.viz.tracking import growth_sentence


@dataclass
class BulletinAdvice:
    """Recommandation d'un bulletin, avec les chiffres qui l'expliquent."""

    options: list[ScoredOption]
    recommendation: Recommendation
    baseline_key: str
    p_hospital_alert: float
    p_saturation_alert: float
    alpha_table: list[tuple[float, str | None]]
    compliance_table: list[tuple[float, float]]
    isolation_factor_low: float
    isolation_factor_high: float
    growth_text: str
    bulletin_day: int
    end_day: int


def prepare_bulletin(
    draws: ParameterDraws,
    *,
    bulletin_day: int,
    allowed: list[str],
    current: tuple[AppliedPolicy, ...] = (),
    end_day: int | None = None,
    observed: ObservationSeries | None = None,
    with_rules: bool = False,
    indicators: list[str] | None = None,
    seed: int | None = None,
    accept: Callable[[tuple[AppliedPolicy, ...]], bool] | None = None,
) -> BulletinAdvice:
    """Compare les paquets autorisés. Les règles ne partent que si on le demande."""
    if bulletin_day != draws.n_past:
        raise ValueError(
            "Le jour du bulletin doit couvrir exactement les jours déjà tirés."
        )
    if observed is not None and observed.n_days != bulletin_day:
        raise ValueError("Les observations ne couvrent pas le jour du bulletin.")
    settings = load_decision_config()
    game = load_game_config()
    scenario_config = load_scenarios_config()
    thresholds = load_thresholds_config()
    inference = load_inference_config()
    used_seed = settings.future_seed if seed is None else seed
    if draws.n_draws > settings.forecast_draws:
        draws = subsample(draws, settings.forecast_draws, used_seed)
    horizon = game.end_day if end_day is None else end_day
    if horizon < bulletin_day:
        raise ValueError("La fin de l'horizon est avant le bulletin.")

    structure = load_age_structure(draws.n_groups)
    contacts = load_contact_matrices(draws.n_groups)
    if structure.n_groups != draws.n_groups:
        raise ValueError("Les groupes d'âge ne correspondent pas aux tirages.")
    effect_day = implementation_day(bulletin_day, game)
    switch_index = min(max(effect_day - 1, draws.n_past), horizon)
    catalog = _with_current(
        [
            item
            for item in enumerate_scenarios(allowed, scenario_config, settings)
            if accept is None or accept(item.policies)
        ],
        current,
    )
    compliance = _compliance(settings)
    matrices = []
    scales = []
    factors = []
    for scenario in catalog:
        matrices.append(
            _matrix(
                scenario.policies,
                current,
                scenario_config,
                contacts,
                settings,
                compliance,
            )
        )
        sampled = factor_samples(
            scenario.policies,
            current,
            scenario_config,
            contacts,
            settings,
            n_draws=draws.n_draws,
            seed=used_seed,
            compliance=compliance,
        )
        scales.append(beta_scales(sampled))
        factors.append(
            point_factor(
                scenario.policies,
                current,
                scenario_config,
                contacts,
                settings,
                compliance=compliance,
            )
        )
    projections = project_scenarios(
        draws,
        structure,
        matrices,
        n_days=horizon,
        switch_index=switch_index,
        scales=np.stack(scales, axis=0),
        seed=used_seed,
        cases_likelihood=inference.likelihood.cases,
        hospital_likelihood=inference.likelihood.hospital,
    )
    current_cost = _unit_cost(current, game)
    options = [
        _from_projection(
            scenario.key,
            scenario.label,
            "fixe",
            _unit_cost(scenario.policies, game),
            _fixed_cost(
                scenario, current_cost, game, switch_index, draws.n_past, horizon
            ),
            factor,
            projection,
            observed,
            bulletin_day,
            thresholds.max_hospitalized,
            thresholds.max_critical,
            settings,
        )
        for scenario, projection, factor in zip(
            catalog, projections, factors, strict=True
        )
    ]
    baseline_key = scenario_key(current)
    baseline_projection = projections[catalog.index(_match(catalog, baseline_key))]
    hospital, critical, _deaths, _positive = _spliced(baseline_projection, observed)
    if with_rules:
        options.extend(
            _rule_options(
                draws,
                structure,
                contacts,
                current,
                allowed,
                scenario_config,
                settings,
                game,
                baseline_projection,
                observed,
                horizon,
                bulletin_day,
                thresholds.max_hospitalized,
                thresholds.max_critical,
                indicators,
                used_seed,
                compliance,
                inference.likelihood.cases,
                inference.likelihood.hospital,
            )
        )
    result = recommend(options, settings.alpha)
    chosen_policies = _chosen_policies(result, catalog, current)
    return BulletinAdvice(
        options=options,
        recommendation=result,
        baseline_key=baseline_key,
        p_hospital_alert=probability_over(
            hospital,
            bulletin_day - 1,
            game.early_tier2.horizon_days,
            game.early_tier2.hospital_threshold,
        ),
        p_saturation_alert=_saturation_window(
            hospital,
            critical,
            bulletin_day - 1,
            game.early_tier3.horizon_days,
            thresholds.max_hospitalized,
            thresholds.max_critical,
        ),
        alpha_table=by_alpha(options, settings.alphas),
        compliance_table=factors_by_compliance(
            chosen_policies,
            current,
            scenario_config,
            contacts,
            settings,
            settings.compliances,
        ),
        isolation_factor_low=isolation_factor(
            0.50,
            scenario_config,
            contacts,
            settings,
            duration_days=10,
            compliance=compliance,
        ),
        isolation_factor_high=isolation_factor(
            0.70,
            scenario_config,
            contacts,
            settings,
            duration_days=10,
            compliance=compliance,
        ),
        growth_text=growth_sentence(
            baseline_projection.new_infections[:, :bulletin_day]
        ),
        bulletin_day=bulletin_day,
        end_day=horizon,
    )


def _compliance(settings: DecisionConfig) -> float:
    return float(settings.compliances[len(settings.compliances) // 2])


def _matrix(
    new_policies: tuple[AppliedPolicy, ...],
    current: tuple[AppliedPolicy, ...],
    scenarios: ScenariosConfig,
    contacts: ContactMatrices,
    settings: DecisionConfig,
    compliance: float,
) -> np.ndarray:
    reach = _reach(settings)
    household = contacts.mean_household_size
    return relative_contacts(
        new_policies,
        current,
        scenarios,
        contacts,
        compliance,
        isolation_reach(new_policies, reach, household),
        isolation_reach(current, reach, household),
    )


def _reach(settings: DecisionConfig) -> ReachDraw:
    item = settings.isolation_reach
    return ReachDraw(
        item.symptomatic_fraction,
        item.confirmation_probability,
        item.infectious_days,
    )


def _with_current(
    catalog: list[Scenario], current: tuple[AppliedPolicy, ...]
) -> list[Scenario]:
    key = scenario_key(current)
    if any(item.key == key for item in catalog):
        return catalog
    return [Scenario(key, scenario_label(current), current), *catalog]


def _match(catalog: list[Scenario], key: str) -> Scenario:
    for item in catalog:
        if item.key == key:
            return item
    raise ValueError("La politique en cours n'est pas dans les scénarios.")


def _unit_cost(policies: tuple[AppliedPolicy, ...], game: GameConfig) -> float:
    total = 0.0
    for policy in policies:
        if policy.name not in game.constraint_costs:
            raise ValueError(f"Coût manquant pour {policy.name}.")
        total += float(game.constraint_costs[policy.name])
    return total


def _fixed_cost(
    scenario: Scenario,
    current_cost: float,
    game: GameConfig,
    switch_index: int,
    n_past: int,
    horizon: int,
) -> float:
    pre_days = max(switch_index - n_past, 0)
    post_days = horizon - switch_index
    return current_cost * pre_days + _unit_cost(scenario.policies, game) * post_days


def _from_projection(
    key: str,
    label: str,
    kind: str,
    unit_cost: float,
    expected_cost: float,
    factor: float,
    projection: Projection,
    observed: ObservationSeries | None,
    bulletin_day: int,
    max_hospitalized: float,
    max_critical: float,
    settings: DecisionConfig,
) -> ScoredOption:
    hospital, critical, deaths, positive = _spliced(projection, observed)
    summary = summarize(
        hospital,
        critical,
        deaths,
        projection.rt,
        projection.day,
        max_hospitalized=max_hospitalized,
        max_critical=max_critical,
        decision_index=bulletin_day - 1,
    )
    fans = _fans(projection, hospital, critical, deaths, positive, settings)
    return _option(
        key,
        label,
        kind,
        unit_cost,
        expected_cost,
        factor,
        summary,
        fans,
        weekly_probability_below(projection.rt),
    )


def _spliced(
    projection: Projection, observed: ObservationSeries | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if observed is None:
        return (
            projection.num_hospitalised,
            projection.num_critical,
            projection.death_increments,
            projection.num_positive,
        )
    return (
        _paste(projection.num_hospitalised, observed.num_hospitalised),
        _paste(projection.num_critical, observed.num_critical),
        _paste(projection.death_increments, observed.death_increments),
        _paste(projection.num_positive, observed.num_positive),
    )


def _paste(simulated: np.ndarray, observed: np.ndarray) -> np.ndarray:
    pasted = simulated.copy()
    pasted[:, : observed.shape[0]] = observed
    return pasted


def _ratio(infections: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Infections estimées par cas confirmé. Un cas nul compte comme 1."""
    return infections / np.clip(positive, 1.0, None)


def _fans(
    projection: Projection,
    hospital: np.ndarray,
    critical: np.ndarray,
    deaths: np.ndarray,
    positive: np.ndarray,
    settings: DecisionConfig,
) -> dict[str, Fan]:
    masses = tuple(settings.interval_masses)
    day = projection.day
    cumulative = np.cumsum(np.clip(deaths, 0.0, None), axis=1)
    beta = np.nan_to_num(projection.beta_effective, nan=0.0)
    return {
        "hospital": fan_from_draws(day, hospital, masses),
        "critical": fan_from_draws(day, critical, masses),
        "positive": fan_from_draws(day, positive, masses),
        "infections": fan_from_draws(day, projection.new_infections, masses),
        "rt": fan_from_draws(day, projection.rt, masses),
        "deaths": fan_from_draws(day, cumulative, masses),
        "beta": fan_from_draws(day, beta, masses),
        "ratio": fan_from_draws(
            day, _ratio(projection.new_infections, positive), masses
        ),
    }


def _option(
    key: str,
    label: str,
    kind: str,
    unit_cost: float,
    expected_cost: float,
    factor: float,
    summary: RiskSummary,
    fans: dict[str, Fan],
    weekly_rt_below: np.ndarray,
) -> ScoredOption:
    median, low, high = death_interval(summary.cumulative_deaths, 0.9)
    return ScoredOption(
        key=key,
        label=label,
        kind=kind,
        unit_cost=unit_cost,
        expected_cost=expected_cost,
        factor=factor,
        p_hospital=summary.p_hospital,
        p_critical=summary.p_critical,
        p_saturation=summary.p_saturation,
        deaths_median=median,
        deaths_low=low,
        deaths_high=high,
        peak_hospital_median=float(np.median(summary.peak_hospital)),
        peak_hospital_day=float(np.median(summary.peak_hospital_day)),
        peak_critical_median=float(np.median(summary.peak_critical)),
        peak_critical_day=float(np.median(summary.peak_critical_day)),
        days_saturated_median=float(np.median(summary.days_saturated)),
        p_rt_now=summary.p_rt_now,
        p_rt_end=summary.p_rt_end,
        first_day=summary.first_day,
        weekly_rt_below=weekly_rt_below,
        fans=fans,
    )


def _saturation_window(
    hospital: np.ndarray,
    critical: np.ndarray,
    start: int,
    horizon: int,
    max_hospitalized: float,
    max_critical: float,
) -> float:
    window_h = hospital[:, start : start + horizon]
    window_c = critical[:, start : start + horizon]
    hit = (window_h > max_hospitalized) | (window_c > max_critical)
    return float(np.mean(np.any(hit, axis=1)))


def _chosen_policies(
    result: Recommendation,
    catalog: list[Scenario],
    current: tuple[AppliedPolicy, ...],
) -> tuple[AppliedPolicy, ...]:
    if result.chosen is None:
        return current
    for scenario in catalog:
        if scenario.key == result.chosen.key:
            return scenario.policies
    return current


def _rule_options(
    draws: ParameterDraws,
    structure: AgeStructure,
    contacts: ContactMatrices,
    current: tuple[AppliedPolicy, ...],
    allowed: list[str],
    scenarios: ScenariosConfig,
    settings: DecisionConfig,
    game: GameConfig,
    baseline: Projection,
    observed: ObservationSeries | None,
    horizon: int,
    bulletin_day: int,
    max_hospitalized: float,
    max_critical: float,
    indicators: list[str] | None,
    seed: int,
    compliance: float,
    cases_likelihood: str,
    hospital_likelihood: str,
) -> list[ScoredOption]:
    rules = iter_rules(allowed, current, scenarios, settings, indicators)
    if not rules:
        return []
    past_h, past_c, past_p, past_t = _past_vectors(baseline, observed, bulletin_day)
    current_cost = _unit_cost(current, game)
    built: list[ScoredOption] = []
    for rule in rules:
        package = tuple(current) + rule.policies
        sampled = factor_samples(
            package,
            current,
            scenarios,
            contacts,
            settings,
            n_draws=draws.n_draws,
            seed=seed,
            compliance=compliance,
        )
        rollout = rollout_rule(
            draws,
            structure,
            rule,
            n_days=horizon,
            contacts_off=np.asarray(structure.contacts, dtype=float),
            contacts_on=_matrix(
                package, current, scenarios, contacts, settings, compliance
            ),
            scales=beta_scales(sampled),
            past_hospital=past_h,
            past_critical=past_c,
            past_positive=past_p,
            past_tested=past_t,
            delay=_delay(bulletin_day, game),
            min_days=game.min_policy_days,
            seed=seed,
            cases_likelihood=cases_likelihood,
            hospital_likelihood=hospital_likelihood,
        )
        hospital = np.concatenate(
            [
                _past_block(past_h, baseline.num_hospitalised, observed, bulletin_day),
                rollout.num_hospitalised,
            ],
            axis=1,
        )
        critical = np.concatenate(
            [
                _past_block(past_c, baseline.num_critical, observed, bulletin_day),
                rollout.num_critical,
            ],
            axis=1,
        )
        positive = np.concatenate(
            [
                _past_block(past_p, baseline.num_positive, observed, bulletin_day),
                rollout.num_positive,
            ],
            axis=1,
        )
        deaths = np.concatenate(
            [
                _past_block(
                    _past_deaths(baseline, observed, bulletin_day),
                    baseline.death_increments,
                    observed,
                    bulletin_day,
                ),
                rollout.death_increments,
            ],
            axis=1,
        )
        rt = np.concatenate([baseline.rt[:, :bulletin_day], rollout.rt], axis=1)
        day = np.arange(1, horizon + 1)
        summary = summarize(
            hospital,
            critical,
            deaths,
            rt,
            day,
            max_hospitalized=max_hospitalized,
            max_critical=max_critical,
            decision_index=bulletin_day - 1,
        )
        extra = _unit_cost(rule.policies, game)
        per_draw = current_cost * rollout.active.shape[1] + extra * rollout.active.sum(
            axis=1
        )
        infections = np.concatenate(
            [baseline.new_infections[:, :bulletin_day], rollout.new_infections],
            axis=1,
        )
        fans = {
            "hospital": fan_from_draws(day, hospital, tuple(settings.interval_masses)),
            "critical": fan_from_draws(day, critical, tuple(settings.interval_masses)),
            "positive": fan_from_draws(day, positive, tuple(settings.interval_masses)),
            "infections": fan_from_draws(
                day, infections, tuple(settings.interval_masses)
            ),
            "rt": fan_from_draws(day, rt, tuple(settings.interval_masses)),
            "deaths": fan_from_draws(
                day, np.cumsum(deaths, axis=1), tuple(settings.interval_masses)
            ),
            "ratio": fan_from_draws(
                day,
                _ratio(infections, positive),
                tuple(settings.interval_masses),
            ),
        }
        built.append(
            _option(
                rule.key,
                rule.label,
                "conditionnelle",
                current_cost + extra,
                float(np.mean(per_draw)),
                float(np.mean(sampled)),
                summary,
                fans,
                weekly_probability_below(rt),
            )
        )
    return built


def _past_vectors(
    baseline: Projection, observed: ObservationSeries | None, bulletin_day: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if observed is not None:
        return (
            observed.num_hospitalised,
            observed.num_critical,
            observed.num_positive,
            observed.num_tested,
        )
    return (
        baseline.num_hospitalised[0, :bulletin_day],
        baseline.num_critical[0, :bulletin_day],
        baseline.num_positive[0, :bulletin_day],
        baseline.num_tested[0, :bulletin_day],
    )


def _past_deaths(
    baseline: Projection, observed: ObservationSeries | None, bulletin_day: int
) -> np.ndarray:
    if observed is not None:
        return observed.death_increments
    return baseline.death_increments[:, :bulletin_day].mean(axis=0)


def _past_block(
    public: np.ndarray,
    simulated: np.ndarray,
    observed: ObservationSeries | None,
    bulletin_day: int,
) -> np.ndarray:
    """Le passé publié est le même pour chaque tirage. Sinon on garde le tirage."""
    if observed is None:
        return simulated[:, :bulletin_day]
    return np.broadcast_to(public, (simulated.shape[0], public.shape[0])).copy()


def _delay(bulletin_day: int, game: GameConfig) -> int:
    return implementation_day(bulletin_day, game) - bulletin_day
