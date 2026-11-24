"""Page Jouer. Au bulletin, la recommandation vient du posterior enregistré."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from epidemic.config import load_inference_config, resolve_path
from epidemic.decision.evaluate import prepare_bulletin
from epidemic.forecast.idata import draws_from_idata
from epidemic.forecast.probabilities import saturation_by_day
from epidemic.series import observations_from_frame
from epidemic.simulator.game import Decision, Game
from epidemic.simulator.interventions import LABELS
from epidemic.viz.tracking import (
    plot_attack_rate,
    plot_capacity,
    plot_incidence,
    plot_loglog,
    plot_pareto,
    plot_reservoir,
    plot_rt,
    plot_scenario_fans,
    plot_underreporting,
)


def _start_game(seed: int, level: int, course: bool) -> Game:
    """Ouvre une partie et range les paramètres secrets hors de la page."""
    directory: Path = resolve_path("data/generated")
    return Game.start(
        seed=seed,
        level=level,
        observation_source="course" if course else "simulator",
        output_dir=directory,
        write_secret=True,
    )


def _needs_decision(game: Game) -> bool:
    return game.day in game.bulletin_day_list() and game.day not in game.decision_log


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Partie")
        seed = int(
            st.number_input("Graine", min_value=0, max_value=10_000, value=0, step=1)
        )
        level = int(st.selectbox("Stochasticité", [0, 1, 2, 3], index=1))
        course = st.checkbox("Rejouer les 71 jours du cours")
        st.caption(
            "0 déterministe, 1 démographique, 2 environnementale, 3 hétérogénéité."
        )
        if st.button("Nouvelle partie"):
            st.session_state.game = _start_game(seed, level, course)
            st.rerun()


def _render_status(game: Game) -> None:
    frame = game.observations()
    last = frame.iloc[-1] if not frame.empty else None
    columns = st.columns(5)
    columns[0].metric("Jour", game.day)
    columns[1].metric("Palier", game.tier_level())
    columns[2].metric(
        "Positifs du jour", 0 if last is None else int(last["num_positive"])
    )
    columns[3].metric(
        "Hospitalisés", 0 if last is None else int(last["num_hospitalised"])
    )
    columns[4].metric(
        "Soins intensifs", 0 if last is None else int(last["num_critical"])
    )
    if frame.empty:
        st.info("Aucune donnée publiée pour l'instant.")
        return
    chart = frame.set_index("Day")[["num_positive", "num_hospitalised", "num_critical"]]
    chart = chart.rename(
        columns={
            "num_positive": "Cas confirmés",
            "num_hospitalised": "Hospitalisés",
            "num_critical": "Soins intensifs",
        }
    )
    st.line_chart(chart)
    active = game.active_names()
    if active:
        st.write(
            "Mesures en vigueur : "
            + ", ".join(LABELS.get(name, name) for name in active)
        )


def _render_reveals(game: Game) -> None:
    reveals = game.revealed()
    if not reveals:
        return
    st.subheader("Informations publiées")
    for item in reveals:
        st.markdown(f"**J{item.day} — {item.title}.** {item.body.strip()}")
    if game.day >= game.game_config.stats_public_day:
        rows = [
            {"Tranche": group.name, "Population": group.population}
            for group in game.simulator.contacts.groups
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def _decision_from_form(selected: list[str], duration: int, age: int) -> list[Decision]:
    decisions: list[Decision] = []
    for name in selected:
        if name in ("case_isolation", "household_quarantine"):
            decisions.append(Decision(name, duration_days=duration))
        elif name == "social_distancing":
            decisions.append(Decision(name, age_threshold=age))
        else:
            decisions.append(Decision(name))
    return decisions


def _render_decision(game: Game) -> None:
    allowed = game.allowed_interventions()
    st.subheader(f"Décision du jour {game.day}")
    st.caption(
        "Elle prend effet après le délai prévu. Une mesure non répétée est levée."
    )
    with st.form("decision"):
        if allowed:
            selected = st.multiselect(
                "Mesures",
                options=allowed,
                format_func=lambda name: LABELS.get(name, name),
            )
            duration = int(
                st.number_input("Durée d'isolement (jours)", min_value=1, value=10)
            )
            age = int(
                st.number_input("Âge seuil de la distanciation", min_value=1, value=60)
            )
        else:
            selected = []
            duration = 10
            age = 60
            st.info("Palier 0 : observation seulement.")
        submitted = st.form_submit_button("Valider la décision")
    if submitted:
        try:
            game.decide(_decision_from_form(list(selected), duration, age))
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _render_advice(game: Game) -> None:
    """Recommandation bayésienne, si un posterior couvre le jour du bulletin."""
    st.subheader("Recommandation bayésienne")
    with st.sidebar:
        with_rules = st.checkbox("Évaluer les règles conditionnelles")
    path = _latest_posterior()
    if path is None:
        st.info(
            "Aucun posterior enregistré. "
            "La commande epidemic fit produit le fichier utilisé ici."
        )
        return
    try:
        advice = _cached_advice(game, path, with_rules)
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    _show_calibration(game, advice)
    allowed_before = tuple(game.allowed_interventions())
    game.submit_risk(advice.p_hospital_alert, advice.p_saturation_alert)
    if tuple(game.allowed_interventions()) != allowed_before:
        advice = _cached_advice(game, path, with_rules)
    st.write(advice.recommendation.message)
    rows = [
        {
            "Option": option.label,
            "Type": option.kind,
            "Coût cumulé": round(option.expected_cost),
            "P(saturation)": f"{option.p_saturation:.0%}",
            "P(H)": f"{option.p_hospital:.0%}",
            "P(ICU)": f"{option.p_critical:.0%}",
            "Décès médian": round(option.deaths_median),
            "Facteur": f"{option.factor:.2f}",
        }
        for option in advice.options
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    chosen = advice.recommendation.chosen
    if chosen is not None:
        st.caption(
            f"P(Rt < 1) aujourd'hui {chosen.p_rt_now:.0%}, "
            f"en fin d'horizon {chosen.p_rt_end:.0%}. "
            f"Pic d'hospitalisés médian {chosen.peak_hospital_median:.0f} "
            f"au jour {chosen.peak_hospital_day:.0f}."
        )
    alpha_text = ", ".join(
        f"{alpha:.0%} → {label or 'seuil dépassé'}"
        for alpha, label in advice.alpha_table
    )
    st.caption("Seuil de risque : " + alpha_text)
    _show_figures(game, advice, path)


def _latest_posterior():
    directory = resolve_path(load_inference_config().output_dir)
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.nc"), key=lambda item: item.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def _cached_advice(game: Game, path, with_rules: bool):
    import arviz as az

    allowed = tuple(game.allowed_interventions())
    key = (str(path), path.stat().st_mtime, game.day, allowed, with_rules)
    if st.session_state.get("advice_key") == key:
        return st.session_state["advice"]
    draws = draws_from_idata(az.from_netcdf(path))
    observed = observations_from_frame(game.observations())
    advice = prepare_bulletin(
        draws,
        bulletin_day=game.day,
        allowed=list(allowed),
        current=tuple(game._active(game.day)),
        observed=observed,
        with_rules=with_rules,
    )
    st.session_state["advice_key"] = key
    st.session_state["advice"] = advice
    return advice


def _show_calibration(game: Game, advice) -> None:
    previous = st.session_state.get("forecast_check")
    frame = game.observations()
    if previous and previous["issued"] < game.day and not frame.empty:
        hits = 0
        total = 0
        for day, value in zip(frame["Day"], frame["num_hospitalised"], strict=False):
            if int(day) <= previous["issued"]:
                continue
            matches = previous["day"] == int(day)
            if not matches.any():
                continue
            index = int(matches.argmax())
            total += 1
            if previous["low"][index] <= value <= previous["high"][index]:
                hits += 1
        if total:
            st.caption(
                f"Depuis le bulletin du jour {previous['issued']}, "
                f"{hits} valeur(s) d'hospitalisés sur {total} "
                "tombent dans l'intervalle à 90 %."
            )
    baseline = next(
        option for option in advice.options if option.key == advice.baseline_key
    )
    fan = baseline.fans["hospital"]
    st.session_state["forecast_check"] = {
        "issued": game.day,
        "day": fan.day,
        "low": fan.low90,
        "high": fan.high90,
    }


def _show_figures(game: Game, advice, path: Path) -> None:
    """Graphes de suivi du bulletin, pour la politique en cours."""
    frame = game.observations()
    if frame.empty:
        return
    baseline = _baseline(advice)
    spans = _spans(game)
    day = frame["Day"].to_numpy(dtype=int)
    positive = frame["num_positive"].to_numpy(dtype=float)
    tested = np.clip(frame["num_tested"].to_numpy(dtype=float), 1.0, None)
    _draw(
        plot_incidence(
            positive,
            day,
            baseline.fans["infections"],
            bulletin_day=game.day,
            spans=spans,
            note=advice.growth_text,
        )
    )
    if "beta" in baseline.fans:
        _draw(
            plot_attack_rate(
                baseline.fans["beta"],
                baseline.fans["rt"],
                bulletin_day=game.day,
                spans=spans,
            )
        )
    weekly = np.empty(positive.size, dtype=float)
    for index in range(positive.size):
        start = max(0, index - 6)
        weekly[index] = float(positive[start : index + 1].sum())
    _draw(plot_loglog(np.cumsum(positive), weekly))
    _draw(
        plot_reservoir(
            advice.baseline_latent,
            intervention_days=_intervention_starts(game),
        )
    )
    st.caption(
        "En phase exponentielle, les nouvelles infections estimées sont "
        "proportionnelles au réservoir infectieux : dI/I = β S/N, "
        "une droite de pente 1 en log-log. "
        "Le panneau du bas suit ce rapport au cours du temps."
    )
    _draw(
        plot_rt(
            baseline.fans["rt"],
            baseline.weekly_rt_below,
            bulletin_day=game.day,
            spans=spans,
        )
    )
    hospital = baseline.fans["hospital"]
    _draw(
        plot_capacity(
            frame["num_hospitalised"].to_numpy(dtype=float),
            frame["num_critical"].to_numpy(dtype=float),
            day,
            hospital,
            baseline.fans["critical"],
            max_hospitalized=game.thresholds.max_hospitalized,
            max_critical=game.thresholds.max_critical,
            saturation=saturation_by_day(baseline.first_day, hospital.day),
            bulletin_day=game.day,
            spans=spans,
        )
    )
    _draw(
        plot_underreporting(
            baseline.fans["ratio"],
            positive / tested,
            day,
            bulletin_day=game.day,
            spans=spans,
        )
    )
    shown = _frontier(advice)
    _draw(
        plot_scenario_fans(
            [(item.label, item.fans["hospital"]) for item in shown],
            ylabel="Hospitalisés",
        )
    )
    _draw(
        plot_scenario_fans(
            [(item.label, item.fans["critical"]) for item in shown],
            ylabel="Soins intensifs",
        )
    )
    st.caption(
        "Les courbes superposées sont la frontière de Pareto et la politique en cours. "
        "Le tableau liste tous les paquets."
    )
    _draw(_pareto_figure(advice))
    _show_diagnostics(path)


def _baseline(advice):
    return next(
        option for option in advice.options if option.key == advice.baseline_key
    )


def _intervention_starts(game: Game) -> list[int]:
    return sorted({int(policy.effect_start) for policy in game.policies})


def _spans(game: Game) -> list[tuple[int, int]]:
    spans = []
    for policy in game.policies:
        end = game.end_day if policy.effect_end is None else policy.effect_end
        spans.append((policy.effect_start, end))
    return spans


def _frontier(advice) -> list:
    keys = {advice.baseline_key}
    chosen = advice.recommendation.chosen
    if chosen is not None:
        keys.add(chosen.key)
    keys.update(option.key for option in advice.recommendation.pareto)
    return [option for option in advice.options if option.key in keys]


def _pareto_figure(advice):
    front_keys = {option.key for option in advice.recommendation.pareto}
    return plot_pareto(
        [option.label for option in advice.options],
        np.asarray([option.expected_cost for option in advice.options], dtype=float),
        np.asarray([option.p_saturation for option in advice.options], dtype=float),
        np.asarray([option.key in front_keys for option in advice.options]),
        [option.kind for option in advice.options],
    )


def _draw(figure) -> None:
    st.pyplot(figure)
    plt.close(figure)


def _show_diagnostics(path: Path) -> None:
    found = False
    for name in ("traces.png", "prior_posterior.png"):
        image = path.parent / name
        if image.is_file():
            st.image(str(image), caption=name)
            found = True
    if not found:
        st.caption(
            "Les traces, le R̂, l'ESS et les densités a priori / a posteriori "
            "sont écrits à côté du posterior par epidemic fit."
        )


def _render_progress(game: Game) -> None:
    if game.day >= game.end_day:
        result = game.score()
        st.subheader("Score")
        st.write(
            f"Jours en saturation : {result.days_saturated}. "
            f"Décès : {result.deaths}. "
            f"Coût des contraintes : {result.constraint_cost:g}."
        )
        st.write(_calibration_text(game))
        return
    if game.next_bulletin_day() is not None:
        label = "Bulletin suivant"
    else:
        label = "Aller à la fin"
    if st.button(label):
        if game.next_bulletin_day() is not None:
            game.next_bulletin()
        else:
            game.advance(game.end_day - game.day)
        st.rerun()


def render() -> None:
    """Affiche la partie en cours."""
    st.title("Jouer")
    st.caption(
        "Les courbes du haut sont les séries publiées. "
        "Au bulletin, la recommandation utilise un posterior déjà enregistré."
    )
    _render_sidebar()
    if "game" not in st.session_state:
        st.session_state.game = _start_game(0, 1, False)
    game: Game = st.session_state.game
    _render_status(game)
    _render_reveals(game)
    if _needs_decision(game) and game.day < game.end_day:
        _render_advice(game)
        _render_decision(game)
    else:
        _render_progress(game)


def _calibration_text(game: Game) -> str:
    """Couverture des références naïves sur les séries publiées."""
    from epidemic.config import load_game_config, load_reporting_config
    from epidemic.forecast.scoring import interpret_coverage
    from epidemic.reporting.context import score_labeled
    from epidemic.reporting.text import format_percent

    reporting = load_reporting_config()
    try:
        result = score_labeled(
            observations_from_frame(game.observations()),
            "partie",
            reporting,
            load_game_config().bulletin_period_days,
        )
    except ValueError as exc:
        return str(exc)
    high = reporting.interval_masses[1]
    coverage = format_percent(result.coverage(high=True), 0)
    comment = interpret_coverage(
        result.coverage(high=True),
        high,
        reporting.coverage_tolerance,
    )
    return (
        f"Calibration naïve, couverture {format_percent(high, 0)} : "
        f"{coverage} sur {len(result.hits)} prévisions. {comment}"
    )
