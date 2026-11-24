"""Ajustement sur des séries publiées. Le simulateur n'est pas lancé."""

from pathlib import Path

import pandas as pd
import streamlit as st

from epidemic.config import (
    load_game_config,
    load_inference_config,
    load_reporting_config,
    resolve_path,
)
from epidemic.inference.fit import configured, fit_observations
from epidemic.reporting.context import score_labeled
from epidemic.reporting.text import format_percent
from epidemic.series import observations_from_frame


def render() -> None:
    """Affiche les séries chargées et, sur demande, un score ou un SVI."""
    st.title("Analyser")
    st.caption(
        "Les séries viennent d'un fichier. Rien n'est lu dans les paramètres secrets."
    )
    frame = _load_frame()
    if frame is None:
        return
    st.line_chart(
        frame.set_index("Day")[["num_positive", "num_hospitalised", "num_critical"]]
    )
    if st.button("Score des références naïves"):
        _show_score(frame)
    st.divider()
    st.write(
        "L'ajustement court utilise le SVI du fichier d'inférence. "
        "Le NUTS de référence reste la commande `epidemic fit`."
    )
    if st.button("Lancer le SVI"):
        _run_svi(frame)


def _load_frame() -> pd.DataFrame | None:
    choice = st.radio("Source", ["Cours, 71 jours", "Fichier CSV"], horizontal=True)
    if choice == "Cours, 71 jours":
        path = resolve_path(load_inference_config().course_data)
        return pd.read_csv(path)
    upload = st.file_uploader("CSV au format du cours", type=["csv"])
    if upload is None:
        st.info("Choisissez un fichier pour continuer.")
        return None
    return pd.read_csv(upload)


def _show_score(frame: pd.DataFrame) -> None:
    reporting = load_reporting_config()
    try:
        result = score_labeled(
            observations_from_frame(frame),
            "fichier",
            reporting,
            load_game_config().bulletin_period_days,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    high = reporting.interval_masses[1]
    st.write(
        f"Couverture {format_percent(high, 0)} de la persistance : "
        f"{format_percent(result.coverage(high=True), 0)} "
        f"sur {len(result.hits)} prévisions."
    )


def _run_svi(frame: pd.DataFrame) -> None:
    settings = configured(load_inference_config(), method="svi", strict=False)
    try:
        series = observations_from_frame(frame)
    except ValueError as exc:
        st.error(str(exc))
        return
    with st.spinner("SVI en cours."):
        result = fit_observations(series, settings, show_progress=False)
    st.write(result.diagnostics.message)
    if result.output_path is not None:
        st.write(f"Posterior : {Path(result.output_path)}")
