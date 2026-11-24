"""Accueil. Les deux modes sont dans le menu latéral."""

import streamlit as st

st.set_page_config(page_title="SEIR bayésien", layout="wide")
st.title("SEIR bayésien")
st.write(
    "Deux modes. **Jouer** avance la ville jour après jour et demande une décision "
    "à chaque bulletin. **Analyser** charge des séries déjà publiées, "
    "sans ouvrir l'état caché du simulateur."
)
st.write(
    "L'installation et la reproduction des chiffres sont dans le README : "
    "`make install`, puis `make all`."
)
