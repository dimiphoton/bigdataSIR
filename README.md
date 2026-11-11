# SEIR bayésien — PROJ0016

[![CI](https://github.com/dimiphoton/bigdataSIR/actions/workflows/ci.yml/badge.svg)](https://github.com/dimiphoton/bigdataSIR/actions/workflows/ci.yml)

Refonte du projet « Big Data Project 2020 – COVID-20 » (Gilles Louppe, ULiège). Une ville fictive d'un million d'habitants est observée jour après jour. Le but est d'estimer un modèle SEIR bayésien, de projeter l'épidémie et de recommander l'intervention la moins contraignante qui garde les hospitalisations sous 1 500 et les soins intensifs sous 300 jusqu'au 31 janvier.

Les choix qui ne figurent pas dans les consignes du cours sont dans `docs/hypotheses.md`. La documentation du cadre bayésien, générée, est dans `docs/framework-bayesien.md`. Le rapport est `reports/rapport.qmd`.

`old/`, `main.py` et les modules historiques de `src/` (hors `src/epidemic/`) restent en place. Ils ne font pas partie du nouveau code.

## Installation

Il faut [uv](https://docs.astral.sh/uv/) et Python 3.11 ou plus.

```bash
uv sync --all-groups
uv run pre-commit install
make all
```

`make all` lance le lint, les tests, la documentation, le rapport et les diapositives. `make install` ne fait que la première commande.

## Utilisation

```bash
uv run epidemic --help
uv run epidemic simulate --days 30
uv run epidemic play
uv run epidemic fit --method svi --days 21 --no-strict
uv run epidemic docs
uv run epidemic report
uv run epidemic bulletin --day 21
uv run epidemic slides
uv run streamlit run app/accueil.py
```

L'application a deux modes. **Jouer** avance la ville et demande une décision à chaque bulletin. **Analyser** charge `data.csv` ou un autre CSV public, calcule le score des références naïves, et peut lancer un SVI. Le NUTS de référence reste `epidemic fit`.

`epidemic docs` remplit les gabarits Jinja à partir des YAML et des calculs : aucun chiffre du texte n'est saisi à la main. `epidemic bulletin --day 21` écrit un bulletin de deux pages, avec les probabilités en langage clair et le chiffre à côté. Sans loi a posteriori enregistrée, les projections du bulletin sont celles de la référence naïve, et le texte le dit.

## Données

Les données du cours sont dans `data/external/`, avec la source dans `data/external/SOURCE.md`. Le simulateur écrit dans `data/generated/` (ignoré par git, hormis `.gitkeep`).

## Résultats

La documentation générée et les figures sont dans `docs/` et `docs/generated/figures/`. Le rapport HTML est `reports/rapport.html`. Le PDF est produit par `make report` si Edge ou Chrome est installé. Les diapositives Marp sont `docs/presentation.md`, exportées par `make slides`.

## Structure

- `configs/` : YAML du simulateur, des priors, de l'inférence, de la décision et du rendu.
- `src/epidemic/` : paquet du projet (`simulator`, `model`, `inference`, `forecast`, `decision`, `viz`, `reporting`).
- `app/` : application Streamlit, modes Jouer et Analyser.
- `docs/templates/` : gabarits de la documentation.
- `reports/` : rapport Quarto et bulletins.
- `tests/` : pytest.
- `.github/workflows/ci.yml` : ruff, mypy, pytest et `epidemic docs`.
