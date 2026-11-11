# Source des données externes

Fichiers téléchargés depuis le dépôt du cours PROJ0016 :

- dépôt : https://github.com/ADelau/proj0016-epidemic-data
- commit : `42303fde5ab9a4af15c8d114e71adc30db482740`
- date : 15 février 2021
- auteur : Arnaud Delaunoy (ULiège)

Contenu retenu :

- `data.csv` : 71 jours. Colonnes d'origine : `Day`, `num_positive`, `num_tested`, `num_hospitalised`, `num_cumulative_hospitalizations`, `num_critical`, `num_fatalities`. Le nom `num_hospitalised` est celui du fichier (orthographe britannique).
- `stats/` : âges, ménages, écoles, lieux de travail et communautés (`*.json`), plus les figures `*.png` publiées avec ces fichiers.

`data/data.csv`, déjà présent dans ce dépôt, a la même empreinte SHA-256 que `data/external/data.csv`. La copie de référence pour la suite est `data/external/`.

Les séries produites par notre simulateur iront dans `data/generated/`. Ce dossier est ignoré par git, hormis `.gitkeep`.
