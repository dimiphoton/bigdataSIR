# Prompt Cursor – Refonte du projet SEIR bayésien (PROJ0016 ULiège)

> À coller dans Cursor (mode Agent), à la racine du dépôt `bigdataSIR`.
> Travaille par phases. **Arrête-toi à la fin de chaque phase**, résume ce qui a été fait et attends ma validation avant de passer à la suivante.

---

## 0. Contexte

Ce dépôt contient une ancienne version d'un projet universitaire : le cours PROJ0016 « Big Data Project 2020 – COVID-20 » de l'ULiège (Gilles Louppe).

**Le principe du cours :**
- Une ville fictive d'un million d'habitants est simulée par un simulateur à agents. Le simulateur du prof est très probablement inspiré de `github.com/cni-iisc/epidemic-simulator`.
- Les étudiants recevaient les données un jour à la fois.
- Après quelques semaines, ils devaient recommander des interventions publiques.
- Le prof relançait alors son simulateur avec leurs choix et renvoyait la suite des données.

Je refais tout le projet proprement. L'objectif est un travail de qualité professionnelle, qui utilise l'approche bayésienne pour produire des projections et des probabilités (notamment la probabilité de saturation des hôpitaux).

**Consignes officielles à respecter.** Source : `github.com/ADelau/proj0016-epidemic-data`, README.

*Données publiées chaque jour :*
- `num_positive` : nombre de positifs du jour ;
- `num_tested` : nombre de tests du jour ;
- `num_hospitalized` : hospitalisés actuels, sans les soins intensifs ;
- `num_critical` : patients actuellement en soins intensifs ;
- `num_cumulative_hospitalizations` : hospitalisations cumulées ;
- `num_fatalities` : décès cumulés.

*Informations épidémiologiques connues :*
- **Population** : 1 000 000 d'habitants.
- **Asymptomatiques** : on peut être infecté et contagieux sans symptômes. 50 à 70 % des infectés développent des symptômes.
- **Tests** :
  - au moins 50 % des symptomatiques sont testés ;
  - sensibilité de 70 à 85 %, spécificité de 100 % ;
  - toute hospitalisation entraîne un test positif.
- **Durées** :
  - incubation de 1 à 5 jours ;
  - période symptomatique avant hospitalisation de 4 à 10 jours.
- **Structure de la ville** : distributions des âges, des ménages, des écoles, des lieux de travail et des communautés (fichiers `stats/*.json` du dépôt ADelau).

*Interventions possibles.* 90 % de la population les suit. Quand plusieurs sont combinées, c'est la réduction la plus forte qui s'applique, sauf pour le masque, qui se multiplie avec les autres.

| Intervention | Paramètre | Effet |
|---|---|---|
| Masque | aucun | taux d'infection −20 % (effet multiplicatif) |
| Isolement des cas | durée (jours) | le symptomatique s'isole : −25 % à la maison, −90 % en communauté, −100 % au travail et à l'école |
| Quarantaine du ménage | durée (jours) | tout le ménage du symptomatique s'isole, avec les mêmes réductions. Ne peut pas être combinée avec l'isolement des cas |
| Confinement | aucun | −75 % en communauté, −75 % au travail, −100 % à l'école, 0 % à la maison |
| Distanciation sociale | âge seuil | les personnes au-dessus de cet âge : −75 % au travail et en communauté |

*Objectif de décision* : garder `num_hospitalized` sous 1 500 et `num_critical` sous 300 jusqu'à fin janvier, avec l'intervention **la moins contraignante possible**.

**Ce qui existe déjà** et peut être réutilisé s'il est propre :
- `src/simulate.py` : modèle SEIR-HICURD en JAX ;
- `src/build_model_numpyro.py` : modèle NumPyro ;
- `data/data.csv` : 71 jours de données réelles du cours.

`old/` et `main.py` sont obsolètes. Ne les supprime pas, mais ne t'en sers pas.

**Langue : tout le projet est en anglais** : code, noms de variables, docstrings, commentaires, README, documentation (`docs/`), rapport, bulletins, présentation, interface Streamlit, messages de la CLI, messages de commit et PR. Ce prompt est rédigé en français, mais **aucun texte en français ne doit apparaître dans le dépôt**. Les titres de fichiers donnés ici en français sont à traduire : par exemple `reports/rapport.qmd` devient `reports/report.qmd`, et la branche s'appelle `bayesian-rewrite`.

---

## Phase 1 – Branche et squelette du projet

1. Vérifie que l'arbre de travail est propre (`git status`). S'il ne l'est pas, montre-moi les changements avant de faire quoi que ce soit.
2. Crée et bascule sur une nouvelle branche : `git switch -c bayesian-rewrite`.
3. Mets en place une structure de projet professionnelle :

```
bigdataSIR/
├── pyproject.toml          # géré avec uv ; Python 3.11+
├── README.md               # in English: goal, installation, usage, results
├── Makefile                # make install / test / lint / simulate / fit / report / app
├── configs/                # YAML : simulateur, priors, scénarios, seuils
├── data/
│   ├── external/           # données du cours (data.csv + stats/*.json), avec la source citée
│   └── generated/          # données produites par notre simulateur (ignorées par git)
├── src/epidemic/
│   ├── simulator/          # « la réalité » : simulateur stochastique caché
│   ├── model/              # modèle SEIRH bayésien (ne voit QUE les observations)
│   ├── inference/          # MCMC, diagnostics
│   ├── forecast/           # projections, scénarios, probabilités de saturation
│   ├── decision/           # comparaison des interventions, recommandation
│   └── viz/                # graphiques
├── app/                    # application Streamlit (le « jeu » avec décisions)
├── notebooks/              # exploration uniquement, aucune logique métier
├── reports/                # rapport final (Quarto ou Jupyter → HTML/PDF)
├── tests/
└── .github/workflows/ci.yml   # ruff + mypy + pytest
```

4. Outils :
   - `ruff` pour le lint et le formatage ;
   - `mypy` pour le typage ;
   - `pytest` pour les tests ;
   - `pre-commit` ;
   - dépendances principales : `numpy`, `pandas`, `jax`, `numpyro`, `arviz`, `matplotlib`, `streamlit`, `pydantic` (configuration), `typer` (CLI), `jinja2` (documentation générée). Pour les slides : Marp CLI.
5. Télécharge `data.csv` et `stats/` depuis `https://github.com/ADelau/proj0016-epidemic-data` vers `data/external/`, avec un petit `SOURCE.md`.

**Point d'arrêt.**

---

## Phase 2 – Simulateur « réalité » avec décisions en cours de route

But : reproduire l'expérience du cours. Un simulateur cache les vrais paramètres, publie des données jour après jour, et **s'arrête à des dates clés pour demander des décisions**.

**Modèle de la réalité.** Le modèle doit être plus riche que le modèle d'inférence, pour que l'inférence soit honnête.
- Modèle stochastique en chaîne binomiale (tau-leaping, pas d'un jour) **structuré par âge et par lieu de contact** : maison, école, travail, communauté.
- Matrices de contact construites à partir de `stats/*.json`.
- Compartiments : S, E, I_asymptomatique, I_présymptomatique, I_symptomatique, H, ICU, R, D.
- Les vrais paramètres sont tirés dans les plages de la section 0 à partir d'une graine, puis gardés **secrets**, dans un fichier `data/generated/.truth.json` ignoré par git.

**Niveaux de stochasticité** : option `stochasticity_level` dans `configs/simulator.yaml`. Chaque niveau ajoute une source d'aléa au précédent, et chacune peut être activée ou désactivée séparément.

| Niveau | Nom | Contenu | Usage |
|---|---|---|---|
| 0 | Déterministe | équations moyennes, aucun hasard | débogage, tests unitaires |
| 1 | Démographique | chaîne binomiale : qui est infecté, testé ou hospitalisé est tiré au sort ; les paramètres sont fixes | **niveau par défaut** du développement |
| 2 | Environnementale | les paramètres varient lentement dans le temps : β(t) suit un processus d'Ornstein-Uhlenbeck sur log β (petite volatilité, environ 5 %/jour), une saisonnalité optionnelle, p_test(t) et le fond de syndromes grippaux dérivent, la sensibilité varie d'un jour à l'autre (Beta autour de sa moyenne) | **niveau par défaut du jeu** : réalisme |
| 3 | Hétérogénéité individuelle | la contagiosité individuelle suit une loi Gamma de dispersion k ≈ 0,1-0,5 (super-propagateurs), le respect des mesures varie d'un individu à l'autre, les durées suivent des lois Gamma plutôt qu'exponentielles | test de robustesse |

- Les niveaux 2 et 3 restent **modérés** : ils servent à rendre la « réalité » un peu imprévisible, pas à la rendre chaotique. Les volatilités sont paramétrables.
- Les **mêmes graines** donnent les mêmes trajectoires à tous les niveaux. On peut ainsi comparer l'effet d'une décision sur la même épidémie.
- **Test de robustesse obligatoire** : ajuster le modèle bayésien sur des données de niveau 1, 2 et 3, puis comparer la couverture des intervalles. On montre ainsi jusqu'où le modèle reste fiable quand la réalité est plus complexe que lui.

**Trois notions à ne jamais confondre** : l'état infectieux, le fait d'être testé et le résultat du test. Ce sont **deux dimensions séparées** :

1. **L'état infectieux** (latent, jamais observé directement) : S, E, I_a, I_p, I_s, H, ICU, R, D.
2. **L'état de surveillance**, porté par chaque individu ou par une sous-cohorte de chaque compartiment :
   - *non testé* ;
   - *testé négatif* : vrai négatif si non infecté, **faux négatif** si infecté ;
   - *testé positif*, autrement dit **cas confirmé**.

Processus d'observation à implémenter :
- **Qui est testé.**
  - Une fraction p_test(t) des **nouveaux symptomatiques infectés** est testée (≥ 50 %).
  - Une partie des tests porte sur des personnes **non infectées** qui ont des symptômes semblables (fond de syndromes grippaux, paramètre caché et variable dans le temps). C'est pour cela que `num_tested` est supérieur à `num_positive`.
  - Les asymptomatiques et présymptomatiques ne sont pas testés, sauf si une extension « dépistage élargi » est activée.
- **Résultat du test.**
  - Infecté testé : positif avec probabilité = sensibilité (70-85 %), sinon **faux négatif**.
  - Non infecté testé : négatif avec probabilité = spécificité. La valeur par défaut est 100 %, comme dans les consignes, mais le paramètre doit être configurable pour une analyse de sensibilité, par exemple à 99,5 %.
- **Conséquences.**
  - Un **faux négatif reste infectieux** et ne déclenche pas l'isolement des cas ni la quarantaine du ménage.
  - Un cas confirmé déclenche les interventions liées au test.
  - Un symptomatique non testé suit les règles d'isolement « sur symptômes » s'il en existe.
- **Hospitalisation.**
  - Toute hospitalisation implique un test positif.
  - Un hospitalisé jamais testé ou faux négatif est **compté comme positif le jour de son admission**, sans double comptage.
- **Bruit.** Bruit sur le nombre de tests et capacité de test journalière maximale (paramètre).
- **Sortie.** Exactement les colonnes de `data.csv`, plus un fichier caché `truth_daily.csv` : vraies incidences, prévalence, nombres de faux négatifs. Ce fichier sert uniquement à la validation.
- **Tests unitaires** :
  - `num_positive ≤ num_tested` ;
  - aucun positif chez les non-infectés si la spécificité vaut 100 % ;
  - la fraction de faux négatifs converge vers 1 − sensibilité.
- Option : un mode qui charge le vrai `data.csv` du cours comme « réalité » des 71 premiers jours.

**Interventions.**
- Implémente les 5 interventions de la section 0 avec leurs règles exactes : 90 % de respect, réduction la plus forte quand on combine, masque multiplicatif, isolement des cas incompatible avec la quarantaine du ménage.
- Chaque intervention a une date de début et de fin.

**Calendrier : rapports et paliers de politiques.** Le jour 1 correspond au 2 octobre 2020. Tout le calendrier doit être paramétrable dans `configs/game.yaml`.

- **Données** : publiées **chaque jour**, comme dans le cours.
- **Bulletin épidémiologique** :
  - publié **chaque semaine**, au jour 7, 14, 21, etc. ;
  - c'est le seul moment où l'on peut décider ;
  - il est produit automatiquement par le pipeline bayésien : Rt, projections à 14 et 28 jours, P(saturation), et la recommandation avec son niveau de risque.
- **Informations révélées** aux mêmes dates que dans le cours :
  - J4 : population, asymptomatiques, sensibilité 70-90 % ;
  - J25 : politique de test, sensibilité affinée à 70-85 %, incubation, durée symptomatique ;
  - J60 : statistiques de la ville, fraction symptomatique.
- **Paliers de politiques.** Ils se débloquent progressivement, par date **ou** par seuil d'alerte :

| Palier | Débloqué | Interventions autorisées | Justification |
|---|---|---|---|
| 0 – Veille | J1 → J20 | aucune (observation seulement) | l'épidémie n'est pas encore caractérisée |
| 1 – Mesures légères | bulletin J21 | masque, isolement des cas (durée à choisir) | mesures peu coûteuses, ciblées sur les cas |
| 2 – Mesures ciblées | bulletin J42, **ou plus tôt** si P(H > 750 dans les 28 jours) > 20 % | quarantaine du ménage, distanciation sociale par âge | elles nécessitent la structure des ménages et des âges, ce qui oblige à les modéliser explicitement |
| 3 – Crise | bulletin J63 (« l'épidémie devient hors de contrôle »), **ou plus tôt** si P(saturation dans les 21 jours) > 20 % | confinement | mesure la plus coûteuse, réservée à la crise |

- **Règles de décision.**
  - **Délai de mise en œuvre** : 3 jours entre la décision et l'effet. Pour la décision finale, la mise en œuvre est fixée à J75 (15 décembre), comme dans le cours.
  - **Durée minimale** : 14 jours pour une mesure, afin d'éviter l'effet yo-yo.
  - **Levée** : une mesure peut être levée à n'importe quel bulletin, une fois la durée minimale passée.
  - **Fin de partie** : J122 (31 janvier).
- **Score de fin de partie** : jours en saturation, décès et coût cumulé des contraintes (somme des coûts × jours). C'est ce score qui permet de comparer des stratégies.

**Boucle de jeu.** Moteur sans interface dans `simulator/game.py` :
- `Game.advance(days)` fait avancer la simulation et renvoie les nouvelles observations ;
- `Game.next_bulletin()` avance jusqu'au prochain bulletin et renvoie les données, les informations révélées et les paliers débloqués ;
- `Game.decide(interventions)` valide la décision (palier débloqué, compatibilités, durée minimale) et la programme avec le délai de mise en œuvre ;
- l'état peut être sauvegardé et rechargé (JSON).

**Interfaces.**
- Une CLI `epidemic play` qui affiche les données et pose les questions à chaque bulletin hebdomadaire.
- Une page Streamlit `app/` : graphiques des données publiées, formulaire de décision à chaque bulletin, limité aux paliers débloqués, révélation progressive des informations épidémiologiques comme dans le cours.

**Tests.**
- Conservation de la population.
- Reproductibilité avec une graine fixe.
- Effet monotone des interventions : un confinement réduit l'incidence attendue.
- Règles de combinaison correctes.

**Point d'arrêt.**

---

## Phase 3 – Modèle SEIRH bayésien et inférence

**Modèle d'inférence** (`model/`) : plus simple que la réalité, mais adapté aux données.
- Compartiments : S, E, I_a, I_s, H, ICU, R, D, en temps discret avec `jax.lax.scan`. Le nom de travail est « SEIRH étendu ».
- **Structure par âge, au minimum 2 groupes** (< 60 ans et ≥ 60 ans), configurable jusqu'à 3 ou 4 groupes.
  - Matrice de contact entre groupes, construite à partir de `stats/*.json`.
  - Taux d'hospitalisation, de passage en soins intensifs et de létalité **propres à chaque groupe**. Priors hiérarchiques : les groupes partagent une tendance, mais les plus âgés ont un prior nettement plus élevé.
  - Sans cette structure, le modèle ne peut pas évaluer la distanciation sociale par âge.
  - Les données publiées n'étant pas ventilées par âge, la séparation entre groupes repose sur les priors et la matrice de contact : le dire clairement dans le rapport.
- Transmission variable dans le temps :
  - β(t) = β₀ × exp(marche aléatoire hebdomadaire), *ou* β constant par morceaux entre les dates d'intervention connues ;
  - implémente les deux et compare-les avec LOO / WAIC.
- **Paramètres stochastiques : option du modèle d'inférence**, cohérente avec les niveaux du simulateur.
  - *Variante A, par défaut* : seul β(t) varie dans le temps ; les autres paramètres sont fixes et inconnus.
  - *Variante B* : p_test(t) suit lui aussi une marche aléatoire, et la surdispersion des observations est inférée.
  - *Variante C* : dynamique latente stochastique (chaîne binomiale approchée par une loi normale, ou filtre particulaire avec PMMH). Plus coûteuse : à faire seulement si les variantes A et B sont mal calibrées sur les données de niveau 2 et 3.
  - Comparer les variantes par LOO et par la couverture en backtesting, puis justifier le choix dans le rapport.
- Priors **informatifs, justifiés dans le code et dans le rapport** à partir des consignes :
  - incubation 1-5 j ;
  - durée symptomatique 4-10 j ;
  - fraction symptomatique 50-70 % ;
  - sensibilité 70-85 % ;
  - taux de dépistage ≥ 50 % ;
  - priors faiblement informatifs sur les taux d'hospitalisation, de passage en soins intensifs et de létalité ;
  - condition initiale (E₀, I₀) à inférer.
- **Ne jamais confondre** infectés réels et cas confirmés. Le modèle infère les infections réelles (latentes) et relie les cas confirmés aux infections par un modèle d'observation explicite :
  - positifs attendus(t) = p_test × sensibilité × nouveaux symptomatiques(t) + admissions à l'hôpital non encore confirmées(t) ;
  - `num_tested`(t) = tests sur infectés + tests sur non-infectés (fond de syndromes grippaux, inféré). On utilise ainsi le **taux de positivité**, pas seulement le nombre de positifs ;
  - la spécificité vaut 1 par défaut, avec une variante à spécificité < 1 en analyse de sensibilité.
- Vraisemblance :
  - binomiale négative (surdispersion) sur `num_positive` ;
  - binomiale ou bêta-binomiale sur `num_positive` sachant `num_tested` ;
  - binomiale négative ou normale sur les échelles log pour `num_hospitalized` et `num_critical` ;
  - binomiale négative sur les incréments de `num_fatalities`.
- **Sorties à produire et à bien distinguer dans le rapport** :
  - infections réelles par jour ;
  - cas confirmés ;
  - **facteur de sous-déclaration** (infections / cas confirmés), avec son intervalle crédible ;
  - prévalence ;
  - nombre estimé de faux négatifs.

**Inférence** (`inference/`).
- NumPyro NUTS avec 4 chaînes, résultats sauvegardés en `InferenceData` ArviZ (`.nc`).
- Diagnostics automatiques : R̂ < 1,01, ESS, divergences, trace plots, rapport prior/posterior. **Échoue explicitement** si les diagnostics sont mauvais.
- Option : SVI pour aller vite pendant le développement.

**Validation : c'est ce qui rend le travail sérieux.**
1. **Retrouver les paramètres** : simule avec des paramètres connus, ajuste, et vérifie la couverture des intervalles crédibles à 90 %.
2. **Prior predictive check** : les trajectoires a priori sont-elles plausibles ?
3. **Posterior predictive check** sur toutes les séries observées.
4. **Backtesting en origine glissante** : ajuster jusqu'au jour t, prévoir sur 7, 14 et 21 jours, et mesurer le CRPS, la couverture des intervalles et le score de log-vraisemblance. Compare avec une référence naïve (persistance, croissance exponentielle).
5. Ajustement sur le vrai `data.csv` du cours (71 jours) comme cas réel.
6. **Ce qui est vraiment estimable** (identifiabilité). Montrer explicitement ce que les données permettent ou non d'apprendre :
   - **Courbe d'apprentissage** : réajuster le modèle à chaque bulletin (J7, J14, …), puis tracer la largeur de l'intervalle crédible de chaque paramètre en fonction du temps, ainsi que le rapport entre largeur a posteriori et largeur a priori.
   - **Paires de paramètres confondus** : graphes des lois a posteriori conjointes, par exemple taux de dépistage × taux d'hospitalisation, ou β × fraction symptomatique. On y voit les « crêtes » où les données ne permettent pas de trancher.
   - **Tableau de synthèse** : pour chaque paramètre, indiquer s'il est « bien identifié », « partiellement identifié » ou « dicté par le prior ».
   - Montrer que l'ajout des hospitalisations, puis du taux de positivité, réduit l'ambiguïté.

**Point d'arrêt.**

---

## Phase 4 – Projections, probabilités et aide à la décision

**Projections** (`forecast/`).
- Tirer des échantillons de la loi a posteriori, puis simuler les trajectoires futures **avec le bruit stochastique du processus**, pas seulement l'incertitude sur les paramètres.
- Graphiques en éventail (médiane, intervalles 50 % et 90 %) pour H, ICU, décès, positifs et Rt.

**Probabilités à calculer et afficher** pour chaque scénario :
- P(H dépasse 1 500 avant le 31 janvier) et P(ICU dépasse 300 avant le 31 janvier) ;
- P(saturation), c'est-à-dire H > 1 500 **ou** ICU > 300 ;
- la loi de la **date de première saturation** ;
- la loi du **pic** de H et d'ICU (valeur et date) ;
- la loi du nombre de **jours en saturation** ;
- P(Rt < 1) à la date de décision et à la fin ;
- les décès cumulés attendus, avec un intervalle crédible.

**Scénarios** (`decision/`).
- Énumérer les interventions possibles et leurs combinaisons valides, avec une grille sur les paramètres : durée d'isolement 7, 10 ou 14 jours ; âge seuil 50, 60 ou 70 ans.
- Traduire chaque intervention en facteur de réduction de β. Le calcul est fait à partir des matrices de contact par lieu (maison, école, travail, communauté) et des parts de transmission par lieu, et **l'incertitude sur ces parts est propagée**.
- Définir un **coût de contrainte** explicite et paramétrable dans le YAML. Par exemple : confinement > quarantaine du ménage > isolement des cas > distanciation > masque.
- **Recommandation** : l'intervention la moins coûteuse telle que P(saturation) ≤ α, avec α = 5 % par défaut et paramétrable. Afficher la frontière coût / risque (Pareto).
- Analyse de sensibilité : comment la recommandation change selon α, selon le taux de respect des mesures et selon les priors.

**Règles d'intervention conditionnelles** (`decision/policies.py`). En plus des mesures fixes, évaluer des **règles qui réagissent aux données observées**, comme un gouvernement réel :
- Format d'une règle : « activer X si l'indicateur dépasse un seuil haut ; désactiver si l'indicateur repasse sous un seuil bas pendant N jours ». Les deux seuils créent une hystérésis.
  - X : une intervention ou une combinaison ;
  - indicateur : H, ICU, cas sur 7 jours ou taux de positivité, **observés** ;
  - les règles tiennent compte de la durée minimale de 14 jours et du délai de mise en œuvre.
- Les règles sont évaluées **sur les trajectoires simulées à partir de la loi a posteriori**, en boucle fermée : la règle ne voit que les observations simulées, bruitées, jamais l'état caché. Elle subit donc les mêmes faux négatifs et la même sous-déclaration.
- Optimiser les seuils par recherche en grille ou par optimisation bayésienne : minimiser le coût de contrainte cumulé sous la contrainte P(saturation) ≤ α.
- Comparer sur un même graphe coût / risque les mesures fixes et les règles conditionnelles.
- Extension facultative : une version **à horizon glissant** (MPC). À chaque bulletin, on réoptimise la décision sur les 4 semaines suivantes à partir de la loi a posteriori mise à jour.

**Boucle avec le simulateur.** À chaque bulletin hebdomadaire, l'application Streamlit propose la recommandation bayésienne avec ses probabilités. L'utilisateur décide. Ensuite le simulateur révèle la suite, et on compare a posteriori la prévision avec la « réalité » (score de calibration).

**Graphes de suivi** (`viz/`). Ils sont présents dans chaque bulletin hebdomadaire, dans l'application et dans le rapport. Partout, les **cas confirmés** et les **infections réelles estimées** (avec leur intervalle crédible) sont tracés avec un code couleur constant. Les dates d'intervention sont marquées par des bandes verticales, et la date de publication du bulletin par un trait vertical.

1. **Incidence en échelle semi-log.**
   - Nouveaux cas par jour en échelle log, en fonction du temps : une phase exponentielle devient une droite.
   - Moyenne glissante sur 7 jours.
   - Pente = taux de croissance r, affiché avec le **temps de doublement** ou de **division par deux** (ln 2 / r) et son intervalle crédible.
2. **Taux d'attaque par personne exposée.**
   - Nouvelles infections / (S × I / N), en échelle log : c'est le **β(t) effectif**.
   - Avec Rt(t) = β(t) × S(t)/N × durée infectieuse.
   - Ce graphe montre directement l'effet de chaque intervention (marche descendante de β) et l'épuisement des susceptibles (S/N qui baisse).
3. **Trajectoire log-log.**
   - Nouveaux cas de la dernière semaine en fonction des cas cumulés, sur deux axes log, sans le temps.
   - Toute croissance exponentielle suit la même droite. Le décrochage vers le bas signale le ralentissement plus tôt que la courbe temporelle.
4. **Rt(t)** a posteriori, en éventail, avec la ligne Rt = 1, et **P(Rt < 1)** pour chaque semaine.
5. **Hôpital face à la capacité.**
   - H et ICU observés, puis projetés en éventail (50 % et 90 %).
   - Lignes horizontales aux seuils 1 500 et 300.
   - Courbe de **P(saturation avant la date t)** sous le graphe.
6. **Sous-déclaration** : infections réelles estimées / cas confirmés au cours du temps, et taux de positivité (`num_positive / num_tested`).
7. **Comparaison de scénarios** : petits multiples ou superposition des éventails H et ICU pour chaque intervention candidate, et le graphe coût / P(saturation) avec la frontière de Pareto.
8. **Diagnostics MCMC** : traces, R̂, ESS, densités prior/posterior superposées.

**Dessiner les modèles.** Deux schémas différents, à générer **automatiquement depuis le code** pour qu'ils restent synchronisés :

- **Graphe des compartiments** (`viz/model_diagram.py`) :
  - les flux S → E → I_p / I_a → I_s → H → ICU → R / D ;
  - chaque flèche est annotée par son taux (β, σ, γ, h, …) et chaque branche par sa probabilité (fraction symptomatique, fraction hospitalisée…) ;
  - la partie observation (test, sensibilité, positifs) est dessinée en pointillés ;
  - outil : Graphviz, ou Mermaid pour le README ;
  - une version pour le simulateur « réalité » et une pour le modèle d'inférence.
- **Modèle graphique bayésien** (DAG) :
  - généré avec `numpyro.render_model(..., render_distributions=True)` ;
  - il montre les hyperparamètres, les paramètres, les trajectoires latentes, les observations (nœuds grisés) et les plaques sur les jours ;
  - il est accompagné d'un tableau des priors (paramètre, loi, justification tirée des consignes).

**Point d'arrêt.**

---

## Phase 5 – Rendu professionnel

- **Rapport** `reports/report.qmd` en anglais, exporté en HTML et PDF :
  1. contexte ;
  2. données ;
  3. modèle (graphe des compartiments, équations, priors justifiés) ;
  4. inférence et diagnostics ;
  5. validation (paramètres retrouvés, backtesting, identifiabilité, robustesse selon le niveau de stochasticité) ;
  6. projections ;
  7. probabilités de saturation par scénario ;
  8. recommandation (mesures fixes et règles conditionnelles) et sensibilité ;
  9. calibration en fin de partie ;
  10. limites ;
  11. reproductibilité.
- **Documentation du framework bayésien : livrable prioritaire.** Deux fichiers markdown en anglais, **générés par le projet** avec `make docs`. Ce sont des gabarits Jinja2 (`docs/templates/*.md.j2`) remplis à partir des configs et des résultats (`.nc`, scores). Ainsi, tous les chiffres, tableaux de priors et figures restent synchronisés avec le code. Aucun chiffre n'est tapé à la main.

  **1. `docs/bayesian-framework.md`** : document long, à lire, pédagogique mais rigoureux. Il doit pouvoir être compris par quelqu'un qui connaît les probabilités mais pas l'épidémiologie. Chaque section suit la même progression : intuition, puis formalisation (équations LaTeX), puis choix fait dans le projet et justification, puis ce qu'on en a tiré (figure ou chiffre généré). Contenu minimal :
  1. **Pourquoi bayésien** : quantifier l'incertitude, intégrer les connaissances des consignes, produire des probabilités utiles à la décision, comme P(saturation). Comparaison avec l'approche par moindres carrés du rapport de 2020 (`proj0016-covid/Modelling-COVID-20`).
  2. **Le problème** : réalité cachée, données partielles et bruitées. Distinction entre infectés, testés et confirmés ; sous-déclaration ; faux négatifs.
  3. **Le modèle génératif**, de bout en bout :
     - graphe des compartiments et équations de la dynamique ;
     - structure par âge ;
     - β(t) et ses variantes ;
     - modèle d'observation (dépistage, sensibilité, tests sur non-infectés) ;
     - vraisemblances choisies et pourquoi (binomiale négative, surdispersion) ;
     - le DAG NumPyro.
  4. **Les priors, un par un** : tableau généré (paramètre, signification, loi, paramètres, intervalle à 95 %, source dans les consignes ou hypothèse) et graphe de chaque loi a priori. Pour chacun : pourquoi cette famille de lois (Beta pour une proportion, LogNormal ou Gamma pour une durée…), comment les bornes des consignes ont été traduites en paramètres, et ce qui est informatif ou faiblement informatif. Puis la **prior predictive check** : à quoi ressemblent les épidémies que le modèle juge plausibles avant d'avoir vu les données ?
  5. **L'inférence** :
     - le théorème de Bayes appliqué ici ;
     - pourquoi la loi a posteriori n'a pas de forme explicite ;
     - MCMC, puis Hamiltonian Monte Carlo et NUTS, expliqués intuitivement ;
     - pourquoi JAX/NumPyro (gradients automatiques à travers la simulation) ;
     - les reparamétrisations utilisées (non centrée, échelle log) ;
     - SVI en option.
  6. **Diagnostics** : R̂, ESS, divergences, traces. Comment les lire, ce qu'on a obtenu, ce qu'on a corrigé.
  7. **Ce que les données ont appris** : prior face à posterior pour chaque paramètre, courbe d'apprentissage semaine après semaine, paramètres confondus et identifiabilité.
  8. **Vérifications** : posterior predictive check, retrouver des paramètres connus, backtesting, robustesse selon le niveau de stochasticité.
  9. **De la loi a posteriori à la décision** :
     - loi prédictive a posteriori ;
     - incertitude sur les paramètres et bruit du processus ;
     - calcul de P(saturation), de la date et du pic ;
     - traduction des interventions en réduction de β ;
     - règle de décision (coût minimal sous P(saturation) ≤ α), règles conditionnelles, analyse de sensibilité.
  10. **Calibration** : les probabilités annoncées se sont-elles réalisées ? Diagramme de fiabilité, Brier, CRPS.
  11. **Limites et hypothèses** : reprend `docs/hypotheses.md`, avec les écarts entre le modèle et la réalité simulée.
  12. **Glossary** et **références** (Chowell 2017, Gelman et al. *Bayesian Workflow* 2020, documentation NumPyro, simulateur IISc…).

  **2. Deux présentations Marp, pour deux publics différents** : `docs/slides/recruiter.md` et `docs/slides/technical.md`, exportées en PDF et HTML avec `make slides`.

  > **Attention : les présentations sont le point faible habituel.** Je ne veux pas de « slop » : murs de puces, jargon, titres vides comme « Overview » ou « Results », phrases génériques qui pourraient s'appliquer à n'importe quel projet. Le plan ci-dessous est **imposé**. Ne l'invente pas, ne l'enrichis pas, remplis-le.

  **Règles communes, vérifiées automatiquement par `scripts/lint_slides.py` dans la CI.**
  - **Le titre de chaque diapositive est une phrase-conclusion** (sujet, verbe, chiffre si possible), par exemple « Without action, hospitals saturate with 87% probability by Dec 20 ». Jamais un mot-étiquette comme « Results ». Le linter refuse les titres de moins de 5 mots.
  - **Une seule idée par diapositive.** Au maximum **40 mots** de texte hors titre pour la présentation recruteur, **60 mots** pour la technique.
  - **Au maximum 3 puces**, et pas de puces imbriquées.
  - **Chaque diapositive porte une figure générée ou un chiffre clé en grand**, sauf le titre et la conclusion. Pas d'image décorative, pas d'icône, pas d'emoji.
  - **Tous les chiffres viennent des résultats**, injectés par le gabarit Jinja2. Aucun chiffre tapé à la main.
  - **Mots interdits**, refusés par le linter : *leverage, robust, cutting-edge, seamless, delve, unlock, harness, powerful, comprehensive, state-of-the-art, journey, game-changer*, ainsi que « In this presentation », « Let's dive in », « Key takeaways » utilisé comme titre.
  - **Figures lisibles en projection** : police ≥ 18 pt, au maximum 3 séries par graphe, légende directement sur les courbes plutôt que dans une boîte, palette cohérente avec le reste du projet (confirmés et estimés toujours de la même couleur).
  - **Thème Marp sobre**, défini une fois dans `docs/slides/theme.css` : fond blanc, une couleur d'accent, numéro de diapositive en bas.
  - **Notes de l'orateur** (`<!-- ... -->` sous chaque diapositive) : 2 à 4 phrases de ce que je dois **dire**, qui ne répètent pas l'écran.

  **2a. `recruiter.md` : 7 diapositives, 5 minutes, public non technique** (recruteur, manager). Aucune formule, aucun nom de loi de probabilité, aucun nom de bibliothèque sauf sur la dernière diapositive. Le vocabulaire est celui d'une personne qui doit **prendre une décision**.

  | # | Titre (modèle à remplir) | Contenu |
  |---|---|---|
  | 1 | Deciding under uncertainty: steering a simulated epidemic with Bayesian forecasts | nom, date, lien vers l'application |
  | 2 | A city of 1 million, incomplete data, and a decision every week | 3 éléments du problème : on ne voit que les cas testés, les tests se trompent, il faut décider avant de savoir |
  | 3 | Confirmed cases show only {{ ratio }}% of real infections | graphe confirmés face aux infections estimées |
  | 4 | Without action, P(hospital saturation) = {{ p_sat_none }}% | graphe hôpital face à la capacité, en éventail |
  | 5 | {{ recommended }} keeps the risk below {{ alpha }}% at the lowest social cost | graphe coût face au risque avec la recommandation mise en évidence |
  | 6 | The forecasts were honest: {{ coverage }}% of outcomes fell inside the 90% ranges | une barre « annoncé / réalisé » |
  | 7 | What this project shows about how I work | 3 puces sur les compétences démontrées (modélisation, quantification de l'incertitude, aide à la décision) et la pile technique en une ligne |

  **2b. `technical.md` : 12 diapositives, 15 minutes, public data scientist ou jury.**

  | # | Titre (modèle à remplir) | Contenu |
  |---|---|---|
  | 1 | Bayesian SEIRH inference and decision support for a simulated epidemic | titre, lien vers le dépôt |
  | 2 | The analyst sees only noisy, partial observations of a hidden process | schéma réalité cachée → test → données publiées |
  | 3 | The generative model: age-structured SEIRH with an explicit testing layer | graphe des compartiments généré |
  | 4 | Priors encode the brief, not guesses | tableau des priors en 6 lignes maximum (les plus importants) ; renvoi vers `bayesian-framework.md` pour la suite |
  | 5 | NUTS converged: R̂ < {{ rhat_max }}, {{ n_div }} divergences | trace plot et ESS minimal |
  | 6 | The data pin down {{ n_identified }} of {{ n_params }} parameters | prior face à posterior, avec en couleur les paramètres bien identifiés |
  | 7 | Testing rate and hospitalisation rate remain confounded | graphe de la loi a posteriori conjointe des deux paramètres |
  | 8 | Posterior predictive checks reproduce all {{ n_series }} observed series | graphe des données observées dans l'éventail prédictif |
  | 9 | Forecasts beat naive baselines by {{ crps_gain }}% CRPS at 14 days | graphe en barres : CRPS du modèle face aux références |
  | 10 | Adaptive triggers dominate fixed policies on the cost–risk frontier | frontière de Pareto : mesures fixes face aux règles conditionnelles |
  | 11 | Calibration holds up to stochasticity level {{ max_ok_level }} | diagramme de fiabilité par niveau |
  | 12 | Limits and next steps | 3 limites concrètes, chacune avec la piste correspondante |

  Si un résultat contredit un titre (par exemple si les règles conditionnelles ne dominent pas), **réécris le titre pour dire ce que montrent les données**. Ne force jamais le résultat.

  **Contrôle qualité avant de me rendre les slides :**
  1. `make slides`, puis exporter chaque diapositive en PNG (`marp --images png`).
  2. **Regarde chaque PNG** et vérifie : le texte ne déborde pas, la figure est lisible, le titre se comprend sans le reste de la diapositive.
  3. **Test de lecture des titres seuls** : génère `docs/slides/*_titles.txt` avec uniquement les titres. Lus à la suite, ils doivent raconter toute l'histoire. Montre-les-moi.
  4. Je valide les titres **avant** que tu finalises les figures.

  Tests : `make docs` doit tourner dans la CI ; un test vérifie qu'aucun champ du gabarit n'est resté vide et que toutes les figures référencées existent.

  **À rédiger au fil de l'eau** : à la fin de chaque phase 2, 3 et 4, mettre à jour les sections correspondantes du gabarit, sans attendre la phase 5.
- **Bulletin épidémiologique hebdomadaire en PDF** (`reports/bulletin/`) :
  - généré automatiquement à chaque bulletin du jeu par `epidemic bulletin --day N`, à partir d'un gabarit Quarto ;
  - 2 pages maximum, dans le style d'un bulletin d'agence de santé publique (par exemple Sciensano) :
    - un encadré « Situation at a glance » de 3 à 5 phrases ;
    - les indicateurs clés avec leur tendance par rapport à la semaine précédente (cas, positivité, H, ICU, décès, Rt) ;
    - les graphes de suivi essentiels (semi-log, hôpital face à la capacité, Rt) ;
    - les projections à 14 et 28 jours ;
    - P(saturation) ;
    - la recommandation et les paliers débloqués ;
  - les probabilités sont **traduites en langage clair** selon l'échelle verbale de l'IPCC (*calibrated uncertainty language*), avec le chiffre à côté :

    | Probabilité | Formulation |
    |---|---|
    | ≥ 99 % | virtually certain |
    | 90-99 % | very likely |
    | 66-90 % | likely |
    | 33-66 % | about as likely as not |
    | 10-33 % | unlikely |
    | 1-10 % | very unlikely |
    | < 1 % | exceptionally unlikely |

  - les phrases sont générées par des règles à partir des chiffres, sans LLM, pour que le bulletin reste reproductible ;
  - les mots *estimated* ou *confirmed* sont toujours précisés.
- **Tableau de calibration en fin de partie** (`forecast/scoring.py`), affiché à la fin du jeu et dans le rapport :
  - pour chaque bulletin, chaque horizon (7, 14 et 28 jours) et chaque série (positifs, H, ICU, décès) : la valeur réalisée tombe-t-elle dans l'intervalle à 50 % et à 90 % ?
  - **couverture empirique** par rapport à la couverture nominale, par exemple « intervalles à 90 % : 85 % de réussite sur 48 prévisions » ;
  - **diagramme de fiabilité** : probabilités annoncées (P(H > seuil), P(saturation)…) face aux fréquences observées ;
  - **score de Brier** pour les événements de saturation et **CRPS** pour les séries ;
  - comparaison avec les références naïves ;
  - ventilation par niveau de stochasticité du simulateur ;
  - un court texte d'interprétation : le modèle est-il trop sûr de lui, ou trop prudent ?
- **README** : badge CI, installation en 3 commandes, `make all` qui reproduit tous les résultats, captures de l'application.
- **Application Streamlit** déployable : deux modes, « Jouer » (le simulateur avec décisions) et « Analyser » (ajustement du modèle sur des données chargées).
- **Qualité** :
  - typage complet et docstrings en anglais ;
  - aucune valeur codée en dur : tout passe par `configs/` ;
  - graines fixées partout ;
  - couverture de tests ≥ 80 % sur `simulator/` et `model/`.
- **Commits** : petits et lisibles, en Conventional Commits, sur la branche `bayesian-rewrite`. Ouvre une PR vers `main` à la fin, avec un résumé.

---

## Règles générales

- **Étanchéité totale** entre `simulator/` (la réalité) et `model/` / `forecast/` (ce que voit l'analyste). Le modèle ne lit jamais `.truth.json` ni l'état interne du simulateur, sauf dans les tests de validation explicitement marqués.
- Toute hypothèse non tirée des consignes doit être écrite dans `docs/hypotheses.md`.
- Si un choix de modélisation est ambigu, propose 2 options avec leurs avantages et leurs inconvénients, et demande-moi avant d'implémenter.
- Préfère un code simple, testé et compris à un code exhaustif.
