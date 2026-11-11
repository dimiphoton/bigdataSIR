# Hypothèses du simulateur

Ce fichier liste ce qui n'est pas fixé par les consignes du cours. Le futur modèle d'inférence ne lit ni ce simulateur, ni `data/generated/.truth.json`.

## Structure

La ville est découpée en cinq tranches : 0-17, 18-49, 50-59, 60-69 et 70 ans et plus. Les seuils 50, 60 et 70 tombent sur des bords de tranche. L'âge 80 du fichier `age.json` regroupe les 80 ans et plus ; il est traité comme l'âge 80.

L'école concerne les 3-17 ans, avec une participation de 95 %. Le travail concerne les 18-64 ans, avec une participation de 70 %. Ces deux taux sont dans `configs/simulator.yaml`.

## Contacts

Les matrices viennent de `stats/*.json`.

- Domicile : une personne a en moyenne autant de contacts que de cohabitants. Les âges se mélangent au hasard dans le ménage. Il n'y a pas d'assortativité par âge.
- École, travail, communauté : le degré vaut `min(plafond, taille du groupe - 1)`, pondéré par les personnes. Les écoles et les quartiers sont grands, donc le plafond est presque toujours atteint. Les plafonds (15, 15 et 4) représentent des contacts effectifs par jour, pas la taille du groupe.
- Un poids par lieu module la transmission : domicile 1, école 0,45, travail 0,45, communauté 0,25.

## Compartiments et durées

Les consignes donnent l'incubation (1 à 5 jours) et la période symptomatique avant hospitalisation (4 à 10 jours). Le reste est choisi ici, puis tiré dans une plage :

- présymptomatique infectieux : 1 à 3 jours ;
- asymptomatique infectieux : même plage que la période symptomatique, avec une contagiosité relative de 0,4 à 0,8 ;
- séjour en hôpital : 6 à 12 jours ;
- séjour en soins intensifs : 7 à 14 jours ;
- pas de transmission depuis l'hôpital ni les soins intensifs.

La gravité par âge (hospitalisation, passage en soins intensifs, décès) est dans `configs/simulator.yaml`. Elle est choisie pour qu'une vague puisse approcher 1 500 hospitalisés et 300 lits de soins intensifs, sans que ce soit automatique.

En temps discret, la probabilité de quitter un stade est `1 - exp(-durée^-1)`. La durée moyenne est un peu plus longue qu'une loi géométrique de paramètre `1 / durée`. Au niveau 3, chaque durée est une somme de stades gamma de même moyenne.

## Aléa

Quatre générateurs partent de la même graine : paramètres, démographie, environnement, individus. Une graine fixe donne la même trajectoire pour un même niveau. Les paramètres secrets tirés dans les plages des consignes ne dépendent pas du niveau.

- Niveau 1 : chaîne binomiale quotidienne.
- Niveau 2 : en plus, β(t) suit un Ornstein-Uhlenbeck sur log β (volatilité 5 % par jour, retour vers la moyenne), le taux de test et le fond grippal dérivent, et la sensibilité varie un peu d'un jour à l'autre. La saisonnalité est éteinte par défaut.
- Niveau 3 : en plus, la contagiosité agrégée suit une gamma de dispersion k entre 0,1 et 0,5, le respect des mesures varie autour de 90 %, et les durées deviennent gamma.

Chaque source peut être forcée à vrai ou faux dans le YAML. `null` suit le niveau.

## Observation

Trois états de surveillance : non testé, testé négatif, testé positif. Seuls les nouveaux symptomatiques sont testés, avec une probabilité tirée entre 50 et 85 %. Les asymptomatiques et les présymptomatiques ne le sont pas, sauf si `extended_screening_rate` est supérieur à 0. Il vaut 0.

Un faux négatif reste infectieux et ne déclenche pas l'isolement. Une admission jamais confirmée est comptée positive le jour de l'entrée, et compte aussi comme un test. La spécificité vaut 100 % dans les priors ; on peut la baisser dans `configs/priors.yaml`. Le plafond de tests journaliers est 20 000. Les tests hors hôpital sont réduits s'il est atteint. Les admissions restent comptées.

## Interventions

Le respect de 90 % réduit l'effet moyen des mesures générales : un confinement de −75 % au travail retire en moyenne 67,5 % des contacts. Pour une personne qui s'isole vraiment, la réduction est la plus forte entre la mesure générale et l'isolement, appliquée une seule fois au contact. Le masque multiplie le taux d'infection par `1 - 0,9 × 0,20`. Il ne passe pas par la règle du maximum.

L'isolement et la quarantaine portent sur les cas confirmés, pas sur les faux négatifs. La durée demandée (7, 10, 14 jours…) est comparée à la durée symptomatique : la part des symptomatiques confirmés qui s'isolent vaut `min(1, durée / durée symptomatique)`. On ne garde pas un ménage en quarantaine après la guérison de l'index. La quarantaine d'un ménage touche aussi une fraction des autres personnes, estimée par `cas confirmés × (taille moyenne du ménage - 1) / population`.

`isolate_on_symptoms` est faux : un symptomatique non testé ne s'isole pas tout seul.

## Calendrier et score

Le bulletin du jour 70 prend effet le jour 75, et non trois jours plus tard. Les autres bulletins gardent le délai de 3 jours. Une mesure peut être levée à un bulletin une fois le jour courant au moins égal à son quatorzième jour d'effet. La levée prend effet après le même délai.

Le score additionne, pour chaque jour, le coût de chaque mesure active. Les coûts sont dans `configs/game.yaml`. Ils ne viennent pas des consignes. Les effets épidémiologiques, eux, ne s'additionnent pas : c'est la plus forte réduction qui s'applique, puis le masque.

La saturation compare `num_hospitalised` à 1 500 et `num_critical` à 300, comme les colonnes publiées. Les hospitalisés hors soins intensifs et les soins intensifs sont donc deux stocks séparés.

Le déblocage anticipé des paliers 2 et 3 utilise les probabilités du bulletin quand un posterior est chargé dans l'application. Sans ce fichier, seules les dates du calendrier débloquent les paliers.

## Projections et décision

Le modèle d'inférence a un seul β. Une intervention est traduite en matrice de contact, construite à partir des quatre lieux. Cette matrice est le rapport entre la politique choisie et la politique déjà en cours : β(t) porte déjà l'effet des mesures passées, et on ne le compte pas une seconde fois.

L'isolement des cas et la quarantaine du ménage ne concernent qu'une part des infectieux. Au point de référence, cette part vaut fraction symptomatique 0,60 × probabilité de confirmation 0,55 × min(1, durée / 10 jours). Dix jours est le haut de `symptomatic_before_hospital_days` dans les priors : 7 jours ne couvrent pas toute la plage, 10 et 14 jours la couvrent entièrement. La quarantaine multiplie par la taille moyenne du ménage, puis on tronque à 1. Ces trois nombres, et les poids de transmission des lieux, ne sont pas dans les consignes. On les fait varier : Dirichlet de concentration 40 autour des poids de `configs/simulator.yaml`, et lois normales autour du point de référence (`configs/decision.yaml`). Le rapport des facteurs multiplie β, une fois la matrice moyenne fixée.

Le coût utilisé pour classer est un coût cumulé. Pendant le délai de mise en œuvre, la politique en cours continue de coûter. Ensuite, c'est le paquet choisi. Les coûts journaliers sont ceux de `configs/game.yaml`. La durée d'isolement et l'âge seuil changent l'effet, pas le coût d'un jour.

Les jours futurs des variantes A et B reçoivent un bruit normal d'échelle binomiale sur les nouvelles infections, même si `process_sigma` vaut 0 dans l'ajustement. Les chocs des jours déjà ajustés restent nuls. Le taux de dépistage est prolongé à sa dernière valeur a posteriori.

Une règle conditionnelle est évaluée en boucle fermée. À chaque jour elle ne voit que des observations bruitées (hospitalisés, soins intensifs, cas sur 7 jours, positivité sur 7 jours). Le franchissement du seuil haut programme l'activation après le délai du jeu. La levée demande que l'indicateur reste sous le seuil bas pendant N jours, et que la mesure ait déjà duré au moins 14 jours. La grille par défaut ne fait varier que les seuils d'hospitalisés, pour que le calcul reste court. Les autres indicateurs sont dans le YAML.

P(saturation) compare les stocks bruités aux plafonds 1 500 et 300, parce que ce sont les séries publiées qui définissent la saturation. Les jours déjà publiés sont recollés tels quels. Rt affiché reste le proxy β × contacts × durée infectieuse × S/N.

La sensibilité à alpha reclasse les mêmes trajectoires. La sensibilité au taux de respect et à la fraction symptomatique est rapportée par le facteur de transmission. On ne relance pas un MCMC à chaque changement de prior.


## Mode cours

Avec `observation_source: course`, les 71 jours de `data/external/data.csv` remplacent les observations publiées. L'état caché continue d'être simulé et ne coïncide pas avec ce fichier. Après le jour 71, les observations redeviennent celles du simulateur.

## Hypothèses du modèle d'inférence

Le modèle d'inférence est plus simple que la ville. Il n'a pas de stade présymptomatique, pas de lieux séparés, et pas de chaîne binomiale dans les variantes A et B. Les contacts des quatre lieux sont additionnés, avec leurs poids, en une seule matrice entre groupes d'âge. Cette matrice vient des statistiques publiques. Le modèle ne lit pas `.truth.json`.

Les données publiées ne sont pas ventilées par âge. La séparation entre groupes repose sur les priors et sur la matrice de contact, pas sur une observation directe.

Les consignes fixent des intervalles, pas des familles de lois. L'incubation (1 à 5 jours) et la durée symptomatique (4 à 10 jours) sont des log-normales dont l'intervalle central à 90 % coïncide avec ces bornes. La fraction symptomatique (50 à 70 %) et la sensibilité (70 à 85 %) sont des lois Beta de même moyenne, avec un écart-type calé sur la largeur de l'intervalle. Le taux de dépistage vaut `0,50 + 0,50 × Beta(2, 2)` : le plancher de 50 % est une consigne, la forme au-dessus est peu informative.

La spécificité vaut 1, comme dans `configs/priors.yaml`. On peut l'estimer avec une Beta(199, 1), proche de 1, seulement si `estimate_specificity` est vrai.

Ce qui n'est pas dans les consignes :

- β₀ est une log-normale de médiane 0,03 par contact pondéré, pour un R de l'ordre de 1. Pour que l'optimiseur reste stable, les exponentielles utilisées dans la dynamique sont bornées (β₀ entre 0,0001 et 0,5, durées entre une fraction de jour et quelques dizaines de jours). Ces bornes sont loin des priors.
- La marche aléatoire hebdomadaire de log β a une volatilité demi-normale d'échelle 0,15. En mode constant par morceaux, l'échelle du saut est 0,8.
- Les asymptomatiques ont la même durée infectieuse que les symptomatiques, et une contagiosité relative Beta(2, 2).
- Le séjour en hôpital a une médiane de 8 jours, le séjour en soins intensifs une médiane de 10 jours, tous deux en log-normale d'écart 0,35.
- L'hôpital et les soins intensifs ne transmettent pas.
- La gravité (hospitalisation, passage en soins intensifs, décès) est une logit-normale hiérarchique, non centrée. La moyenne est faible. Une pente d'âge strictement positive, de médiane 1,5 sur l'échelle logit, rend le prior des plus âgés nettement plus élevé. L'intensité de cette pente est un choix, pas une consigne chiffrée.
- Au jour 0, seules des personnes exposées et infectieuses sont présentes. Personne n'est encore à l'hôpital. Le germe est réparti comme la population.
- Le fond de syndromes grippaux est une log-normale de médiane 300 tests par jour.
- Variante A : la concentration de la binomiale négative est fixée à 100. Variante B : elle est estimée, une par série, log-normale de médiane 30.
- Variante C : le bruit sur les nouvelles infections est normal, d'écart-type proportionnel à celui d'une binomiale. Le coefficient est une demi-normale d'échelle 1. Ce n'est pas un filtre particulaire. Le PMMH n'est pas lancé tant que la couverture des variantes A et B n'a pas été jugée mauvaise sur des NUTS complets.
- Un test est immédiat le jour où la personne devient symptomatique. Il n'y a pas de délai de rendu.
- La vraisemblance par défaut est une binomiale négative sur les positifs et une autre sur les tests. Le taux de positivité est donc dans le rapport des deux moyennes. Une bêta-binomiale des positifs sachant les tests est disponible, mais elle remplace la loi sur les positifs : les mêmes cas ne sont pas comptés deux fois. Les stocks d'hôpital et de soins intensifs sont des binomiales négatives, ou des normales sur log(1 + stock). Les nouveaux décès sont une binomiale négative.
- Le facteur de sous-déclaration du modèle est le rapport infections réelles / positifs attendus. Le rapport aux positifs observés se calcule après l'ajustement.
- Les seuils d'identifiabilité (rapport de largeurs 0,50 et 0,85) sont des conventions de lecture, dans `configs/inference.yaml`.
- Rt affiché par le modèle est un proxy : β × contacts moyens × durée infectieuse pondérée × part de susceptibles. Ce n'est pas le rayon spectral.

Les deux formes de β(t), marche hebdomadaire et paliers entre dates, sont toutes les deux dans le code. Leur comparaison LOO / WAIC demande des ajustements NUTS, parce que le SVI ne garantit pas la log-vraisemblance point par point.
