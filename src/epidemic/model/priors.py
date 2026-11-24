"""Priors du modèle d'inférence, reliés aux consignes du cours.

Les bornes chiffrées viennent de ``configs/priors.yaml``. Ici on choisit la
famille de lois et on traduit l'intervalle en paramètres. Ce qui n'est pas
dans les consignes est marqué comme hypothèse.
"""

import math
from dataclasses import dataclass

# Quantile d'une loi normale : P(|Z| < z) = 0,90.
_Z90 = 1.6448536269514722
# Même idée pour un intervalle à 95 %, utilisé pour caler une Beta sur une plage.
_Z95 = 1.959963984540054


def lognormal_on_interval(low: float, high: float) -> tuple[float, float]:
    """Loi log-normale dont l'intervalle central à 90 % vaut [low, high]."""
    if low <= 0 or high <= low:
        raise ValueError("L'intervalle doit être strictement positif et ordonné.")
    mu = 0.5 * (math.log(low) + math.log(high))
    sigma = (math.log(high) - math.log(low)) / (2.0 * _Z90)
    return mu, sigma


def beta_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    """Paramètres alpha, beta d'une loi Beta de moyenne et d'écart-type donnés."""
    if not 0.0 < mean < 1.0:
        raise ValueError("La moyenne d'une proportion doit être entre 0 et 1.")
    variance = sd**2
    maximum = mean * (1.0 - mean)
    if variance <= 0 or variance >= maximum:
        raise ValueError("Cet écart-type n'est pas compatible avec la moyenne.")
    # kappa = alpha + beta. Il vient de var = m(1-m) / (kappa + 1).
    kappa = mean * (1.0 - mean) / variance - 1.0
    return mean * kappa, (1.0 - mean) * kappa


def logit(probability: float) -> float:
    """Logit d'une probabilité, pour écrire la moyenne d'un prior gaussien."""
    if not 0.0 < probability < 1.0:
        raise ValueError("Le logit n'est défini qu'entre 0 et 1.")
    return math.log(probability) - math.log(1.0 - probability)


# Incubation 1 à 5 jours, période symptomatique 4 à 10 jours : consignes.
INCUBATION_MU, INCUBATION_SIGMA = lognormal_on_interval(1.0, 5.0)
SYMPTOMATIC_MU, SYMPTOMATIC_SIGMA = lognormal_on_interval(4.0, 10.0)

# Fraction symptomatique 50 à 70 %. L'écart-type est celui d'une normale
# qui tiendrait la même plage à 95 %.
SYMPTOM_ALPHA, SYMPTOM_BETA = beta_from_mean_sd(0.60, 0.10 / _Z95)
# Sensibilité 70 à 85 %. Même traduction.
SENSITIVITY_ALPHA, SENSITIVITY_BETA = beta_from_mean_sd(0.775, 0.075 / _Z95)

# β par contact pondéré. L'échelle vient de l'ordre de grandeur des contacts
# de la ville (quelques contacts effectifs par jour) pour un R proche de 1.
BETA_MU = math.log(0.03)
BETA_SIGMA = 0.6
WEEKLY_BETA_SCALE = 0.15
PIECEWISE_BETA_SCALE = 0.8

# Hypothèses de durée : séjour hospitalier et soins intensifs, hors consignes.
HOSPITAL_MU = math.log(8.0)
HOSPITAL_SIGMA = 0.35
ICU_MU = math.log(10.0)
ICU_SIGMA = 0.35

# Gravité : moyenne faible, et une pente d'âge strictement positive.
HOSP_MU_LOC = logit(0.02)
ICU_MU_LOC = logit(0.15)
DEATH_WARD_MU_LOC = logit(0.02)
DEATH_ICU_MU_LOC = logit(0.25)
SEVERITY_MU_SCALE = 1.0
AGE_SLOPE_MU = math.log(1.5)
AGE_SLOPE_SIGMA = 0.35
SEVERITY_TAU_SCALE = 0.4

# Condition initiale : quelques dizaines de personnes déjà exposées.
E0_MU = math.log(40.0)
E0_SIGMA = 0.8

# Fond de tests sur des personnes non infectées. Hypothèse, pas une consigne.
FLU_MU = math.log(300.0)
FLU_SIGMA = 0.6

# Surdispersion de la variante B : concentration d'une binomiale négative.
CONCENTRATION_MU = math.log(30.0)
CONCENTRATION_SIGMA = 0.7

# Variante C : le choc est de l'ordre de l'écart-type binomial.
PROCESS_SIGMA_SCALE = 1.0


@dataclass(frozen=True)
class PriorRow:
    """Une ligne du tableau des priors, en français."""

    label: str
    family: str
    parameters: str
    source: str
    justification: str


def prior_catalogue() -> tuple[PriorRow, ...]:
    """Tableau des priors. Les chiffres viennent des constantes ci-dessus."""
    inc_mu, inc_sd = INCUBATION_MU, INCUBATION_SIGMA
    sym_mu, sym_sd = SYMPTOMATIC_MU, SYMPTOMATIC_SIGMA
    return (
        PriorRow(
            "Durée d'incubation",
            "Log-normale",
            f"μ = {inc_mu:.2f}, σ = {inc_sd:.2f}",
            "consigne",
            "L'intervalle central à 90 % vaut 1 à 5 jours.",
        ),
        PriorRow(
            "Durée symptomatique",
            "Log-normale",
            f"μ = {sym_mu:.2f}, σ = {sym_sd:.2f}",
            "consigne",
            "L'intervalle central à 90 % vaut 4 à 10 jours, avant l'hôpital.",
        ),
        PriorRow(
            "Fraction symptomatique",
            "Beta",
            f"α = {SYMPTOM_ALPHA:.1f}, β = {SYMPTOM_BETA:.1f}",
            "consigne",
            "Moyenne 0,60, pour couvrir 50 à 70 % des infectés.",
        ),
        PriorRow(
            "Sensibilité des tests",
            "Beta",
            f"α = {SENSITIVITY_ALPHA:.1f}, β = {SENSITIVITY_BETA:.1f}",
            "consigne",
            "Moyenne 0,775, pour couvrir 70 à 85 %.",
        ),
        PriorRow(
            "Taux de dépistage",
            "Beta décalée",
            "0,50 + 0,50 × Beta(2, 2)",
            "consigne",
            "Plancher de 50 % (consigne). La forme au-dessus est peu informative.",
        ),
        PriorRow(
            "Spécificité des tests",
            "Fixée, ou Beta",
            "1, ou Beta(199, 1) si on l'estime",
            "consigne",
            "Les consignes donnent 100 %. L'estimation est une analyse de sensibilité.",
        ),
        PriorRow(
            "Transmission β₀",
            "Log-normale",
            f"μ = {BETA_MU:.2f}, σ = {BETA_SIGMA:.2f}",
            "hypothèse",
            "Par contact pondéré. La médiane 0,03 vise un R de l'ordre de 1.",
        ),
        PriorRow(
            "Volatilité hebdomadaire de β",
            "Demi-normale",
            f"échelle {WEEKLY_BETA_SCALE}",
            "hypothèse",
            "La marche aléatoire hebdomadaire reste modérée.",
        ),
        PriorRow(
            "Saut de β entre deux dates",
            "Demi-normale",
            f"échelle {PIECEWISE_BETA_SCALE}",
            "hypothèse",
            "En mode constant par morceaux, un changement peut être fort.",
        ),
        PriorRow(
            "Contagiosité des asymptomatiques",
            "Beta",
            "α = 2, β = 2",
            "hypothèse",
            "Relative aux symptomatiques, moyenne 0,5, sans borne dans les consignes.",
        ),
        PriorRow(
            "Durée d'hospitalisation",
            "Log-normale",
            f"médiane 8 j, σ = {HOSPITAL_SIGMA}",
            "hypothèse",
            "Les consignes ne donnent pas la durée de séjour.",
        ),
        PriorRow(
            "Durée en soins intensifs",
            "Log-normale",
            f"médiane 10 j, σ = {ICU_SIGMA}",
            "hypothèse",
            "Les consignes ne donnent pas la durée de séjour.",
        ),
        PriorRow(
            "Probabilité d'hospitalisation",
            "Logit-normale hiérarchique",
            f"moyenne logit {HOSP_MU_LOC:.2f}, échelle {SEVERITY_MU_SCALE}",
            "hypothèse",
            "Autour de 2 % chez les plus jeunes. Pas de chiffre dans les consignes.",
        ),
        PriorRow(
            "Pente d'âge de la gravité",
            "Log-normale",
            f"μ = {AGE_SLOPE_MU:.2f}, σ = {AGE_SLOPE_SIGMA}",
            "consigne du projet",
            "Les plus âgés ont un prior plus élevé. La pente est positive.",
        ),
        PriorRow(
            "Passage en soins intensifs",
            "Logit-normale hiérarchique",
            f"moyenne logit {ICU_MU_LOC:.2f}",
            "hypothèse",
            "Parmi les sorties d'hôpital, autour de 15 %, même pente d'âge.",
        ),
        PriorRow(
            "Décès en service",
            "Logit-normale hiérarchique",
            f"moyenne logit {DEATH_WARD_MU_LOC:.2f}",
            "hypothèse",
            "Parmi les sorties d'hôpital qui ne vont pas en soins intensifs.",
        ),
        PriorRow(
            "Décès en soins intensifs",
            "Logit-normale hiérarchique",
            f"moyenne logit {DEATH_ICU_MU_LOC:.2f}",
            "hypothèse",
            "Parmi les sorties de soins intensifs.",
        ),
        PriorRow(
            "Exposés initiaux",
            "Log-normale",
            f"μ = {E0_MU:.2f}, σ = {E0_SIGMA}",
            "hypothèse",
            "Quelques dizaines de personnes au jour 0, réparties comme la population.",
        ),
        PriorRow(
            "Part initiale déjà infectieuse",
            "Beta",
            "α = 2, β = 5",
            "hypothèse",
            "Moins d'infectieux que d'exposés au départ.",
        ),
        PriorRow(
            "Fond de syndromes grippaux",
            "Log-normale",
            f"μ = {FLU_MU:.2f}, σ = {FLU_SIGMA}",
            "hypothèse",
            "Tests de personnes non infectées. Les consignes ne donnent pas le niveau.",
        ),
        PriorRow(
            "Surdispersion des observations",
            "Log-normale",
            f"μ = {CONCENTRATION_MU:.2f}, σ = {CONCENTRATION_SIGMA}",
            "hypothèse",
            "Variante B seulement. Variante A : concentration fixée dans le YAML.",
        ),
        PriorRow(
            "Bruit de la dynamique latente",
            "Demi-normale",
            f"échelle {PROCESS_SIGMA_SCALE}",
            "hypothèse",
            "Variante C : bruit normal, en écarts-types binomiaux.",
        ),
    )
