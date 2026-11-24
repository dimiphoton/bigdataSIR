"""Bruit d'observation des projections. Même histoire que le modèle d'inférence."""

import numpy as np


def expected_counts(
    new_symptomatic: np.ndarray,
    new_hospital: np.ndarray,
    p_test: np.ndarray,
    sensitivity: np.ndarray,
    specificity: np.ndarray,
    flu_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moyennes des positifs, des tests et des faux négatifs. Forme (tirages, jours)."""
    rate = p_test[:, None] if p_test.ndim == 1 else p_test
    sens = sensitivity[:, None]
    spec = specificity[:, None]
    flu = flu_mean[:, None]
    infected_tests = rate * new_symptomatic
    confirmed = sens * infected_tests
    false_negatives = infected_tests * (1.0 - sens)
    # Admissions pas encore confirmées au moment des symptômes.
    admissions = new_hospital * (1.0 - rate * sens)
    flu_positives = (1.0 - spec) * flu
    positive = confirmed + flu_positives + admissions
    tested = infected_tests + flu + admissions
    return positive, tested, false_negatives


def sample_negative_binomial(
    rng: np.random.Generator,
    mean: np.ndarray,
    concentration: np.ndarray,
) -> np.ndarray:
    """Mélange gamma-Poisson. concentration a une valeur par tirage."""
    safe_mean = np.clip(mean, 1e-6, None)
    conc = np.clip(concentration, 1e-6, None)[:, None]
    lam = rng.gamma(np.broadcast_to(conc, safe_mean.shape), safe_mean / conc)
    return rng.poisson(lam).astype(float)


def sample_beta_binomial(
    rng: np.random.Generator,
    positive_mean: np.ndarray,
    tested_mean: np.ndarray,
    concentration: np.ndarray,
    tested: np.ndarray,
) -> np.ndarray:
    """Positifs sachant le nombre de tests tiré."""
    ratio = np.clip(
        positive_mean / np.clip(tested_mean, 1e-6, None),
        1e-4,
        1.0 - 1e-4,
    )
    conc = np.clip(concentration, 1e-6, None)[:, None]
    alpha = ratio * conc
    beta = (1.0 - ratio) * conc
    probability = rng.beta(alpha, beta)
    return rng.binomial(np.asarray(tested, dtype=int), probability).astype(float)


def sample_log1p(
    rng: np.random.Generator,
    mean: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Bruit normal sur log(1 + stock), puis retour à l'échelle."""
    noisy = rng.normal(size=mean.shape) * scale
    return np.expm1(np.log1p(np.clip(mean, 0.0, None)) + noisy).clip(0.0)
