"""Relie les infections latentes aux séries publiées.

Trois quantités distinctes : les nouvelles infections, les tests, et les
positifs. Un faux négatif n'est pas un cas confirmé. Une admission encore
non confirmée est comptée positive le jour de l'entrée.
"""

import jax
import jax.numpy as jnp


def expected_observations(
    new_symptomatic: jax.Array,
    new_hospital: jax.Array,
    p_test: jax.Array,
    sensitivity: jax.Array,
    specificity: jax.Array,
    flu_mean: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Moyennes de positifs, de tests, et nombre de faux négatifs.

    ``new_symptomatic`` et ``new_hospital`` peuvent être (jours, groupes)
    ou déjà sommés. ``p_test`` est un scalaire ou un vecteur de jours.
    """
    symptomatic = _total(new_symptomatic)
    admitted = _total(new_hospital)
    infected_tests = p_test * symptomatic
    confirmed_on_test = sensitivity * infected_tests
    false_negatives = infected_tests * (1.0 - sensitivity)
    # Déjà confirmés au moment des symptômes : on ne les recompte pas.
    admission_tests = admitted * (1.0 - p_test * sensitivity)
    flu_positives = (1.0 - specificity) * flu_mean
    expected_positive = confirmed_on_test + flu_positives + admission_tests
    expected_tested = infected_tests + flu_mean + admission_tests
    return expected_positive, expected_tested, false_negatives


def _total(values: jax.Array) -> jax.Array:
    if values.ndim == 2:
        return jnp.sum(values, axis=1)
    return values
