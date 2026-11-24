"""Dynamique SEIRH étendue, en temps discret, avec jax.lax.scan.

Compartiments, par groupe d'âge : S, E, I_a, I_s, H, ICU, R, D.
Les flux sont des espérances. La variante C ajoute un bruit normal sur
les nouvelles infections, calé sur la variance binomiale.
"""

from typing import TypedDict

import jax
import jax.numpy as jnp

S = 0
E = 1
IA = 2
IS = 3
H = 4
ICU = 5
R = 6
D = 7
N_COMPARTMENTS = 8


class Trajectory(TypedDict):
    """Séries quotidiennes produites par le scan."""

    state: jax.Array
    new_infections: jax.Array
    new_symptomatic: jax.Array
    new_hospital: jax.Array
    new_deaths: jax.Array


def leave_probability(rate: jax.Array) -> jax.Array:
    """Probabilité de quitter un stade dans la journée. rate = 1 / durée."""
    return 1.0 - jnp.exp(-rate)


def initial_state(
    population: jax.Array,
    exposed: jax.Array,
    infectious: jax.Array,
    symptomatic_fraction: jax.Array,
) -> jax.Array:
    """Répartit le germe au prorata de la population. Le reste est susceptible."""
    shares = population / jnp.sum(population)
    state = jnp.zeros((population.shape[0], N_COMPARTMENTS))
    state = state.at[:, E].set(exposed * shares)
    state = state.at[:, IA].set(infectious * (1.0 - symptomatic_fraction) * shares)
    state = state.at[:, IS].set(infectious * symptomatic_fraction * shares)
    state = state.at[:, S].set(population - jnp.sum(state, axis=1))
    return state


def run_dynamics(
    state0: jax.Array,
    beta_t: jax.Array,
    shocks: jax.Array,
    *,
    population: jax.Array,
    contacts: jax.Array,
    sigma: jax.Array,
    gamma: jax.Array,
    gamma_h: jax.Array,
    gamma_icu: jax.Array,
    symptomatic_fraction: jax.Array,
    p_hosp: jax.Array,
    p_icu: jax.Array,
    p_death_ward: jax.Array,
    p_death_icu: jax.Array,
    asymptomatic_infectiousness: jax.Array,
    process_sigma: jax.Array,
) -> Trajectory:
    """Avance un jour à la fois. ``shocks`` a la forme (jours, groupes)."""

    def step(
        state: jax.Array, inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, Trajectory]:
        beta, shock = inputs
        beta = jnp.clip(beta, 0.0, 5.0)
        infectious = asymptomatic_infectiousness * state[:, IA] + state[:, IS]
        # contacts[g, j] : contacts moyens d'une personne du groupe g avec le groupe j.
        pressure = contacts @ (infectious / population)
        force = jnp.clip(beta * pressure, 0.0, 30.0)
        infection_probability = 1.0 - jnp.exp(-force)
        mean_new = state[:, S] * infection_probability
        binomial_variance = jnp.clip(
            state[:, S] * infection_probability * (1.0 - infection_probability),
            0.0,
            None,
        )
        noisy_new = mean_new + process_sigma * jnp.sqrt(binomial_variance) * shock
        new_infections = jnp.clip(noisy_new, 0.0, state[:, S])

        leave_e = state[:, E] * leave_probability(sigma)
        new_symptomatic = leave_e * symptomatic_fraction
        new_asymptomatic = leave_e * (1.0 - symptomatic_fraction)

        leave_ia = state[:, IA] * leave_probability(gamma)
        leave_is = state[:, IS] * leave_probability(gamma)
        new_hospital = leave_is * p_hosp
        recover_is = leave_is * (1.0 - p_hosp)

        leave_h = state[:, H] * leave_probability(gamma_h)
        new_icu = leave_h * p_icu
        leave_ward = leave_h * (1.0 - p_icu)
        death_ward = leave_ward * p_death_ward
        recover_ward = leave_ward * (1.0 - p_death_ward)

        leave_icu = state[:, ICU] * leave_probability(gamma_icu)
        death_icu = leave_icu * p_death_icu
        recover_icu = leave_icu * (1.0 - p_death_icu)
        new_deaths = death_ward + death_icu

        updated = jnp.stack(
            [
                state[:, S] - new_infections,
                state[:, E] + new_infections - leave_e,
                state[:, IA] + new_asymptomatic - leave_ia,
                state[:, IS] + new_symptomatic - leave_is,
                state[:, H] + new_hospital - leave_h,
                state[:, ICU] + new_icu - leave_icu,
                state[:, R] + leave_ia + recover_is + recover_ward + recover_icu,
                state[:, D] + new_deaths,
            ],
            axis=1,
        )
        aux: Trajectory = {
            "state": updated,
            "new_infections": new_infections,
            "new_symptomatic": new_symptomatic,
            "new_hospital": new_hospital,
            "new_deaths": new_deaths,
        }
        return updated, aux

    _final, trajectory = jax.lax.scan(step, state0, (beta_t, shocks))
    return trajectory
