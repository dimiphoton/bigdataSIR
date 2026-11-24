import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from simulate import simulate_symbolic  # Must be JAX-compatible

def seir_model(observed_cases, initial_state, N, days):
    """
    SEIR-HICURD model in NumPyro.

    Args:
        observed_cases (array): shape (days,), daily observed positive cases.
        initial_state (array): shape (7,), initial [S, E, I, H, ICU, R, D].
        N (int): Total population.
        days (int): Number of simulation days.

    Returns:
        Sample site 'obs' for observed cases.
        Deterministics:
            - peak_I: Peak number of infected individuals
            - peak_H: Peak hospitalized
            - peak_ICU: Peak ICU
            - total_deaths: Final number of deaths
    """

    # --- Priors ---
    beta         = numpyro.sample("beta",         dist.LogNormal(0.0, 1.0))
    incubation   = numpyro.sample("incubation",   dist.Uniform(1.0, 5.0))
    symptomatic  = numpyro.sample("symptomatic",  dist.Uniform(4.0, 10.0))
    h_rate       = numpyro.sample("h_rate",       dist.Beta(2.0, 5.0))
    h_duration   = numpyro.sample("h_duration",   dist.Uniform(4.0, 14.0))
    icu_rate     = numpyro.sample("icu_rate",     dist.Beta(2.0, 5.0))
    icu_duration = numpyro.sample("icu_duration", dist.Uniform(3.0, 14.0))
    mu           = numpyro.sample("mu",           dist.Beta(2.0, 5.0))  # ICU mortality

    # --- Derived parameters ---
    sigma      = 1.0 / incubation
    gamma      = 1.0 / symptomatic
    gamma_h    = 1.0 / h_duration
    gamma_icu  = 1.0 / icu_duration

    params = {
        "beta": beta,
        "sigma": sigma,
        "gamma": gamma,
        "h": h_rate,
        "icu": icu_rate,
        "gamma_h": gamma_h,
        "gamma_icu": gamma_icu,
        "mu_h": 0.0,  # All deaths in ICU
        "mu_icu": mu
    }

    # --- Simulate compartment transitions ---
    states, new_infections = simulate_symbolic(initial_state, params, N, days)

    # --- Observation model ---
    test_sens = 0.8      # Test sensitivity
    test_cov  = 0.5      # Fraction of symptomatic tested
    expected_pos = new_infections * test_sens * test_cov

    numpyro.sample("obs", dist.Poisson(expected_pos), obs=observed_cases)

    # --- Deterministic metrics ---
    numpyro.deterministic("peak_I", jnp.max(states[:, 2]))
    numpyro.deterministic("peak_H", jnp.max(states[:, 3]))
    numpyro.deterministic("peak_ICU", jnp.max(states[:, 4]))
    numpyro.deterministic("total_deaths", states[-1, 6])
