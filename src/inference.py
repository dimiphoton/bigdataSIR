import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, AutoNormal
from numpyro.optim import Adam
from typing import Dict, Any, Optional, Union, Tuple
from build_model_numpyro import seir_model


def run_mcmc(
    initial_state: jnp.ndarray,
    params_init: Dict[str, Any],
    observed_data: jnp.ndarray,
    population: float,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
    rng_key: Optional[jax.random.PRNGKey] = None,
) -> MCMC:
    """
    Run MCMC inference on the SEIR model using NUTS sampler.

    Args:
        initial_state: Initial compartment state vector (7,).
        params_init: Initial guesses or fixed parameters for the model.
        observed_data: Observed infection data (or any observed trajectory).
        population: Total population size.
        num_warmup: Number of warmup steps.
        num_samples: Number of samples to draw.
        num_chains: Number of parallel chains.
        rng_key: JAX random key (optional).

    Returns:
        numpyro.infer.MCMC object with posterior samples.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Define a closure to wrap seir_model with fixed inputs except parameters
    def model_wrapper():
        return seir_model(initial_state, params_init, population, observed_data)

    kernel = NUTS(model_wrapper)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key)
    mcmc.print_summary()
    return mcmc


def run_svi(
    initial_state: jnp.ndarray,
    params_init: Dict[str, Any],
    observed_data: jnp.ndarray,
    population: float,
    num_steps: int = 5000,
    rng_key: Optional[jax.random.PRNGKey] = None,
) -> Tuple[SVI, Dict[str, Any]]:
    """
    Run Variational Inference (SVI) on the SEIR model.

    Args:
        initial_state: Initial compartment state vector (7,).
        params_init: Initial guesses or fixed parameters for the model.
        observed_data: Observed infection data (or any observed trajectory).
        population: Total population size.
        num_steps: Number of optimization steps.
        rng_key: JAX random key (optional).

    Returns:
        Tuple of:
            - trained SVI object
            - inferred parameters dict (guide params)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    guide = AutoNormal(seir_model)
    optimizer = Adam(learning_rate=1e-3)
    svi = SVI(seir_model, guide, optimizer, loss=Trace_ELBO())

    svi_state = svi.init(rng_key, initial_state, params_init, population, observed_data)
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, initial_state, params_init, population, observed_data)
        if step % 500 == 0:
            print(f"Step {step} - ELBO loss: {loss:.4f}")

    params = svi.get_params(svi_state)
    return svi, params


def get_posterior_samples(mcmc: MCMC) -> Dict[str, Any]:
    """
    Extract posterior samples from MCMC object.

    Args:
        mcmc: MCMC inference object.

    Returns:
        Dictionary of posterior samples.
    """
    return mcmc.get_samples()


# Example usage:
if __name__ == "__main__":
    import jax.random as random

    key = random.PRNGKey(42)

    # Dummy placeholders (replace with real data)
    initial_state = jnp.array([999_000, 1_000, 0, 0, 0, 0, 0])  # example SEIRHICURD init
    params_init = {
        "beta": 0.3,
        "sigma": 1/5.2,
        "gamma": 1/2.3,
        "h": 0.15,
        "icu": 0.3,
        "gamma_h": 1/7,
        "gamma_icu": 1/10,
        "mu_h": 0.01,
        "mu_icu": 0.05,
    }
    population = 1_000_000
    observed_data = jnp.zeros((100,))  # Replace with actual observed infection counts or data

    # MCMC example
    mcmc_result = run_mcmc(initial_state, params_init, observed_data, population, rng_key=key)
    posterior_samples = get_posterior_samples(mcmc_result)

    # Uncomment below for SVI example
    # svi_obj, svi_params = run_svi(initial_state, params_init, observed_data, population, rng_key=key)
