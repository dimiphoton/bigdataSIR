import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple
from simulate import seir_step  # Your JAX symbolic step function


def forecast_posterior_trajectories(
    posterior_samples: Dict[str, jnp.ndarray],
    initial_state: jnp.ndarray,
    population: float,
    forecast_days: int,
) -> np.ndarray:
    """
    Propagate posterior samples forward to forecast future trajectories.

    Args:
        posterior_samples: Dict of parameter arrays from inference,
                           each shape = (num_samples, ...)
        initial_state: JAX array of initial compartment values (7,).
        population: Total population size.
        forecast_days: Number of days to simulate forward.

    Returns:
        trajectories: np.ndarray of shape (num_samples, forecast_days + 1, 7)
                      Each trajectory corresponds to one posterior sample.
    """

    num_samples = next(iter(posterior_samples.values())).shape[0]
    trajectories = np.zeros((num_samples, forecast_days + 1, initial_state.shape[0]))

    for i in range(num_samples):
        # Extract parameter values for this sample (convert to float scalar)
        params_i = {k: float(posterior_samples[k][i]) for k in posterior_samples}

        # Initialize trajectory and state
        traj = [initial_state]
        state = initial_state

        for day in range(forecast_days):
            state, _ = seir_step(state, params_i, population)
            traj.append(state)

        trajectories[i] = np.stack(traj)

    return trajectories


def summarize_forecast(trajectories: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Summarize forecast trajectories into median and credible intervals.

    Args:
        trajectories: shape (num_samples, days+1, 7)

    Returns:
        summary: Dict with keys 'median', 'lower_95', 'upper_95', each
                 an array of shape (days+1, 7)
    """
    median = np.median(trajectories, axis=0)
    lower_95 = np.percentile(trajectories, 2.5, axis=0)
    upper_95 = np.percentile(trajectories, 97.5, axis=0)

    return {
        "median": median,
        "lower_95": lower_95,
        "upper_95": upper_95,
    }


# Example usage:
if __name__ == "__main__":
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    # Dummy initial state (replace with your actual)
    initial_state = jnp.array([999_000., 1_000., 0., 0., 0., 0., 0.])
    population = 1_000_000
    forecast_days = 30

    # Dummy posterior samples dict (replace with your actual MCMC samples)
    # Example: posterior_samples = mcmc.get_samples()
    posterior_samples = {
        "beta": jnp.array([0.25, 0.27, 0.3]),
        "sigma": jnp.array([1/5.1, 1/5.2, 1/5.3]),
        "gamma": jnp.array([1/2.2, 1/2.3, 1/2.4]),
        "h": jnp.array([0.15, 0.14, 0.16]),
        "icu": jnp.array([0.3, 0.31, 0.29]),
        "gamma_h": jnp.array([1/7, 1/6.9, 1/7.1]),
        "gamma_icu": jnp.array([1/10, 1/9.8, 1/10.2]),
        "mu_h": jnp.array([0.01, 0.011, 0.009]),
        "mu_icu": jnp.array([0.05, 0.051, 0.049]),
    }

    trajectories = forecast_posterior_trajectories(posterior_samples, initial_state, population, forecast_days)
    summary = summarize_forecast(trajectories)

    print("Median forecast shape:", summary["median"].shape)
    print("Median infections at final day:", summary["median"][-1, 2])  # I compartment
