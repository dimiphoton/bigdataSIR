# src/simulate.py

import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import Tuple, Dict, Any


def _compute_transitions(S, E, I, H, ICU, R, D, params, N):
    """Compute state transitions for SEIR-HICURD model."""
    beta = params["beta"]
    sigma = params["sigma"]
    gamma = params["gamma"]
    h = params["h"]
    icu = params["icu"]
    gamma_h = params["gamma_h"]
    gamma_icu = params["gamma_icu"]
    mu_h = params["mu_h"]
    mu_icu = params["mu_icu"]

    new_E = beta * S * I / N
    new_I = sigma * E

    to_H = gamma * I * h
    to_R_from_I = gamma * I * (1 - h)

    to_ICU = gamma_h * H * icu
    to_R_from_H = gamma_h * H * (1 - icu) * (1 - mu_h)
    to_D_from_H = gamma_h * H * (1 - icu) * mu_h

    to_R_from_ICU = gamma_icu * ICU * (1 - mu_icu)
    to_D_from_ICU = gamma_icu * ICU * mu_icu

    S_new = S - new_E
    E_new = E + new_E - new_I
    I_new = I + new_I - to_H - to_R_from_I
    H_new = H + to_H - to_ICU - to_R_from_H - to_D_from_H
    ICU_new = ICU + to_ICU - to_R_from_ICU - to_D_from_ICU
    R_new = R + to_R_from_I + to_R_from_H + to_R_from_ICU
    D_new = D + to_D_from_H + to_D_from_ICU

    return (S_new, E_new, I_new, H_new, ICU_new, R_new, D_new), new_I


# ---------------------- Symbolic JAX version ----------------------

def seir_step(state: jnp.ndarray, params: Dict[str, Any], N: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    One symbolic step of the SEIR-HICURD model (JAX-based).
    """
    S, E, I, H, ICU, R, D = state
    new_state, new_I = _compute_transitions(S, E, I, H, ICU, R, D, params, N)
    return jnp.stack(new_state), new_I


def simulate_symbolic(
    initial_state: jnp.ndarray,
    params: Dict[str, Any],
    N: float,
    days: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX symbolic simulation (for inference with NumPyro).

    Returns:
        - trajectory: (days+1, 7)
        - new_infections: (days,)
    """
    def step_fn(state, _):
        # Calculate the next state and new infections
        new_state, new_infections = seir_step(state, params, N)
        # Return the new state for the next step's input (carry),
        # and also return it as an output to be collected.
        return new_state, (new_state, new_infections)

    # Run the scan. The second returned value will be a tuple
    # containing the stacked history of states and new infections.
    final_state, (trajectory, new_infections) = lax.scan(
        step_fn, initial_state, xs=None, length=days
    )

    # Prepend the initial state to the collected trajectory
    full_trajectory = jnp.concatenate([initial_state[None, :], trajectory], axis=0)
    
    return full_trajectory, new_infections


# ---------------------- NumPy (deterministic) version ----------------------

def seir_step_numpy(state: np.ndarray, params: Dict[str, Any], N: float) -> Tuple[np.ndarray, float]:
    """
    One deterministic SEIR-HICURD step using NumPy.
    """
    S, E, I, H, ICU, R, D = state
    new_state, new_I = _compute_transitions(S, E, I, H, ICU, R, D, params, N)
    return np.array(new_state), float(new_I)


def simulate_numpy(
    initial_state: np.ndarray, 
    params: Dict[str, Any], 
    N: float, 
    days: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic NumPy simulation for testing and plotting (non-JAX).
    
    Returns:
        - trajectory: (days+1, 7)
        - new_infections: (days,)
    """
    state = initial_state
    trajectory = [initial_state]
    new_infections = []

    for _ in range(days):
        state, new_I = seir_step_numpy(state, params, N)
        trajectory.append(state)
        new_infections.append(new_I)

    return np.stack(trajectory), np.array(new_infections)


# ---------------------- Summary ----------------------

def get_simulation_summary(
    trajectory: np.ndarray, 
    new_infections: np.ndarray
) -> Dict[str, float]:
    """
    Returns key indicators from a simulation run.
    """
    return {
        'peak_infected': float(np.max(trajectory[:, 2])),
        'peak_hospitalized': float(np.max(trajectory[:, 3])),
        'peak_icu': float(np.max(trajectory[:, 4])),
        'total_deaths': float(trajectory[-1, 6]),
        'total_recovered': float(trajectory[-1, 5]),
        'total_infections': float(np.sum(new_infections)),
        'peak_daily_infections': float(np.max(new_infections)),
    }
