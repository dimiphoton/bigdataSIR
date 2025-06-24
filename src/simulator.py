"""
Epidemiological Model Simulator

This module provides simulation capabilities for a 7-compartment SEIHRD model:
S (Susceptible), E (Exposed), I (Infected), H (Hospitalized), 
ICU (Intensive Care), R (Recovered), D (Deceased).

Functions:
    seir_step: Advance one time step using deterministic transitions
    simulate: Simulate the model for multiple days
"""

import numpy as np
from typing import Dict, Tuple
import warnings


def seir_step(state: np.ndarray, params: Dict[str, float], N: int) -> Tuple[np.ndarray, int]:
    """
    Advance one time step using deterministic transitions between compartments.
    
    Args:
        state: Current state vector [S, E, I, H, ICU, R, D]
        params: Dictionary containing transition rates:
            - 'beta': Transmission rate (S -> E)
            - 'sigma': Incubation rate (E -> I)
            - 'gamma': Recovery rate from I (I -> R)
            - 'h': Hospitalization rate (I -> H)
            - 'icu': ICU admission rate (H -> ICU)
            - 'gamma_h': Recovery rate from H (H -> R)
            - 'gamma_icu': Recovery rate from ICU (ICU -> R)
            - 'mu_h': Death rate from H (H -> D)
            - 'mu_icu': Death rate from ICU (ICU -> D)
        N: Total population size
    
    Returns:
        new_state: Updated state vector [S, E, I, H, ICU, R, D]
        new_infections: Number of new infections entering I compartment
    """
    # Unpack current state
    S, E, I, H, ICU, R, D = state
    
    # Validate inputs
    if len(state) != 7:
        raise ValueError("State must have exactly 7 components: [S, E, I, H, ICU, R, D]")
    
    if abs(sum(state) - N) > 1e-10:
        warnings.warn(f"State sum ({sum(state):.2f}) differs from population N ({N})")
    
    # Calculate transition rates
    infection_rate = params['beta'] * S * I / N
    incubation_rate = params['sigma'] * E
    recovery_rate_I = params['gamma'] * I
    hospitalization_rate = params['h'] * I
    icu_rate = params['icu'] * H
    recovery_rate_H = params['gamma_h'] * H
    recovery_rate_ICU = params['gamma_icu'] * ICU
    death_rate_H = params['mu_h'] * H
    death_rate_ICU = params['mu_icu'] * ICU
    
    # Ensure non-negative transitions
    infection_rate = max(0, infection_rate)
    incubation_rate = max(0, incubation_rate)
    recovery_rate_I = max(0, recovery_rate_I)
    hospitalization_rate = max(0, hospitalization_rate)
    icu_rate = max(0, icu_rate)
    recovery_rate_H = max(0, recovery_rate_H)
    recovery_rate_ICU = max(0, recovery_rate_ICU)
    death_rate_H = max(0, death_rate_H)
    death_rate_ICU = max(0, death_rate_ICU)
    
    # Calculate new state
    new_S = S - infection_rate
    new_E = E + infection_rate - incubation_rate
    new_I = I + incubation_rate - recovery_rate_I - hospitalization_rate
    new_H = H + hospitalization_rate - icu_rate - recovery_rate_H - death_rate_H
    new_ICU = ICU + icu_rate - recovery_rate_ICU - death_rate_ICU
    new_R = R + recovery_rate_I + recovery_rate_H + recovery_rate_ICU
    new_D = D + death_rate_H + death_rate_ICU
    
    # Ensure non-negative states
    new_state = np.array([
        max(0, new_S), max(0, new_E), max(0, new_I), 
        max(0, new_H), max(0, new_ICU), max(0, new_R), max(0, new_D)
    ])
    
    # Normalize to maintain population size
    total = np.sum(new_state)
    if total > 0:
        new_state = new_state * N / total
    
    return new_state, int(incubation_rate)


def simulate(initial_state: np.ndarray, 
            params: Dict[str, float], 
            N: int, 
            days: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the model for a specified number of days.
    
    Args:
        initial_state: Initial state vector [S, E, I, H, ICU, R, D]
        params: Dictionary containing transition rates (see seir_step)
        N: Total population size
        days: Number of days to simulate
    
    Returns:
        trajectory: Array of shape (days+1, 7) containing state vectors for each day
        new_infections: Array of shape (days,) containing daily new infections
    """
    # Validate inputs
    if len(initial_state) != 7:
        raise ValueError("Initial state must have exactly 7 components: [S, E, I, H, ICU, R, D]")
    
    if days < 0:
        raise ValueError("Days must be non-negative")
    
    # Initialize arrays
    trajectory = np.zeros((days + 1, 7))
    new_infections = np.zeros(days)
    
    # Set initial state
    trajectory[0] = initial_state.copy()
    
    # Simulate day by day
    for day in range(days):
        current_state = trajectory[day]
        new_state, daily_infections = seir_step(current_state, params, N)
        
        trajectory[day + 1] = new_state
        new_infections[day] = daily_infections
    
    return trajectory, new_infections


def get_default_params() -> Dict[str, float]:
    """
    Get default epidemiological parameters.
    
    Returns:
        Dictionary with default parameter values
    """
    return {
        'beta': 0.3,        # Transmission rate
        'sigma': 0.2,       # Incubation rate (1/5 days)
        'gamma': 0.1,       # Recovery rate from I (1/10 days)
        'h': 0.05,          # Hospitalization rate (5% of I)
        'icu': 0.1,         # ICU admission rate (10% of H)
        'gamma_h': 0.15,    # Recovery rate from H (1/6.7 days)
        'gamma_icu': 0.1,   # Recovery rate from ICU (1/10 days)
        'mu_h': 0.02,       # Death rate from H (2% of H)
        'mu_icu': 0.15      # Death rate from ICU (15% of ICU)
    }


def get_default_initial_state(N: int) -> np.ndarray:
    """
    Get default initial state for a population of size N.
    
    Args:
        N: Total population size
    
    Returns:
        Initial state vector [S, E, I, H, ICU, R, D]
    """
    return np.array([N - 10, 5, 5, 0, 0, 0, 0])


def validate_params(params: Dict[str, float]) -> None:
    """
    Validate that all required parameters are present and non-negative.
    
    Args:
        params: Dictionary of parameters to validate
    
    Raises:
        ValueError: If required parameters are missing or negative
    """
    required_params = [
        'beta', 'sigma', 'gamma', 'h', 'icu', 
        'gamma_h', 'gamma_icu', 'mu_h', 'mu_icu'
    ]
    
    for param in required_params:
        if param not in params:
            raise ValueError(f"Missing required parameter: {param}")
        if params[param] < 0:
            raise ValueError(f"Parameter {param} must be non-negative")


def get_simulation_summary(trajectory: np.ndarray, 
                          new_infections: np.ndarray) -> Dict:
    """
    Get summary statistics from simulation results.
    
    Args:
        trajectory: State trajectory array (days+1, 7)
        new_infections: Daily new infections array (days,)
    
    Returns:
        Dictionary with summary statistics
    """
    return {
        'peak_infected': np.max(trajectory[:, 2]),  # Peak I
        'peak_hospitalized': np.max(trajectory[:, 3]),  # Peak H
        'peak_icu': np.max(trajectory[:, 4]),  # Peak ICU
        'total_deaths': trajectory[-1, 6],  # Final D
        'total_recovered': trajectory[-1, 5],  # Final R
        'total_infections': np.sum(new_infections),
        'peak_daily_infections': np.max(new_infections)
    }
