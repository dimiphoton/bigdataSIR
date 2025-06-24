# src/build_model.py

"""
PyMC Model Builder for SEIR-HICURD Bayesian Inference

This module constructs PyMC models for Bayesian inference using the SEIR-HICURD
simulator with testing-based observability. It defines priors for 
epidemiological parameters and creates observation models that account for 
testing sensitivity and coverage.

Functions:
    build_model: Construct PyMC model for SEIR-HICURD inference with testing
"""

import pymc as pm
import numpy as np
from simulator import simulate


def build_model(observed_cases: np.ndarray, 
                initial_state: np.ndarray, 
                N: int, 
                days: int) -> pm.Model:
    """
    Build a PyMC probabilistic model for a deterministic SEIR-HICURD 
    compartmental epidemic model with testing-based observability.
    
    This function creates a PyMC model that:
    1. Defines priors for all epidemiological parameters
    2. Runs the deterministic SEIR-HICURD simulator
    3. Models observed cases as a function of new infections with testing parameters
    4. Returns the built model for sampling
    
    Args:
        observed_cases: Array of observed daily positive tests (length = days)
        initial_state: Initial state vector [S, E, I, H, ICU, R, D]
        N: Total population size
        days: Number of days to simulate
    
    Returns:
        PyMC model object ready for sampling
        
    Raises:
        ValueError: If inputs are invalid or inconsistent
    """
    # Validate inputs
    if len(observed_cases) != days:
        raise ValueError(f"observed_cases length ({len(observed_cases)}) "
                        f"must equal days ({days})")
    
    if len(initial_state) != 7:
        raise ValueError("initial_state must have exactly 7 components")
    
    if N <= 0:
        raise ValueError("Population size N must be positive")
    
    if days <= 0:
        raise ValueError("Days must be positive")
    
    with pm.Model() as model:
        # --- Epidemiological Parameters (Priors) ---
        
        # Transmission rate
        beta = pm.LogNormal("beta", mu=0, sigma=1)
        
        # Mean days from E → I (incubation period)
        incubation = pm.Uniform("incubation", lower=1, upper=5)
        
        # Days in I before recovery/hospitalization (symptomatic period)
        symptomatic = pm.Uniform("symptomatic", lower=4, upper=10)
        
        # Fraction of I hospitalized
        h_rate = pm.Beta("h_rate", alpha=2, beta=5)
        
        # Days in H (hospital stay duration)
        h_duration = pm.Uniform("h_duration", lower=4, upper=14)
        
        # Fraction of H going to ICU
        icu_rate = pm.Beta("icu_rate", alpha=2, beta=5)
        
        # Days in ICU (ICU stay duration)
        icu_duration = pm.Uniform("icu_duration", lower=3, upper=14)
        
        # ICU fatality rate
        mu = pm.Beta("mu", alpha=2, beta=5)
        
        # --- Convert durations to rates for simulator ---
        sigma = 1.0 / incubation  # Incubation rate (E -> I)
        gamma = 1.0 / symptomatic  # Recovery rate from I (I -> R)
        gamma_h = 1.0 / h_duration  # Recovery rate from H (H -> R)
        gamma_icu = 1.0 / icu_duration  # Recovery rate from ICU (ICU -> R)
        
        # --- Parameter Dictionary for Simulator ---
        params = {
            'beta': beta,
            'sigma': sigma,
            'gamma': gamma,
            'h': h_rate,
            'icu': icu_rate,
            'gamma_h': gamma_h,
            'gamma_icu': gamma_icu,
            'mu_h': 0.0,  # No deaths from regular hospitalization
            'mu_icu': mu
        }
        
        # --- Deterministic Simulation ---
        # Run the SEIR-HICURD simulator with current parameter values
        states, new_I = simulate(initial_state, params, N, days)
        
        # --- Observation Model (Testing) ---
        # Fixed testing parameters
        test_sensitivity = 0.8  # 80% test sensitivity (fixed)
        test_coverage = 0.5     # 50% coverage of symptomatic (fixed)
        
        # Expected positive tests = new_I × test_coverage × test_sensitivity
        expected_positives = new_I * test_coverage * test_sensitivity
        
        # Model daily observed cases with Poisson likelihood
        pm.Poisson("obs", mu=expected_positives, observed=observed_cases)
        
        # --- Derived Quantities for Analysis ---
        pm.Deterministic("peak_infections", pm.math.max(new_I))
        pm.Deterministic("total_infections", pm.math.sum(new_I))
        pm.Deterministic("peak_hospitalized", pm.math.max(states[:, 3]))
        pm.Deterministic("peak_icu", pm.math.max(states[:, 4]))
        pm.Deterministic("final_deaths", states[-1, 6])
        pm.Deterministic("final_susceptible", states[-1, 0])
        
        # Store full trajectory and new infections for analysis
        pm.Deterministic("trajectory", states)
        pm.Deterministic("new_infections", new_I)
    
    return model


def build_model_with_multiple_observations(
    observed_data: Dict[str, np.ndarray],
    initial_state: np.ndarray,
    N: int,
    days: int
) -> pm.Model:
    """
    Construct a PyMC model with multiple observation types and testing parameters.
    
    This extended version can handle multiple types of observed data:
    - Daily new cases (with testing model)
    - Current hospitalized
    - Current ICU patients
    - Cumulative deaths
    
    Args:
        observed_data: Dictionary with observation arrays
        initial_state: Initial state vector [S, E, I, H, ICU, R, D]
        N: Total population size
        days: Number of days to simulate
    
    Returns:
        PyMC model object ready for sampling
    """
    # Validate inputs
    if len(initial_state) != 7:
        raise ValueError("initial_state must have exactly 7 components")
    
    if N <= 0:
        raise ValueError("Population size N must be positive")
    
    if days <= 0:
        raise ValueError("Days must be positive")
    
    with pm.Model() as model:
        # --- Prior Distributions for Epidemiological Parameters ---
        
        # Transmission rate (S -> E)
        beta = pm.LogNormal("beta", mu=np.log(0.3), sigma=0.5)
        
        # Incubation period
        incubation = pm.LogNormal("incubation", mu=np.log(5.0), sigma=0.3)
        
        # Symptomatic period
        symptomatic = pm.LogNormal("symptomatic", mu=np.log(10.0), sigma=0.3)
        
        # Hospitalization rate
        h_rate = pm.Beta("h_rate", alpha=2, beta=20)
        
        # Hospital stay duration
        h_duration = pm.LogNormal("h_duration", mu=np.log(7.0), sigma=0.3)
        
        # ICU rate
        icu_rate = pm.Beta("icu_rate", alpha=2, beta=10)
        
        # ICU stay duration
        icu_duration = pm.LogNormal("icu_duration", mu=np.log(7.0), sigma=0.3)
        
        # Death rate for ICU patients
        mu = pm.Beta("mu", alpha=3, beta=20)
        
        # --- Convert durations to rates ---
        sigma = 1.0 / incubation
        gamma = 1.0 / symptomatic
        gamma_h = 1.0 / h_duration
        gamma_icu = 1.0 / icu_duration
        
        params = {
            'beta': beta,
            'sigma': sigma,
            'gamma': gamma,
            'h': h_rate,
            'icu': icu_rate,
            'gamma_h': gamma_h,
            'gamma_icu': gamma_icu,
            'mu_h': 0.0,
            'mu_icu': mu
        }
        
        # --- Deterministic Simulation ---
        states, new_I = simulate(initial_state, params, N, days)
        
        # --- Testing Parameters ---
        testing_rate = pm.Beta("testing_rate", alpha=5, beta=5)
        
        # --- Multiple Observation Models ---
        
        # 1. Daily new cases (with testing model)
        if "new_cases" in observed_data:
            expected_cases = new_I * testing_rate
            expected_cases = pm.math.clip(expected_cases, 0, N)
            pm.Poisson("observed_new_cases", 
                      mu=expected_cases, 
                      observed=observed_data["new_cases"])
        
        # 2. Current hospitalized (H compartment)
        if "hospitalized" in observed_data:
            obs_sigma_hosp = pm.HalfNormal("obs_sigma_hosp", sigma=5.0)
            pm.Normal("observed_hospitalized", 
                     mu=states[1:, 3],  # H compartment, skip day 0
                     sigma=obs_sigma_hosp, 
                     observed=observed_data["hospitalized"])
        
        # 3. Current ICU patients (ICU compartment)
        if "icu" in observed_data:
            obs_sigma_icu = pm.HalfNormal("obs_sigma_icu", sigma=2.0)
            pm.Normal("observed_icu", 
                     mu=states[1:, 4],  # ICU compartment, skip day 0
                     sigma=obs_sigma_icu, 
                     observed=observed_data["icu"])
        
        # 4. Cumulative deaths (D compartment)
        if "deaths" in observed_data:
            obs_sigma_deaths = pm.HalfNormal("obs_sigma_deaths", sigma=1.0)
            pm.Normal("observed_deaths", 
                     mu=states[1:, 6],  # D compartment, skip day 0
                     sigma=obs_sigma_deaths, 
                     observed=observed_data["deaths"])
        
        # --- Derived Quantities ---
        pm.Deterministic("peak_infections", pm.math.max(new_I))
        pm.Deterministic("total_infections", pm.math.sum(new_I))
        pm.Deterministic("peak_hospitalized", pm.math.max(states[:, 3]))
        pm.Deterministic("peak_icu", pm.math.max(states[:, 4]))
        pm.Deterministic("final_deaths", states[-1, 6])
    
    return model
