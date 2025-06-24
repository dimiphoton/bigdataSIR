# src/forecast_with_uncertainty.py

"""
Forecast with Uncertainty Module

This module loads posterior samples from PyMC MCMC sampling and uses them to
generate probabilistic forecasts with uncertainty quantification.

Functions:
    load_posterior_samples: Load and sample from posterior trace
    forecast_from_trace: Generate ensemble forecasts from posterior samples
    summarize_forecast: Calculate summary statistics for forecasts
"""

import numpy as np
import arviz as az
from simulator import simulate
from config import N, initial_state
from typing import List, Dict, Tuple, Optional


def load_posterior_samples(trace_path: str, 
                          param_names: List[str], 
                          n_samples: int = 100) -> List[Dict[str, float]]:
    """
    Load posterior samples from trace file and randomly select a subset.
    
    Args:
        trace_path: Path to the trace.nc file
        param_names: List of parameter names to extract
        n_samples: Number of samples to randomly select
    
    Returns:
        List of dictionaries, each containing parameter values for one sample
    """
    # Load trace from NetCDF file
    trace = az.from_netcdf(trace_path).posterior
    
    # Get dimensions
    n_chains, n_draws = trace.dims["chain"], trace.dims["draw"]
    total_samples = n_chains * n_draws
    
    # Randomly select samples
    if n_samples > total_samples:
        n_samples = total_samples
        print(f"Warning: Requested {n_samples} samples but only {total_samples} available")
    
    sample_indices = np.random.choice(total_samples, n_samples, replace=False)
    
    # Extract parameter values for selected samples
    samples = []
    for idx in sample_indices:
        chain_idx, draw_idx = divmod(idx, n_draws)
        sample = {}
        for param_name in param_names:
            try:
                sample[param_name] = float(trace[param_name][chain_idx, draw_idx])
            except KeyError:
                print(f"Warning: Parameter '{param_name}' not found in trace")
                sample[param_name] = 0.0
        samples.append(sample)
    
    return samples


def convert_params_to_simulator_format(params: Dict[str, float]) -> Dict[str, float]:
    """
    Convert posterior parameters to simulator format.
    
    Args:
        params: Dictionary of posterior parameter samples
    
    Returns:
        Dictionary of parameters in simulator format
    """
    # Convert durations to rates for simulator
    sigma = 1.0 / params.get('incubation', 5.0)  # Incubation rate
    gamma = 1.0 / params.get('symptomatic', 10.0)  # Recovery rate from I
    gamma_h = 1.0 / params.get('h_duration', 7.0)  # Recovery rate from H
    gamma_icu = 1.0 / params.get('icu_duration', 7.0)  # Recovery rate from ICU
    
    # Return simulator-compatible parameter dictionary
    return {
        'beta': params.get('beta', 0.3),
        'sigma': sigma,
        'gamma': gamma,
        'h': params.get('h_rate', 0.05),
        'icu': params.get('icu_rate', 0.1),
        'gamma_h': gamma_h,
        'gamma_icu': gamma_icu,
        'mu_h': 0.0,  # No deaths from regular hospitalization
        'mu_icu': params.get('mu', 0.15)
    }


def forecast_from_trace(trace_path: str = "results/trace.nc", 
                       days: int = 60, 
                       n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ensemble forecasts from posterior samples.
    
    Args:
        trace_path: Path to the trace.nc file
        days: Number of days to forecast
        n_samples: Number of posterior samples to use
    
    Returns:
        all_states: Array of shape (n_samples, days+1, 7) containing state trajectories
        all_new_I: Array of shape (n_samples, days) containing daily new infections
    """
    # Parameter names in the posterior trace
    param_names = ["beta", "incubation", "symptomatic", "h_rate",
                   "h_duration", "icu_rate", "icu_duration", "mu"]
    
    # Load posterior samples
    print(f"Loading {n_samples} posterior samples from {trace_path}...")
    samples = load_posterior_samples(trace_path, param_names, n_samples)
    
    # Generate forecasts for each sample
    all_states = []
    all_new_I = []
    
    print(f"Generating forecasts for {len(samples)} samples...")
    for i, params in enumerate(samples):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(samples)} samples")
        
        # Convert parameters to simulator format
        simulator_params = convert_params_to_simulator_format(params)
        
        # Run simulation
        states, new_I = simulate(initial_state, simulator_params, N, days)
        
        # Store results
        all_states.append(states)
        all_new_I.append(new_I)
    
    # Stack into arrays
    all_states = np.stack(all_states)  # shape: (n_samples, days+1, 7)
    all_new_I = np.stack(all_new_I)    # shape: (n_samples, days)
    
    print(f"Forecast complete: {all_states.shape[0]} samples, {all_states.shape[1]} days")
    
    return all_states, all_new_I


def summarize_forecast(forecasts: np.ndarray, 
                      ci: Tuple[float, float] = (5, 95)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate summary statistics for ensemble forecasts.
    
    Args:
        forecasts: Array of shape (n_samples, days+1, 7) or (n_samples, days)
        ci: Confidence interval percentiles (lower, upper)
    
    Returns:
        median: Median trajectory
        lower: Lower percentile trajectory
        upper: Upper percentile trajectory
    """
    # Calculate percentiles across samples
    median = np.median(forecasts, axis=0)
    lower = np.percentile(forecasts, ci[0], axis=0)
    upper = np.percentile(forecasts, ci[1], axis=0)
    
    return median, lower, upper


def get_forecast_summary(all_states: np.ndarray, 
                        all_new_I: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get comprehensive summary of forecast results.
    
    Args:
        all_states: Array of shape (n_samples, days+1, 7)
        all_new_I: Array of shape (n_samples, days)
    
    Returns:
        Dictionary containing summary statistics for each compartment
    """
    summary = {}
    
    # Compartment names
    compartments = ['S', 'E', 'I', 'H', 'ICU', 'R', 'D']
    
    # Summary for each compartment
    for i, comp in enumerate(compartments):
        comp_data = all_states[:, :, i]  # shape: (n_samples, days+1)
        median, lower, upper = summarize_forecast(comp_data)
        
        summary[f'{comp}_median'] = median
        summary[f'{comp}_lower'] = lower
        summary[f'{comp}_upper'] = upper
        summary[f'{comp}_peak'] = np.max(median)
        summary[f'{comp}_final'] = median[-1]
    
    # Summary for new infections
    median_I, lower_I, upper_I = summarize_forecast(all_new_I)
    summary['new_I_median'] = median_I
    summary['new_I_lower'] = lower_I
    summary['new_I_upper'] = upper_I
    summary['new_I_peak'] = np.max(median_I)
    summary['new_I_total'] = np.sum(median_I)
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Running forecast with uncertainty...")
    
    # Generate forecasts
    all_states, all_new_I = forecast_from_trace(
        trace_path="results/trace.nc",
        days=60,
        n_samples=100
    )
    
    # Get summary statistics
    summary = get_forecast_summary(all_states, all_new_I)
    
    # Print key results
    print("\nForecast Summary:")
    print(f"Peak infected: {summary['I_peak']:.0f}")
    print(f"Peak hospitalized: {summary['H_peak']:.0f}")
    print(f"Peak ICU: {summary['ICU_peak']:.0f}")
    print(f"Total deaths: {summary['D_final']:.0f}")
    print(f"Peak daily infections: {summary['new_I_peak']:.0f}")

