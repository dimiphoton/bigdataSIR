"""
Training script for SEIR-HICURD Bayesian inference.

This script loads observed data, builds the PyMC model using build_model(),
runs MCMC sampling with pm.sample(), and saves the trace to disk as trace.nc
using ArviZ.

Usage:
    python src/train.py
    python src/train.py --config config.yaml
"""

import numpy as np
import pymc as pm
import arviz as az
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from build_model import build_model
from simulator import get_default_initial_state, get_default_params


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to YAML config file (optional)
    
    Returns:
        Dictionary containing configuration parameters
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'N': 10000,  # Population size
            'days': 30,  # Number of days to simulate
            'sampling': {
                'draws': 1000,
                'tune': 1000,
                'chains': 4,
                'random_seed': 42
            },
            'output': {
                'trace_file': 'trace.nc',
                'output_dir': 'results'
            }
        }
    
    return config


def generate_synthetic_data(N: int, days: int,
                           initial_state: np.ndarray) -> np.ndarray:
    """
    Generate synthetic observed data for demonstration.
    
    Args:
        N: Population size
        days: Number of days
        initial_state: Initial state vector
    
    Returns:
        Array of observed daily positive tests
    """
    # Use default parameters to generate realistic data
    params = get_default_params()
    
    # Run simulation
    from simulator import simulate
    states, new_I = simulate(initial_state, params, N, days)
    
    # Apply testing model: new_I × 0.5 × 0.8 = new_I × 0.4
    expected_positives = new_I * 0.4
    
    # Add some noise to make it realistic
    observed_cases = np.random.poisson(expected_positives)
    
    return observed_cases


def load_observed_data(data_path: Optional[str] = None,
                      config: Dict[str, Any] = None) -> np.ndarray:
    """
    Load observed data from file or generate synthetic data.
    
    Args:
        data_path: Path to data file (optional)
        config: Configuration dictionary
    
    Returns:
        Array of observed daily positive tests
    """
    if data_path and Path(data_path).exists():
        # Load from file (assuming CSV or similar)
        data = np.loadtxt(data_path)
        return data
    else:
        # Generate synthetic data for demonstration
        N = config.get('N', 10000)
        days = config.get('days', 30)
        initial_state = get_default_initial_state(N)
        
        print("No data file provided. Generating synthetic data...")
        return generate_synthetic_data(N, days, initial_state)


def run_inference(observed_cases: np.ndarray,
                 initial_state: np.ndarray,
                 N: int,
                 days: int,
                 sampling_config: Dict[str, Any]) -> az.InferenceData:
    """
    Run MCMC inference using PyMC.
    
    Args:
        observed_cases: Array of observed daily positive tests
        initial_state: Initial state vector
        N: Population size
        days: Number of days
        sampling_config: Sampling configuration
    
    Returns:
        ArviZ InferenceData object containing the trace
    """
    print("Building PyMC model...")
    model = build_model(observed_cases, initial_state, N, days)
    
    print("Running MCMC sampling...")
    with model:
        # Set random seed for reproducibility
        random_seed = sampling_config.get('random_seed', 42)
        
        # Run sampling with progress bar
        trace = pm.sample(
            draws=sampling_config.get('draws', 1000),
            tune=sampling_config.get('tune', 1000),
            chains=sampling_config.get('chains', 4),
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True
        )
    
    return trace


def save_results(trace: az.InferenceData,
                output_config: Dict[str, Any]) -> None:
    """
    Save inference results to disk.
    
    Args:
        trace: ArviZ InferenceData object
        output_config: Output configuration
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_config.get('output_dir', 'results'))
    output_dir.mkdir(exist_ok=True)
    
    # Save trace to NetCDF file
    trace_file = output_config.get('trace_file', 'trace.nc')
    trace_path = output_dir / trace_file
    
    print(f"Saving trace to {trace_path}...")
    trace.to_netcdf(trace_path)
    
    # Save summary statistics
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(str(az.summary(trace)))
    
    print(f"Results saved to {output_dir}/")
    print(f"  - Trace: {trace_file}")
    print(f"  - Summary: summary.txt")


def main():
    """
    Main function to run the training pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SEIR-HICURD Bayesian inference')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str, help='Path to observed data file')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.seed:
        config['sampling']['random_seed'] = args.seed
    
    # Extract parameters
    N = config['N']
    days = config['days']
    sampling_config = config['sampling']
    output_config = config['output']
    
    print("Configuration:")
    print(f"  Population size: {N}")
    print(f"  Simulation days: {days}")
    print(f"  Sampling draws: {sampling_config['draws']}")
    print(f"  Sampling tune: {sampling_config['tune']}")
    print(f"  Random seed: {sampling_config['random_seed']}")
    
    # Set random seed for reproducibility
    np.random.seed(sampling_config['random_seed'])
    
    # Load or generate observed data
    observed_cases = load_observed_data(args.data, config)
    print(f"Loaded {len(observed_cases)} days of observed data")
    print(f"Total observed cases: {np.sum(observed_cases)}")
    
    # Set up initial state
    initial_state = get_default_initial_state(N)
    print(f"Initial state: {initial_state}")
    
    # Run inference
    trace = run_inference(observed_cases, initial_state, N, days, 
                         sampling_config)
    
    # Save results
    save_results(trace, output_config)
    
    # Print summary
    print("\nInference completed successfully!")
    print("Summary statistics:")
    print(az.summary(trace))


if __name__ == "__main__":
    main()
