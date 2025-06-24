#!/usr/bin/env python3
"""
Test script for improved model and simulator.
"""

import numpy as np
from src.model import SEIHRDModel, ModelParameters
from src.simulator import ModelSimulator, SimulationScenario

def test_model():
    """Test the improved model."""
    print("Testing SEIHRD Model...")
    
    # Create model parameters
    params = ModelParameters(
        beta=0.3, sigma=0.2, gamma=0.1, h=0.05, mu=0.01, N=10000
    )
    
    # Create model
    initial_conditions = {'S': 9990, 'E': 5, 'I': 5, 'H': 0, 'R': 0, 'D': 0}
    time_points = np.linspace(0, 100, 101)
    
    model = SEIHRDModel(
        initial_conditions=initial_conditions,
        time_points=time_points,
        population_size=10000
    )
    
    # Solve ODE system
    results = model.solve_ode_system(params)
    
    print(f"✓ Model created successfully")
    print(f"✓ ODE system solved")
    print(f"✓ Results shape: {len(results['S'])} time points")
    
    return results

def test_simulator():
    """Test the improved simulator."""
    print("\nTesting Model Simulator...")
    
    # Create simulator
    simulator = ModelSimulator(population_size=10000)
    
    # Run baseline simulation
    baseline_results = simulator.run_baseline_simulation(duration_days=50)
    
    print(f"✓ Simulator created successfully")
    print(f"✓ Baseline simulation completed")
    print(f"✓ Peak infected: {np.max(baseline_results['I']):.0f}")
    
    return baseline_results

if __name__ == "__main__":
    print("Testing Improved Model and Simulator")
    print("=" * 40)
    
    try:
        model_results = test_model()
        simulator_results = test_simulator()
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise 