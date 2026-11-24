"""
Configuration module for SEIR-HICURD epidemic modeling.

This module centralizes all project-wide constants including population size,
initial state values, simulation duration, and observed data.

Variables:
    N: Total population size
    initial_state: Initial state vector [S, E, I, H, ICU, R, D]
    days: Number of days to simulate
    observed_cases: Array of observed daily new infections
"""

import numpy as np

# Total population size
N = 1_000_000

# Initial state vector [S, E, I, H, ICU, R, D]
# S: Susceptible, E: Exposed, I: Infected, H: Hospitalized
# ICU: Intensive Care, R: Recovered, D: Deceased
# By adding a decimal point, you create a float array
initial_state = np.array([N - 15., 5., 10., 0., 0., 0., 0.])

# Number of days to simulate
days = 30

# Observed daily new infections (synthetic data for demonstration)
# In practice, this would be loaded from a data file
observed_cases = np.array([
    5, 8, 12, 18, 25, 35, 45, 60, 80, 100,  # Days 1-10
    120, 140, 160, 180, 200, 220, 240, 260, 280, 300,  # Days 11-20
    320, 340, 360, 380, 400, 420, 440, 460, 480, 500   # Days 21-30
])

# Alternative: Load observed data from file
# Uncomment and modify the path as needed:
# import pandas as pd
# data = pd.read_csv("data/observed_cases.csv")
# observed_cases = data["daily_cases"].to_numpy()
# days = len(observed_cases)
