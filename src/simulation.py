import numpy as np
import pandas as pd
import arviz as az
from seir_model import SEIRModel

def simulate_from_posterior(trace, start_day, n_simulations=100, future_days=30, population=1_000_000, initial_state=None):
    # If no initial state is provided, use default values
    if initial_state is None:
        initial_state = {
            'S': population - 10,
            'E': 5,
            'I': 5,
            'Sym': 0,
            'Asym': 0,
            'Hosp': 0,
            'R': 0,
            'D': 0
        }

    posterior = trace.posterior.stack(draws=("chain", "draw"))
    n_available = posterior.draws.size

    simulations = []

    # Loop over posterior samples and run simulations
    for i in range(n_simulations):
        sample_idx = np.random.randint(n_available)
        sample = posterior.isel(draws=sample_idx)

        # Create SEIR model with sampled parameters
        model = SEIRModel(
            beta=float(sample['beta']),
            sigma=float(sample['sigma']),
            p_sym=float(sample['p_sym']),
            gamma=float(sample['gamma']),
            hosp_rate=float(sample['hosp_rate']),
            mu=float(sample['mu']),
            p_test=float(sample['p_test']),
            sens=0.85,  # Sensitivity
            spec=1.0,   # Specificity
            S0=initial_state['S'],
            E0=initial_state['E'],
            I0=initial_state['I'],
            R0=initial_state['R'],
            stochastic=True
        )

        # Set initial state to the state at the end of day 72 (or given start_day)
        state = (
            initial_state['S'], initial_state['E'], initial_state['I'],
            initial_state['Sym'], initial_state['Asym'], initial_state['Hosp'],
            initial_state['R'], initial_state['D']
        )

        # Simulate for the next `future_days` days (after start_day)
        sim_result = []
        for day in range(start_day, start_day + future_days):
            sim_result.append(state)
            derivatives = model.derivatives(*state)
            state = tuple(max(0, s + ds) for s, ds in zip(state, derivatives))

        # Store results in a DataFrame
        sim_df = pd.DataFrame(sim_result, columns=['S', 'E', 'I', 'Sym', 'Asym', 'Hosp', 'R', 'D'])
        sim_df['day'] = np.arange(start_day, start_day + future_days)
        sim_df['simulation'] = i
        simulations.append(sim_df)

    return pd.concat(simulations, ignore_index=True)

# Example usage in a notebook
if __name__ == '__main__':
    # Load trace from your inference results
    trace = az.from_netcdf("posterior_trace.nc")

    # Choose the day to start from (e.g., 72) and the number of future days (e.g., 30)
    start_day = 72
    future_days = 30

    # Simulate future predictions using the posterior
    results = simulate_from_posterior(trace, start_day, n_simulations=100, future_days=future_days)

    # Save to CSV for visualization
    results.to_csv("future_simulations.csv", index=False)

    # Display a quick summary
    print(results.head())
