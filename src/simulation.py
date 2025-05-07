import numpy as np
import pandas as pd
import arviz as az
from seir_model import SEIRModel

def simulate_from_posterior(trace, n_simulations=100, days=60, population=1_000_000, initial_state=None,
                             sens=0.85, spec=1.0):
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

    for i in range(n_simulations):
        sample_idx = np.random.randint(n_available)
        sample = posterior.isel(draws=sample_idx)

        model = SEIRModel(
            beta=float(sample['beta']),
            sigma=float(sample['sigma']),
            p_sym=float(sample['p_sym']),
            gamma=float(sample['gamma']),
            hosp_rate=float(sample['hosp_rate']),
            mu=float(sample['mu']),
            p_test=float(sample['p_test']),
            sens=sens,
            spec=spec,
            S0=initial_state['S'],
            E0=initial_state['E'],
            I0=initial_state['I'],
            R0=initial_state['R'],
            stochastic=True
        )

        state = (
            initial_state['S'], initial_state['E'], initial_state['I'],
            initial_state['Sym'], initial_state['Asym'], initial_state['Hosp'],
            initial_state['R'], initial_state['D']
        )

        sim_result = []
        for day in range(days):
            sim_result.append(state)
            state = model.step(state)

        sim_df = pd.DataFrame(sim_result, columns=['S', 'E', 'I', 'Sym', 'Asym', 'Hosp', 'R', 'D'])
        sim_df['day'] = np.arange(days)
        sim_df['simulation'] = i
        simulations.append(sim_df)

    return pd.concat(simulations, ignore_index=True)


# Example usage
if __name__ == '__main__':
    trace = az.from_netcdf("posterior_trace.nc")
    results = simulate_from_posterior(trace, n_simulations=100, days=60)

    # Save for later visualization
    results.to_csv("simulated_trajectories.csv", index=False)
