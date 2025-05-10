import numpy as np
import pandas as pd
import arviz as az
from model import SEIRModel

def simulate_from_trace(trace, population, days_ahead=30, samples=200):
    posterior = trace.posterior
    total_days = posterior.sizes["chain"] * posterior.sizes["draw"]
    selected_idxs = np.random.choice(total_days, size=samples, replace=False)

    all_simulations = []

    for flat_idx in selected_idxs:
        i_chain = flat_idx // posterior.sizes["draw"]
        i_draw = flat_idx % posterior.sizes["draw"]

        # Extract parameters
        params = {k: float(posterior[k].values[i_chain, i_draw]) for k in [
            "beta", "sigma", "gamma", "hosp_rate", "icu_rate",
            "death_rate", "recov_H", "recov_ICU"
        ]}

        seir = SEIRModel(population=population, days=trace.constant_data["days"] + days_ahead,
                         **params, stochastic=True)
        sim = seir.simulate()
        sim["sample"] = flat_idx
        all_simulations.append(sim)

    df_simulations = pd.concat(all_simulations)
    return df_simulations

def summarize_simulations(df_sim, days):
    summary = {}
    compartments = ["I", "H", "ICU", "D"]

    for var in compartments:
        grouped = df_sim.groupby("day")[var]
        summary[f"{var}_p5"] = grouped.quantile(0.05)
        summary[f"{var}_median"] = grouped.median()
        summary[f"{var}_p95"] = grouped.quantile(0.95)

    return pd.DataFrame(summary).reset_index()
