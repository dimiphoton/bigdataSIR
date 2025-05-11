# seir_model/model.py

import pymc as pm
import numpy as np
from seir_model.simulator import SEIRSimulator

def build_model(data, population, stochastic=True):
    days = len(data)

    with pm.Model() as model:
        # --- Priors ---
        E0 = pm.DiscreteUniform("E0", lower=0, upper=500)
        I0 = pm.DiscreteUniform("I0", lower=0, upper=500)

        beta = pm.Uniform("beta", 0, 2)
        incubation = pm.Uniform("incubation", 1, 5)
        symptomatic = pm.Uniform("symptomatic", 4, 10)

        h_rate = pm.Beta("h_rate", 2, 10)
        h_duration = pm.Uniform("h_duration", 5, 15)

        icu_rate = pm.Beta("icu_rate", 1, 10)
        icu_duration = pm.Uniform("icu_duration", 5, 15)

        mu = pm.Beta("mu", 2, 8)

        # --- Wrap parameters into dict for simulator ---
        params = {
            "beta": beta,
            "incubation": incubation,
            "symptomatic": symptomatic,
            "h_rate": h_rate,
            "h_duration": h_duration,
            "icu_rate": icu_rate,
            "icu_duration": icu_duration,
            "mu": mu,
        }

        # --- Initial state ---
        S0 = population - E0 - I0
        initial_state = [S0, E0, I0, 0, 0, 0, 0]

        # --- Run external simulator ---
        simulator = SEIRSimulator(N=population, params=params, stochastic=stochastic)
        sim_result = simulator.simulate(initial_state, days)

        # --- Extract simulated observables ---
        I_pred = sim_result[:, 2]
        H_pred = sim_result[:, 3]
        ICU_pred = sim_result[:, 4]
        D_pred = sim_result[:, 6]

        # --- Likelihood ---
        sensitivity = 0.8
        tested_fraction = 0.5
        expected_positive = I_pred * tested_fraction * sensitivity

        pm.Poisson("obs_I", mu=expected_positive, observed=data["num_positive"].values)
        pm.Poisson("obs_H", mu=H_pred, observed=data["num_hospitalised"].values)
        pm.Poisson("obs_ICU", mu=ICU_pred, observed=data["num_critical"].values)
        pm.Poisson("obs_D", mu=D_pred, observed=data["num_fatalities"].values)

    return model
