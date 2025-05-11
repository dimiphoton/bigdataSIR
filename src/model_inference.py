import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# Load data
data = pd.read_csv("data.csv")
I_obs = data["num_positive"].values
N = 1_000_000
T = len(I_obs)

with pm.Model() as model:
    # Priors (you can tune these)
    beta = pm.Uniform("beta", 0.1, 1.0)
    sigma = pm.Uniform("sigma", 1/5, 1/1)   # incubation
    gamma_I = pm.Uniform("gamma_I", 1/10, 1/4)  # symptomatic recovery
    h_rate = pm.Beta("h_rate", 2, 5)
    gamma_H = pm.Uniform("gamma_H", 1/14, 1/4)
    icu_rate = pm.Beta("icu_rate", 2, 5)
    gamma_ICU = pm.Uniform("gamma_ICU", 1/20, 1/5)
    mu = pm.Beta("mu", 2, 5)

    # Initial state
    init = np.array([N - 10, 5, 5, 0, 0, 0, 0])

    # Deterministic simulator wrapper
    from seir_model import SEIRSimulator
    def run_model(beta, sigma, gamma_I, h_rate, gamma_H, icu_rate, gamma_ICU, mu):
        sim = SEIRSimulator(N, {
            "beta": beta,
            "sigma": sigma,
            "gamma_I": gamma_I,
            "h_rate": h_rate,
            "gamma_H": gamma_H,
            "icu_rate": icu_rate,
            "gamma_ICU": gamma_ICU,
            "mu": mu
        }, stochastic=False)
        return sim.simulate(init, T)[:, 2]  # Return only I

    I_pred = pm.Deterministic("I_pred", pm.math.stack([run_model(beta, sigma, gamma_I, h_rate, gamma_H, icu_rate, gamma_ICU, mu)]))

    # Observation model: test sensitivity ~75%
    I_reported = pm.Binomial("I_obs", n=I_pred, p=0.75, observed=I_obs)

    trace = pm.sample(1000, tune=1000, target_accept=0.9)
    az.to_netcdf(trace, "trace.nc")
