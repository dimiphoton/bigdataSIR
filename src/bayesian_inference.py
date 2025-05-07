import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from model import SEIRModel

def run_inference(data, population=1_000_000):
    days = len(data)
    observed_positives = data['num_positive'].values
    observed_hospitalised = data['num_hospitalised'].values

    with pm.Model() as model:
        # Priors
        beta = pm.Normal('beta', mu=0.3, sigma=0.1)
        sigma = pm.Normal('sigma', mu=1/3.0, sigma=0.2)
        p_sym = pm.Uniform('p_sym', lower=0.5, upper=0.7)
        gamma = pm.Normal('gamma', mu=1/7.0, sigma=0.1)
        hosp_rate = pm.Beta('hosp_rate', alpha=2, beta=20)
        mu = pm.Beta('mu', alpha=2, beta=50)
        p_test = pm.Uniform('p_test', lower=0.5, upper=1.0)
        sens = pm.Uniform('sens', lower=0.7, upper=0.9)

        # Initial conditions
        E0 = pm.DiscreteUniform('E0', lower=1, upper=100)
        I0 = pm.DiscreteUniform('I0', lower=1, upper=100)
        S0 = population - E0 - I0
        R0 = 0

        seir = SEIRModel(beta, sigma, p_sym, gamma, hosp_rate, mu, S0, E0, I0, R0, p_test, sens, 1.0)
        sim = seir.simulate(days)

        # Likelihoods
        pm.Poisson('obs_positive', mu=sim['positive_tests'], observed=observed_positives)
        pm.Normal('obs_hospitalised', mu=sim['hospitalised'], sigma=10, observed=observed_hospitalised)

        trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)
        az.to_netcdf(trace, 'posterior_trace.nc')

    return trace

if __name__ == "__main__":
    data = pd.read_csv("data/epidemic_data.csv")
    trace = run_inference(data)
    print(trace.posterior)
