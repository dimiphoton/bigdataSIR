import pymc as pm
import numpy as np
import pandas as pd
from model import SEIRModel

def build_pymc_model(data, population, stochastic=True):
    days = len(data)
    with pm.Model() as model:
        # Priors
        beta = pm.Uniform("beta", 0, 1)
        sigma = pm.Uniform("sigma", 0.1, 1)  # Incubation rate
        gamma = pm.Uniform("gamma", 0.1, 1)  # Recovery from I
        hosp_rate = pm.Uniform("hosp_rate", 0, 0.5)
        icu_rate = pm.Uniform("icu_rate", 0, 0.5)
        death_rate = pm.Uniform("death_rate", 0, 0.5)
        recov_H = pm.Uniform("recov_H", 0.1, 1)
        recov_ICU = pm.Uniform("recov_ICU", 0.1, 1)

        params = {
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            "hosp_rate": hosp_rate,
            "icu_rate": icu_rate,
            "death_rate": death_rate,
            "recov_H": recov_H,
            "recov_ICU": recov_ICU,
        }

        # Deterministic model simulation
        seir = SEIRModel(population=population, days=days, **params, stochastic=False)
        sim = seir.simulate()

        # Observed infected (partially observed via test sensitivity)
        test_sensitivity = 0.75
        observed_I = pm.Binomial("obs_I", n=sim["I"], p=test_sensitivity, observed=data["num_positive"])

        # Observed H, ICU, D assumed to be true counts
        pm.Poisson("obs_H", mu=sim["H"], observed=data["num_hospitalised"])
        pm.Poisson("obs_ICU", mu=sim["ICU"], observed=data["num_critical"])
        pm.Poisson("obs_D", mu=sim["D"], observed=data["num_fatalities"])
    
    return model

def run_inference(model, draws=1000, tune=1000, target_accept=0.9):
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, return_inferencedata=True)
    return trace
