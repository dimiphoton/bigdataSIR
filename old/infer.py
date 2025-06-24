import pymc as pm
import arviz as az
import numpy as np
from model import build_model
from config import observed_data, N, initial_state

def run_inference(output_path="trace.nc", draws=1000, tune=1000, chains=4):
    model = build_model(observed_data, N, initial_state)

    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9)
        az.to_netcdf(trace, output_path)
        print(f"Saved trace to {output_path}")

if __name__ == "__main__":
    run_inference()
