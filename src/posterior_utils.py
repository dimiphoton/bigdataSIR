# src/posterior_utils.py

import arviz as az
import numpy as np
import xarray as xr

def load_trace(path="trace.nc"):
    return az.from_netcdf(path)

def summarize_params(trace, parameters=None, hdi_prob=0.9):
    summary = az.summary(trace, var_names=parameters, hdi_prob=hdi_prob)
    return summary[["mean", "hdi_5%", "hdi_95%"]]

def get_most_probable_values(trace, parameters):
    """Get MAP (approximation: mean of posterior)"""
    posterior = trace.posterior
    return {param: float(posterior[param].mean().values) for param in parameters}
