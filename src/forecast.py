# src/forecast.py

from simulator import simulate
from posterior_utils import load_trace, get_most_probable_values
from config import N, initial_state

def run_forecast(trace_path="trace.nc", days=60):
    trace = load_trace(trace_path)
    params = get_most_probable_values(trace, [
        "beta", "incubation", "symptomatic", "h_rate",
        "h_duration", "icu_rate", "icu_duration", "mu"
    ])

    states, new_I = simulate(initial_state, params, N, days)
    return states  # shape: (days+1, 7)

if __name__ == "__main__":
    forecast = run_forecast()
