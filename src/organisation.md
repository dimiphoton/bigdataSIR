# Project Name: SEIR‑HICURD Epidemic Model

## 1. simulate.py
### Function: seir_step
- **Goal:** One time‐step update of SEIR‑HICURD compartments (JAX-compatible).
- **Args:**
  - state: jnp.array(7) — [S, E, I, H, ICU, R, D]
  - params: dict — model parameters (β, σ, γ, h, icu, γ_h, γ_icu, μ_h, μ_icu)
  - N: float — total population
- **Returns:**  
  `(new_state: jnp.array(7), new_I: float)`

---

### Function: simulate_symbolic
- **Goal:** Run `seir_step` over `days`, producing trajectories for inference.
- **Args:**
  - initial_state: jnp.array(7)
  - params: dict
  - N: float
  - days: int
- **Returns:**  
  `(trajectory: jnp.array(days+1,7), new_infections: jnp.array(days))`

---

### Function: simulate_numpy
- **Goal:** Deterministic NumPy version for plotting/validation.
- **Args:** Same as `simulate_symbolic` but arrays are numpy.
- **Returns:** Same as `simulate_symbolic`, but numpy arrays.

---

### Function: seir_step_numpy
- **Goal:** Single NumPy time‑step update.
- **Args:** Same semantics as `seir_step`, dtype = numpy.
- **Returns:** `(new_state: numpy.ndarray(7), new_I: float)`

---

### Function: get_simulation_summary
- **Goal:** Compute key summary stats from trajectory.
- **Args:**
  - trajectory: numpy.ndarray(days+1,7)
  - new_infections: numpy.ndarray(days)
- **Returns:** dict with:
  - peak_infected, peak_hospitalized, peak_icu
  - total_deaths, total_recovered, total_infections
  - peak_daily_infections

---

## 2. build_model_numpyro.py
### Function: seir_model
- **Goal:** Define NumPyro probabilistic model using JAX simulator.
- **Args:**
  - observed_cases: jnp.array(days,)
  - initial_state: jnp.array(7,)
  - N: float
  - days: int
- **Returns:** Implicit through `numpyro.sample`; produces posterior samples and deterministics:
  - peak_I, peak_H, peak_ICU, total_deaths

---

## 3. Optional future files
- `inference.py` → calls `seir_model` with MCMC/SVI, returns posterior
- `forecast.py` → propagates posterior to future predictions
- `data_loader.py` → handles CSV → numpy/jnp array conversion
- `utils.py` → plotting functions (`plot_trace`, `plot_forecast`)