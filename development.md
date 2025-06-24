# SEIR Project Module Plan

This document defines the module structure, responsibilities, and planned functionalities for the SEIR inference and forecasting project.

## 0. Shared Configuration

### `config.py`

* Global constants: population size `N`, initial state `initial_state`
* Time windows: `train_days`, `forecast_days`
* Paths for input/output files

---

## 1. SEIR Simulation Engine (Deterministic)

### `simulator.py`

* `seir_step(state, params, N) -> next_state, new_I`

  * Advances SEIR state by 1 day.
* `simulate(initial_state, params, N, days) -> states, new_I`

  * Runs simulation over multiple days.

#### Optional Additions:

* `simulate_with_intervention()` to add time-varying beta or policies.
* Add noise to outputs for testing robustness.

---

## 2. Probabilistic Model

### `build_model.py`

* `build_model(data) -> pymc.Model`

  * Constructs PyMC model using observed data (e.g., hospitalizations).

#### Optional:

* Allow toggling between different model variants (e.g. simplified SEIR).
* Switch between deterministic and stochastic observation models.

---

## 3. Inference Pipeline

### `infer.py`

* `run_inference(model, draws=1000, tune=1000) -> trace`
* `save_trace(trace, path)`
* `load_trace(path) -> arviz.InferenceData`
* `summarize_trace(trace) -> dict of posterior stats`

#### Optional:

* Compare priors and posteriors visually.
* Save diagnostic plots.

---

## 4. Forecasting

### `forecast.py`

* `load_posterior_samples(trace_path, param_names, n_samples) -> List[Dict]`
* `forecast_from_trace(trace_path, days, n_samples) -> forecasts`
* `summarize_forecast(forecasts, ci=(5,95)) -> median, lower, upper`

#### Optional:

* Support batch forecasting on multiple scenarios.
* Track memory/performance usage.

---

## 5. Data Utilities

### `io_utils.py`

* `load_csv_data(filepath) -> dict`
* `clean_observation_data(raw_df) -> dict`
* `align_data_to_model(data, start_day, duration)`

---

## 6. Plotting

### `viz.py`

* `plot_trajectory(state_array)`
* `plot_forecast_with_uncertainty(median, lower, upper, observed=None)`
* `plot_parameter_distributions(trace)`

---

## 7. Notebooks

| Notebook                     | Purpose                                   |
| ---------------------------- | ----------------------------------------- |
| `01_simulate_baseline.ipynb` | Validate simulation from fixed parameters |
| `02_infer_parameters.ipynb`  | Fit PyMC model to data                    |
| `03_forecast.ipynb`          | Sample forecasts from posterior samples   |
| `04_plot_results.ipynb`      | Compare observed data vs simulations      |

---

## Future Extensions (Ideas)

* Interactive dashboard (e.g. Streamlit or Voila)
* MPC control logic for policy planning
* Multi-region model support (shared ICU, split population)
* Custom likelihood functions for real hospital data (lags, censoring)
* Model versioning with config files or YAML schema
* CLI runner to automate batch jobs

---

## Development Notes

* Stick to pure functions for maximum testability.
* Keep `.py` scripts minimal and well-scoped.
* Avoid hard-coded paths and variables.
* Document parameter units and constraints in `build_model.py`.
* Every module should be importable and testable without notebook dependencies.
