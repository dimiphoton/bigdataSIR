# bigdataSIR
# Epidemiological Modeling Project

## Objectives

-  [ ] Develop a state-based model to simulate the spread of an epidemic.
-  [ ] Implement a Bayesian approach to estimate model parameters.
-  [ ] Create an interactive simulator to visualize epidemic dynamics.
-  [ ] Provide insights into the impact of interventions on epidemic spread.

## Methods

-  **Model Definition**:
  - [ ] Define a compartmental model (e.g., SEIRHD) with states: Susceptible, Exposed, Infected, Hospitalized, Recovered, Deceased.
  - [ ] Represent the model as a state graph to illustrate transitions between compartments.
-  **Bayesian Parameter Estimation**:
  - [ ] Specify prior distributions for model parameters.
  - [ ] Use Markov Chain Monte Carlo (MCMC) to sample from posterior distributions.
  - [ ] Incorporate uncertainty in test sensitivity and specificity.
-  **Data Integration**:
  - [ ] Utilize provided epidemiological data to inform the model.
  - [ ] Validate model outputs against real-world observations.
-  **Simulator Development**:
  - [ ] Develop a simple interactive simulator using Python (e.g., with Streamlit or Dash).
  - [ ] Allow users to adjust parameters and visualize the impact on epidemic spread.

## Expected Outcomes

-  [ ] A clear and well-documented state graph representing the epidemiological model.
-  [ ] Estimated model parameters with credible intervals using Bayesian inference.
-  [ ] An interactive simulator hosted on GitHub or another platform, easy to launch and use.
-  [ ] Insights into how different interventions (e.g., increased testing, earlier hospitalization) affect epidemic outcomes.

## Project Structure

-  `data/`: Directory containing epidemiological data files.

- `src/`: Source code for the model and simulator.

- `model.py`: Implementation of the epidemiological model.
- `bayesian_inference.py`: Bayesian parameter estimation.
- `simulator.py`: Interactive simulator code.

-  `README.md`: Project documentation (this file).

-  `requirements.txt`: List of Python dependencies.

## Getting Started

-  Clone the repository.
-  Install dependencies using `pip install -r requirements.txt`.
-  Run the simulator using the provided script.
