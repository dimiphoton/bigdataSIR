import json
import pandas as pd

def load_inferred_parameters(file_path='inferred_parameters.json'):
    """
    Load inferred parameters from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file containing inferred parameters.

    Returns:
    - dict: Inferred parameters.
    """
    with open(file_path, 'r') as f:
        parameters = json.load(f)
    return parameters

def simulate_epidemic(model, initial_conditions, time_steps, parameters_file='inferred_parameters.json'):
    """
    Simulate the epidemic spread using inferred parameters.

    Parameters:
    - model (EpidemiologicalModel): The epidemiological model instance.
    - initial_conditions (dict): Initial number of individuals in each state.
    - time_steps (int): Number of time steps to simulate.
    - parameters_file (str): Path to the JSON file containing inferred parameters.

    Returns:
    - pd.DataFrame: Simulated data.
    """
    # Load inferred parameters
    parameters = load_inferred_parameters(parameters_file)

    # Update model parameters
    model.beta = parameters['beta']
    model.sigma = parameters['sigma']
    model.gamma = parameters['gamma']
    model.h = parameters['h']
    model.mu = parameters['mu']

    # Simulate the epidemic spread
    simulated_data = model.simulate(initial_conditions, time_steps)
    return simulated_data
