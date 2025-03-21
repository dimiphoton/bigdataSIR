from data_processing import load_data, preprocess_data
from model import EpidemiologicalModel
from bayesian_inference import infer_parameters
from simulation import simulate_epidemic, visualize_simulation, create_dynamic_visualization

def main():
    # Load and preprocess data
    data = load_data('data.csv')
    preprocessed_data = preprocess_data(data)

    # Infer model parameters and save them to a file
    infer_parameters(preprocessed_data)

    # Initialize the model
    model = EpidemiologicalModel(beta=0, sigma=0, gamma=0, h=0, mu=0)

    # Simulate the epidemic spread using inferred parameters
    initial_conditions = {
        'Susceptible': 990,
        'Exposed': 5,
        'Infected': 5,
        'Hospitalized': 0,
        'Recovered': 0,
        'Deceased': 0
    }
    time_steps = 100
    simulated_data = simulate_epidemic(model, initial_conditions, time_steps)

    # Calculate total and new infected for dynamic visualization
    simulated_data['Total Infected'] = simulated_data['Infected'].cumsum()
    simulated_data['New Infected'] = simulated_data['Infected']

    # Visualize the simulation
    visualize_simulation(simulated_data)

    # Create dynamic visualization
    create_dynamic_visualization(simulated_data)

if __name__ == "__main__":
    main()
