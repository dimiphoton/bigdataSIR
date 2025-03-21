import plotly.express as px
import pandas as pd

def create_dynamic_visualization(simulated_data):
    """
    Create a dynamic visualization of total infected vs. new infected.

    Parameters:
    - simulated_data (pd.DataFrame): Simulated data with 'Total Infected' and 'New Infected' columns.
    """
    # Ensure the dataframe has the necessary columns
    if 'Total Infected' not in simulated_data.columns or 'New Infected' not in simulated_data.columns:
        raise ValueError("Simulated data must contain 'Total Infected' and 'New Infected' columns.")

    # Create the interactive plot
    fig = px.scatter(simulated_data, x='Total Infected', y='New Infected',
                     title='Total Infected vs. New Infected',
                     labels={'Total Infected': 'Total Infected Individuals', 'New Infected': 'New Infections'})

    # Show the plot
    fig.show()

# Example usage
# simulated_data = pd.DataFrame({'Total Infected': [...], 'New Infected': [...]})
# create_dynamic_visualization(simulated_data)
