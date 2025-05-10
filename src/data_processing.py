# utils/data_processing.py

import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the dataset for use in the SEIR model."""
    # Assuming the data has columns: Day, num_positive, num_tested, num_hospitalised, num_cumulative_hospitalizations, num_critical, num_fatalities
    return data[['Day', 'num_positive', 'num_hospitalised', 'num_cumulative_hospitalizations', 'num_critical', 'num_fatalities']]
