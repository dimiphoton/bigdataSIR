import pandas as pd

def load_data(file_path):
    """
    Load epidemiological data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the data by handling missing values and transforming it.

    Parameters:
    - data (pd.DataFrame): Raw data.

    Returns:
    - pd.DataFrame: Preprocessed data.
    """
    # Example preprocessing steps
    data = data.dropna()  # Drop missing values
    data['Date'] = pd.to_datetime(data['Date'])  # Convert date column to datetime
    return data
