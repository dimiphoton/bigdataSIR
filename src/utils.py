
import pandas as pd
import numpy as np


def load_data() -> pd.DataFrame:
    """
    Load and preprocess the dataset from a CSV file.
    
    This function reads a CSV file containing daily data, parses the 'Day' column
    as datetime, and returns a DataFrame with the relevant columns.
    
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    # Load the dataset
    return pd.read_csv("../data/data.csv",parse_dates=["Day"])