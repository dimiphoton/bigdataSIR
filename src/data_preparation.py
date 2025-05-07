import pandas as pd

def load_data(file_path):
    """Charge les données à partir d'un fichier CSV."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Prétraite les données pour extraire les informations pertinentes."""
    num_positive = data['num_positive']
    num_tested = data['num_tested']
    num_hospitalised = data['num_hospitalised']
    num_cumulative_hospitalizations = data['num_cumulative_hospitalizations']
    num_critical = data['num_critical']
    num_fatalities = data['num_fatalities']

    # Estimer le nombre réel d'individus infectés en tenant compte de la sensibilité du test
    sensitivity = 0.85  # Exemple de valeur de sensibilité
    true_positive_estimate = num_positive / sensitivity

    return {
        'true_positive_estimate': true_positive_estimate,
        'num_tested': num_tested,
        'num_hospitalised': num_hospitalised,
        'num_cumulative_hospitalizations': num_cumulative_hospitalizations,
        'num_critical': num_critical,
        'num_fatalities': num_fatalities
    }
