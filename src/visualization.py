import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results, title):
    """
    Trace les résultats de la simulation du modèle SEIR.

    Paramètres :
    - results : Un dictionnaire contenant les résultats de la simulation pour chaque compartiment.
    - title : Le titre du graphique.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(results['S'], label='Susceptible')
    plt.plot(results['E'], label='Exposed')
    plt.plot(results['I'], label='Infectious')
    plt.plot(results['Sym'], label='Symptomatic')
    plt.plot(results['Asym'], label='Asymptomatic')
    plt.plot(results['Hosp'], label='Hospitalized')
    plt.plot(results['R'], label='Recovered')
    plt.plot(results['D'], label='Deceased')
    plt.plot(results['Observed_Positives'], label='Observed Positives', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

def plot_posterior(trace):
    """
    Trace les distributions postérieures des paramètres.

    Paramètres :
    - trace : Le résultat de l'inférence bayésienne contenant les distributions postérieures.
    """
    sns.pairplot(trace.to_dataframe(), diag_kind='kde')
    plt.suptitle('Posterior Distributions of Parameters', y=1.02)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_log(results, label='I', log_base=10, show_mean=True):
    """
    Plots log-log of daily increase vs total for a given label (e.g., 'I', 'Hosp', 'D').

    Parameters:
        results (pd.DataFrame): Simulation results with 'simulation', 'day', and `label` columns.
        label (str): The column to analyze (default: 'I' for Infected).
        log_base (int): Base of the logarithm (default: 10).
        show_mean (bool): Whether to plot the average trajectory.

    Returns:
        None (displays plot)
    """
    df = results.sort_values(by=['simulation', 'day']).copy()
    df[f'd{label}'] = df.groupby('simulation')[label].diff().fillna(0)

    # Avoid log(0) or negative values
    plot_data = df[(df[label] > 0) & (df[f'd{label}'] > 0)].copy()

    # Log transformation
    log = np.log10 if log_base == 10 else np.log
    plt.figure(figsize=(8, 6))
    for sim_id, sim_data in plot_data.groupby('simulation'):
        plt.plot(log(sim_data[label]), log(sim_data[f'd{label}']), alpha=0.2, color='gray')

    if show_mean:
        mean_label = plot_data.groupby('day')[label].mean()
        mean_dlabel = plot_data.groupby('day')[f'd{label}'].mean()
        valid = (mean_label > 0) & (mean_dlabel > 0)
        plt.plot(
            log(mean_label[valid]),
            log(mean_dlabel[valid]),
            color='blue', label='Mean trajectory', linewidth=2
        )

    plt.xlabel(f"log₁₀(Total {label})")
    plt.ylabel(f"log₁₀(Daily Increase in {label})")
    plt.title(f"Daily Increase vs Total {label} (log-log scale)")
    plt.grid(True, which="both", ls="--")
    if show_mean:
        plt.legend()
    plt.tight_layout()
    plt.show()
