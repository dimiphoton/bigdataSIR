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
