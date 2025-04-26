import plotly.express as px
import pandas as pd

def create_dynamic_visualization(dataframe,x,y):
    """
    Crée le graphe nouvelles infections / total infectés à partir des colonnes spécifiées d'un DataFrame pandas.

    Paramètres :
    df (pd.DataFrame) : Le DataFrame contenant les données.
    x (str) : Le nom de la colonne à utiliser pour l'axe x.
    y (str) : Le nom de la colonne à utiliser pour l'axe y.

    Retourne :
    None
    """

    # Create the interactive log log plot with dataframe[x] as x and dataframe[y] as y

    fig = px.scatter(dataframe, x, y, log_x=True, log_y=True)

    # Show the plot
    fig.show()

# Example usage
# simulated_data = pd.DataFrame({'Total Infected': [...], 'New Infected': [...]})
# create_dynamic_visualization(simulated_data)
