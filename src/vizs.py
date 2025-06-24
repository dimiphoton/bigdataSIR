import plotly.express as px
import numpy as np

def create_dynamic_visualization(x, y, x_label="x", y_label="y"):
    """
    Create an interactive log-log scatter plot from two arrays.

    Parameters:
    x (array-like): Values for the x-axis.
    y (array-like): Values for the y-axis.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.

    Returns:
    None
    """
    fig = px.scatter(x=x, y=y, log_x=True, log_y=True, labels={"x": x_label, "y": y_label})
    fig.show()

# Example usage
# x = np.array([...])
# y = np.array([...])
# create_dynamic_visualization(x, y, "Total Infected", "New Infected")
