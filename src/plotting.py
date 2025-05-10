# utils/plotting.py

import matplotlib.pyplot as plt

def plot_infected_vs_total_infected(results):
    """Plot the daily increase in infected vs. total infected."""
    infected = [result[2] for result in results]  # I
    total_infected = [result[2] for result in results]  # Total I
    daily_increase = [infected[i] - infected[i - 1] for i in range(1, len(infected))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(total_infected[1:], daily_increase, label='Daily Increase', color='b')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Total Infected')
    plt.ylabel('Daily Increase')
    plt.title('Daily Increase in Infected vs Total Infected (Log Scale)')
    plt.legend()
    plt.show()
