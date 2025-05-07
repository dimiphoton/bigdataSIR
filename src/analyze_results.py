import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
HOSPITAL_CAPACITY = 1500
ICU_CAPACITY = 300

def plot_trajectories(df, var='Hosp', threshold=None, title=''):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='day', y=var, hue='simulation', legend=False, alpha=0.3)
    
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()

def summarize_thresholds(df, hosp_thresh=HOSPITAL_CAPACITY, icu_thresh=ICU_CAPACITY):
    summary = {}
    grouped = df.groupby('simulation')
    
    summary['total_simulations'] = len(grouped)
    summary['hosp_breaches'] = sum(grouped['Hosp'].max() > hosp_thresh)
    summary['icu_breaches'] = sum(grouped['Sym'].max() > icu_thresh)  # Replace with ICU if modeled separately

    summary['hosp_breach_rate'] = summary['hosp_breaches'] / summary['total_simulations']
    summary['icu_breach_rate'] = summary['icu_breaches'] / summary['total_simulations']

    return summary

if __name__ == "__main__":
    df = pd.read_csv("simulated_trajectories.csv")

    print("Summarizing threshold breaches...")
    summary = summarize_thresholds(df)
    for k, v in summary.items():
        print(f"{k}: {v}")

    plot_trajectories(df, var="Hosp", threshold=HOSPITAL_CAPACITY, title="Hospital Occupancy")
    plot_trajectories(df, var="Sym", threshold=ICU_CAPACITY, title="ICU Cases (proxy: severe symptomatics)")
