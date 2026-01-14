import pandas as pd
from scipy import stats
import numpy as np

DATA_FILE = 'snakebot_rampenv_results.csv'
SIGNIFICANCE_LEVEL = 0.05

# Load data
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully from {DATA_FILE}. Total rows: {len(df)}")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("-" * 50)
except FileNotFoundError:
    print(f"Error: The file '{DATA_FILE}' was not found.")
    exit()

undulation_data = df[df['gait'] == 'UNDULATION']
sidewinding_data = df[df['gait'] == 'SIDEWINDING']

if undulation_data.empty or sidewinding_data.empty:
    print("Error: Could not find data for both 'UNDULATION' and 'SIDEWINDING' gaits.")
    exit()

def analyze_metric(metric_name, data_group1, data_group2, group1_name="UNDULATION", group2_name="SIDEWINDING", alpha=SIGNIFICANCE_LEVEL):
    print(f"\n--- Analysis for: {metric_name} ---")

    values_group1 = data_group1[metric_name].dropna()
    values_group2 = data_group2[metric_name].dropna()

    n1 = len(values_group1)
    n2 = len(values_group2)

    if n1 == 0 or n2 == 0:
        print(f"  Skipping {metric_name}: Insufficient data for one or both groups.")
        return

    print(f"  {group1_name} (n={n1}): Mean={values_group1.mean():.4f}, Median={values_group1.median():.4f}, Std Dev={values_group1.std():.4f}")
    print(f"  {group2_name} (n={n2}): Mean={values_group2.mean():.4f}, Median={values_group2.median():.4f}, Std Dev={values_group2.std():.4f}")

    # Normality
    shapiro_g1_p = stats.shapiro(values_group1)[1] if n1 >= 3 else 1.0
    shapiro_g2_p = stats.shapiro(values_group2)[1] if n2 >= 3 else 1.0

    print(f"\n  Normality (Shapiro-Wilk):")
    print(f"    {group1_name}: p-value={shapiro_g1_p:.4f} {'(NOT Normal)' if shapiro_g1_p < alpha else '(Normal)'}")
    print(f"    {group2_name}: p-value={shapiro_g2_p:.4f} {'(NOT Normal)' if shapiro_g2_p < alpha else '(Normal)'}")

    # Variance homogeneity
    levene_stat, levene_p = stats.levene(values_group1, values_group2)
    print(f"\n  Variance Homogeneity (Levene's Test):")
    print(f"    Statistic={levene_stat:.4f}, p-value={levene_p:.4f} {'(UNEQUAL Variances)' if levene_p < alpha else '(Equal Variances)'}")

    group1_normal = shapiro_g1_p >= alpha
    group2_normal = shapiro_g2_p >= alpha

    # Decide test type
    if group1_normal and group2_normal:
        if levene_p >= alpha:
            test_type = "Independent Samples t-test (Equal Variances)"
            t_stat, p_val = stats.ttest_ind(values_group1, values_group2, equal_var=True)
        else:
            test_type = "Welch's t-test (Unequal Variances)"
            t_stat, p_val = stats.ttest_ind(values_group1, values_group2, equal_var=False)

        # Using pooled SD for Cohen's d
        s1 = values_group1.std(ddof=1)
        s2 = values_group2.std(ddof=1)
        pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))

        print(f"    Pooled SD: {pooled_sd:.6f}")
        
        cohen_d = (values_group1.mean() - values_group2.mean()) / pooled_sd

        # Hedges' g correction
        J = 1 - (3 / (4*(n1+n2)-9))
        hedges_g = cohen_d * J

        print(f"\n  Hypothesis Test ({test_type}):")
        print(f"    P-value: {p_val:.4f}")
        print(f"    Cohen's d (pooled SD): {cohen_d:.4f}")
        print(f"    Hedges' g (small-sample corrected): {hedges_g:.4f}")

    else:
        test_type = "Mann-Whitney U Test"
        u_stat, p_val = stats.mannwhitneyu(values_group1, values_group2, alternative='two-sided')
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        print(f"\n  Hypothesis Test ({test_type}):")
        print(f"    P-value: {p_val:.4f}")
        print(f"    Rank-Biserial Correlation: {rank_biserial:.4f}")

    if p_val < alpha:
        print(f"    Conclusion: Statistically significant difference (p < {alpha}).")
    else:
        print(f"    Conclusion: No statistically significant difference (p >= {alpha}).")
    print("-" * 50)


metrics_to_analyze = ['Force CV', 'Cost of Transport', 'Average CSI', 'Time of Completion']

for metric in metrics_to_analyze:
    analyze_metric(metric, undulation_data, sidewinding_data)

print("=" * 50)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set Times New Roman 
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20  # default font size

metrics_to_plot = ['Cost of Transport', 'Force CV', 'Average CSI', 'Time of Completion']

means = []
stds = []
for metric in metrics_to_plot:
    means.append([
        undulation_data[metric].mean(),
        sidewinding_data[metric].mean()
    ])
    stds.append([
        undulation_data[metric].std(),
        sidewinding_data[metric].std()
    ])

means = np.array(means)
stds = np.array(stds)

x = np.arange(len(metrics_to_plot))
width = 0.35

fig, ax1 = plt.subplots(figsize=(5, 4))

# Colors
colors = ['#1f77b4', '#d62728']

# Left axis (for the first three performance metrics)
ax1.set_ylabel('Normalized Performance Metrics')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_to_plot)

# Creating a twin y-axis for Time of Completion
ax2 = ax1.twinx()
ax2.set_ylabel('Time of Completion (s)', color='black')

# Plot bars
for i, metric in enumerate(metrics_to_plot):
    if metric != 'Time of Completion':
        idx = i
        ax = ax1
    else:
        idx = i
        ax = ax2

    if metric == 'Time of Completion':
        # Creating the bars
        rects1 = ax.bar(x[idx] - width/2, means[idx,0], width, yerr=stds[idx,0],
                        color=colors[0], capsize=5, label='Lateral Undulation')
        rects2 = ax.bar(x[idx] + width/2, means[idx,1], width, yerr=stds[idx,1],
                        color=colors[1], capsize=5, label='Sidewinding')
    else:
        rects1 = ax.bar(x[idx] - width/2, means[idx,0], width, yerr=stds[idx,0],
                        color=colors[0], capsize=5)
        rects2 = ax.bar(x[idx] + width/2, means[idx,1], width, yerr=stds[idx,1],
                        color=colors[1], capsize=5)

# Legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=12)

# Value labels
def autolabel(ax, rects, errs):
    for rect, err in zip(rects, errs):
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height + err),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14)

# Apply only to left metrics (avoid overlapping right axis text)
for i, metric in enumerate(metrics_to_plot[:-1]):
    rects = [bar for bar in ax1.patches if np.isclose(bar.get_x() + bar.get_width()/2, x[i] - width/2) or
                                            np.isclose(bar.get_x() + bar.get_width()/2, x[i] + width/2)]
    autolabel(ax1, rects, [stds[i,0], stds[i,1]])

plt.tight_layout()
plt.show()
