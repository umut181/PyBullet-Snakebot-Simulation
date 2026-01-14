import pandas as pd
from scipy import stats
import numpy as np

DATA_FILE = 'snakebot_rubble_results.csv'
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

metrics_to_analyze = ['Cost of Transport', 'Force CV', 'Average CSI']

for metric_name in metrics_to_analyze:
    print(f"\n--- Analysis for: {metric_name} ---")

    values_group1 = undulation_data[metric_name].dropna()
    values_group2 = sidewinding_data[metric_name].dropna()

    n1 = len(values_group1)
    n2 = len(values_group2)

    if n1 == 0 or n2 == 0:
        print(f"  Skipping {metric_name}: Insufficient data for one or both groups.")
        continue

    print(f"  UNDULATION (n={n1}): Mean={values_group1.mean():.4f}, Median={values_group1.median():.4f}, Std Dev={values_group1.std():.4f}")
    print(f"  SIDEWINDING (n={n2}): Mean={values_group2.mean():.4f}, Median={values_group2.median():.4f}, Std Dev={values_group2.std():.4f}")

    # Normality check
    shapiro_g1_p = stats.shapiro(values_group1)[1] if n1 >= 3 else 1.0
    shapiro_g2_p = stats.shapiro(values_group2)[1] if n2 >= 3 else 1.0

    print(f"\n  Normality (Shapiro-Wilk):")
    print(f"    UNDULATION p-value={shapiro_g1_p:.4f} {'(NOT Normal)' if shapiro_g1_p < SIGNIFICANCE_LEVEL else '(Normal)'}")
    print(f"    SIDEWINDING p-value={shapiro_g2_p:.4f} {'(NOT Normal)' if shapiro_g2_p < SIGNIFICANCE_LEVEL else '(Normal)'}")

    # Variance homogeneity
    levene_stat, levene_p = stats.levene(values_group1, values_group2)
    print(f"\n  Variance Homogeneity (Levene's Test):")
    print(f"    Statistic={levene_stat:.4f}, p-value={levene_p:.4f} {'(UNEQUAL Variances)' if levene_p < SIGNIFICANCE_LEVEL else '(Equal Variances)'}")

    group1_normal = shapiro_g1_p >= SIGNIFICANCE_LEVEL
    group2_normal = shapiro_g2_p >= SIGNIFICANCE_LEVEL
    equal_variances = levene_p >= SIGNIFICANCE_LEVEL

    # Determine test type and compute effect size
    if group1_normal and group2_normal:
        if equal_variances:
            test_type = "Independent Samples t-test (Equal Variances)"
            t_stat, p_val = stats.ttest_ind(values_group1, values_group2, equal_var=True)
        else:
            test_type = "Welch's t-test (Unequal Variances)"
            t_stat, p_val = stats.ttest_ind(values_group1, values_group2, equal_var=False)

        # Using Cohen's d with pooled SD
        s1 = values_group1.std(ddof=1)
        s2 = values_group2.std(ddof=1)
        
        pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))

        print(f"    Pooled SD: {pooled_sd:.6f}")

        cohen_d = (values_group1.mean() - values_group2.mean()) / pooled_sd

        # Hedges' g correction
        J = 1 - (3 / (4*(n1+n2) - 9))
        hedges_g = cohen_d * J

        print(f"\n  Hypothesis Test ({test_type}):")
        print(f"    P-value: {p_val:.4f}")
        print(f"    Cohen's d: {cohen_d:.4f}")
        print(f"    Hedges' g (small-sample corrected): {hedges_g:.4f}")

    else:
        # Non-parametric Mann-Whitney U Test
        test_type = "Mann-Whitney U Test"
        u_stat, p_val = stats.mannwhitneyu(values_group1, values_group2, alternative='two-sided')
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        print(f"\n  Hypothesis Test ({test_type}):")
        print(f"    P-value: {p_val:.4f}")
        print(f"    Rank-Biserial Correlation: {rank_biserial:.4f}")

    if p_val < SIGNIFICANCE_LEVEL:
        print(f"    Conclusion: Statistically significant difference (p < {SIGNIFICANCE_LEVEL}).")
    else:
        print(f"    Conclusion: No statistically significant difference (p >= {SIGNIFICANCE_LEVEL}).")
    print("-" * 50)


# Chi-squared test for success rate
print("\n--- Chi-Squared Test for Success Rate: SIDEWINDING vs. UNDULATION ---\n")
contingency_table = pd.crosstab(df['gait'], df['Success/Fail'])
print("Observed Frequencies:\n", contingency_table)

chi2, p_value_chi2, dof_chi2, expected_chi2 = stats.chi2_contingency(contingency_table)

print(f"\nChi-Squared Statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi2:.4f}")
print(f"Degrees of Freedom: {dof_chi2}")

if p_value_chi2 < SIGNIFICANCE_LEVEL:
    print(f"Conclusion: Statistically significant difference in success rates (p < {SIGNIFICANCE_LEVEL}).")
    print(f"  UNDULATION Success Rate: {undulation_data['Success/Fail'].mean()*100:.2f}%")
    print(f"  SIDEWINDING Success Rate: {sidewinding_data['Success/Fail'].mean()*100:.2f}%")
else:
    print(f"Conclusion: No statistically significant difference in success rates (p >= {SIGNIFICANCE_LEVEL}).")
print("=" * 50)


# Distance traveled analysis (censored at 8m)
print("\n--- Analysis for: Distance Traveled (Distance Before Failure) ---")
print("Note: This data is censored at 8.00m (successful trials complete the full distance).")

distance_undulation = undulation_data['Distance Traveled'].dropna()
distance_sidewinding = sidewinding_data['Distance Traveled'].dropna()

print(f"\n  UNDULATION (n={len(distance_undulation)}): Median={distance_undulation.median():.4f}, Mean={distance_undulation.mean():.4f}, Std Dev={distance_undulation.std():.4f}")
print(f"  SIDEWINDING (n={len(distance_sidewinding)}): Median={distance_sidewinding.median():.4f}, Mean={distance_sidewinding.mean():.4f}, Std Dev={distance_sidewinding.std():.4f}")

u_stat_dist, p_val_dist = stats.mannwhitneyu(distance_undulation, distance_sidewinding, alternative='two-sided')
n1_dist = len(distance_undulation)
n2_dist = len(distance_sidewinding)
r_rb_dist = 1 - (2 * u_stat_dist) / (n1_dist * n2_dist)

print(f"\n  Mann-Whitney U Test (Distance Traveled):")
print(f"    P-value: {p_val_dist:.4f}")
print(f"    Effect Size (Rank-Biserial Corr): {r_rb_dist:.4f}")

if p_val_dist < SIGNIFICANCE_LEVEL:
    print(f"    Conclusion: Statistically significant difference (p < {SIGNIFICANCE_LEVEL}).")
    if distance_undulation.median() > distance_sidewinding.median():
        print(f"    The UNDULATION gait generally travels significantly farther (higher median distance).")
    else:
        print(f"    The SIDEWINDING gait generally travels significantly farther (higher median distance).")
else:
    print(f"    Conclusion: No statistically significant difference (p >= {SIGNIFICANCE_LEVEL}).")
print("=" * 50)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set font 
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# Metrics to plot (continuous)
metrics_to_plot = ['Cost of Transport', 'Force CV', 'Average CSI']

# Prepare means and stds
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

# Prepare Success Rate data (as percentages)
success_rates = np.array([
    undulation_data['Success/Fail'].mean() * 100,
    sidewinding_data['Success/Fail'].mean() * 100
])

# X positions for metrics and success rate
x = np.arange(len(metrics_to_plot))
width = 0.35

fig, ax1 = plt.subplots(figsize=(5, 4))

# Define colors
colors = ['#1f77b4', '#d62728']

# --- Primary Y-axis for the first three (continuous) metrics ---
rects1 = ax1.bar(x - width/2, means[:,0], width, yerr=stds[:,0],
                 label='Lateral Undulation', capsize=5, color=colors[0])
rects2 = ax1.bar(x + width/2, means[:,1], width, yerr=stds[:,1],
                 label='Sidewinding', capsize=5, color=colors[1])

ax1.set_ylabel('Metric Value')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_to_plot)
ax1.tick_params(axis='y')
ax1.legend(loc='upper right', fontsize=14)

# --- Secondary Y-axis for success rate ---
ax2 = ax1.twinx()

# Plot success rate bars slightly offset on a new scale
x_success = len(metrics_to_plot) + np.array([-width/2, width/2])
rects3 = ax2.bar(
    x_success, success_rates, width,
    label='Success Rate',
    color=['#1f77b4', '#d62728'],
    linewidth=1.0,
    zorder=5  # ensures they're drawn on top
)


ax2.patch.set_alpha(0)

ax2.set_ylabel('Success Rate (%)')
ax2.set_ylim(0, 110) 
ax2.tick_params(axis='y', labelcolor='black')

# --- Expand X-axis ---
all_xticks = np.concatenate((x, [len(metrics_to_plot)]))
all_xticklabels = metrics_to_plot + ['Success Rate']
ax1.set_xticks(all_xticks)
ax1.set_xticklabels(all_xticklabels)

# --- Value labels ---
def autolabel(rects, errs=None):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        y_pos = height + (errs[i] if errs is not None else 0)
        ax1.annotate(f'{height:.2f}',
                     xy=(rect.get_x() + rect.get_width()/2, y_pos),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=16)

def autolabel_simple(rects, ax=ax2):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=16, color='black')

autolabel(rects1, stds[:,0])
autolabel(rects2, stds[:,1])
autolabel_simple(rects3)

# --- Aesthetics ---
fig.tight_layout()
plt.show()



