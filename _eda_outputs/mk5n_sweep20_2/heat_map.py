import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("compendium_q2_with_metrics.csv")

# Columns to include in heatmap
cols = ["overall_accuracy", "RRM", "rank_agg", "pareto_tier"]

# Copy subset for heatmap
hm = df[cols].copy()

# Determine which metrics must be inverted (lower = better)
invert_cols = ["rank_agg", "pareto_tier"]

# Normalize every column so high = good and low = bad
for col in cols:
    col_min, col_max = hm[col].min(), hm[col].max()
    
    if col in invert_cols:
        # Invert: low is good → high normalized
        hm[col] = (col_max - hm[col]) / (col_max - col_min)
    else:
        # Regular normalization: high is good
        hm[col] = (hm[col] - col_min) / (col_max - col_min)

# Create heatmap
plt.figure(figsize=(10, 12))
plt.imshow(hm, cmap="viridis", aspect="auto")

plt.colorbar(label="Normalized Performance (0 = worst, 1 = best)")
plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
plt.yticks(range(len(df)), df["cfg"])

plt.title("Model Comparison Heat Map \n(Unified Performance Encoding)", fontsize=18)
plt.tight_layout()
plt.show()
