import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Load the compendium file
# ----------------------------------------------------------
df = pd.read_csv("compendium_q2.csv")

# ----------------------------------------------------------
# 1. Compute micro-F1 as mean of all per-genre F1 columns
# ----------------------------------------------------------
genre_f1_cols = [
    "blues_f1", "classical_f1", "country_f1", "disco_f1", "hiphop_f1",
    "jazz_f1", "metal_f1", "pop_f1", "reggae_f1", "rock_f1"
]

df["micro_f1"] = df[genre_f1_cols].mean(axis=1)

# ----------------------------------------------------------
# 2. Rank Aggregation Score
# (lower = better)
# ----------------------------------------------------------
def rank_desc(series):
    """Rank with highest value = 1"""
    return series.rank(ascending=False, method="min")

df["rank_acc"]      = rank_desc(df["overall_accuracy"])
df["rank_rrm"]      = rank_desc(df["RRM"])
df["rank_macro"]    = rank_desc(df["macro_f1"])
df["rank_micro"]    = rank_desc(df["micro_f1"])

df["rank_agg"] = df["rank_acc"] + df["rank_rrm"] + df["rank_macro"] + df["rank_micro"]

# ----------------------------------------------------------
# 3. Pareto Tier Assignment
# ----------------------------------------------------------
def dominates(a, b):
    """Return True if row a dominates row b."""
    return np.all(a >= b) and np.any(a > b)

metrics = ["overall_accuracy", "RRM", "macro_f1", "micro_f1"]
values  = df[metrics].values
N       = len(df)

assigned = np.zeros(N, dtype=int)
tier     = 1

remaining = set(range(N))

while remaining:
    front = []
    for i in remaining:
        dominated_flag = False
        for j in remaining:
            if i != j and dominates(values[j], values[i]):
                dominated_flag = True
                break
        if not dominated_flag:
            front.append(i)

    for idx in front:
        assigned[idx] = tier
    remaining -= set(front)
    tier += 1

df["pareto_tier"] = assigned

# ----------------------------------------------------------
# Save the new full-featured CSV
# ----------------------------------------------------------
df.to_csv("compendium_q2_with_metrics.csv", index=False)

print("✔ DONE — File saved as compendium_q2_with_metrics.csv")
