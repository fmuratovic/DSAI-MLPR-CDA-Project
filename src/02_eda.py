"""
Exploratory data analysis of the night-level dataset, before modelling.
Six sleep quality outcomes, six predictors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("data/processed/night_level.csv", parse_dates=["night"])

predictors = ["steps", "kcal", "eff_lag1", "bedtime_lag1", "bedtime_sd7", "weekend"]
outcomes = {
    "eff": "sleep efficiency (%)",
    "tst": "total sleep time (min)",
    "waso": "WASO (min)",
    "sol": "sleep latency (min)",
    "rem_pct": "REM (% of TST)",
    "n3_pct": "deep sleep (% of TST)",
}

print("dataset:", df.shape)
print("period:", df.night.min().date(), "to", df.night.max().date())


# ---------- 1. Sample characteristics ----------

print("\nsummary of outcomes:")
print(df[list(outcomes)].describe().round(1).T)

print("\nsummary of predictors:")
print(df[predictors].describe().round(1).T)

print("\nbinary outcome balance:")
for c in ["eff_poor", "waso_poor"]:
    print(f"  {c}: {int(df[c].sum())} of {len(df)} "
          f"({100 * df[c].mean():.1f}%), majority baseline "
          f"{100 * max(df[c].mean(), 1 - df[c].mean()):.1f}%")


# ---------- 2. Comparison of good and poor nights ----------

# equivalent of a Table 1, repeated for both binary outcomes
for target, name in [("eff_poor", "poor efficiency"), ("waso_poor", "high WASO")]:
    rows = []
    for c in predictors:
        good = df.loc[df[target] == 0, c]
        poor = df.loc[df[target] == 1, c]
        t, p = stats.ttest_ind(good, poor, equal_var=False)
        rows.append([c, good.mean(), good.std(), poor.mean(), poor.std(), p])
    tab = pd.DataFrame(rows, columns=["variable", "good_mean", "good_sd",
                                      "poor_mean", "poor_sd", "p_value"])
    print(f"\npredictors by {name}:")
    print(tab.round(3).to_string(index=False))


# ---------- 3. Outcome distributions ----------

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, label) in zip(axes.ravel(), outcomes.items()):
    ax.hist(df[col], bins=20, edgecolor="black")
    ax.set_xlabel(label)
    ax.set_ylabel("nights")
    # NSF reference values where they exist
    if col == "eff":
        ax.axvline(85, color="red", linestyle="--", label="NSF 85%")
        ax.legend(fontsize=8)
    if col == "waso":
        ax.axvline(41, color="red", linestyle="--", label="NSF 41 min")
        ax.legend(fontsize=8)
    if col == "sol":
        ax.axvline(15, color="red", linestyle="--", label="NSF 15 min")
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/fig_outcome_distributions.png", dpi=150)
plt.close()


# ---------- 4. Time series of every outcome ----------

fig, axes = plt.subplots(7, 1, figsize=(13, 16), sharex=True)
axes[0].plot(df.night, df.steps, linewidth=1, color="tab:green")
axes[0].set_ylabel("steps")
for ax, (col, label) in zip(axes[1:], outcomes.items()):
    ax.plot(df.night, df[col], linewidth=1)
    # 14-night rolling mean to make any drift visible
    ax.plot(df.night, df[col].rolling(14, center=True).mean(),
            color="red", linewidth=2)
    ax.set_ylabel(label, fontsize=9)
axes[-1].set_xlabel("date")
plt.tight_layout()
plt.savefig("outputs/fig_timeseries.png", dpi=150)
plt.close()


# ---------- 5. Autocorrelation of every outcome ----------

lags = range(1, 11)
band = 1.96 / np.sqrt(len(df))

print("\nlag-1 autocorrelation:")
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (col, label) in zip(axes.ravel(), outcomes.items()):
    acf = [df[col].corr(df[col].shift(l)) for l in lags]
    print(f"  {label:<24} {acf[0]:+.3f}")
    ax.bar(lags, acf)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(band, color="red", linestyle="--")
    ax.axhline(-band, color="red", linestyle="--")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("lag (nights)")
    ax.set_ylabel("correlation")
plt.tight_layout()
plt.savefig("outputs/fig_autocorrelation.png", dpi=150)
plt.close()


# ---------- 6. Correlation matrix ----------

cols = predictors + list(outcomes)
corr = df[cols].corr()

print("\ncorrelation of predictors with outcomes:")
print(corr.loc[predictors, list(outcomes)].round(3))

plt.figure(figsize=(11, 9))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
plt.yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("outputs/fig_correlation.png", dpi=150)
plt.close()

# how many of the 36 predictor-outcome tests reach significance
pvals = []
for p in predictors:
    for o in outcomes:
        pvals.append(stats.pearsonr(df[p], df[o])[1])
pvals = np.array(pvals)
print(f"\n{len(pvals)} predictor-outcome correlations tested")
print(f"  p < 0.05: {(pvals < 0.05).sum()} (expected by chance: {0.05 * len(pvals):.1f})")
print(f"  smallest p-value: {pvals.min():.3f}")
print(f"  Bonferroni threshold: {0.05 / len(pvals):.4f}")


# ---------- 7. Steps against every outcome ----------

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, label) in zip(axes.ravel(), outcomes.items()):
    ax.scatter(df.steps, df[col], s=15, alpha=0.6)
    b, a = np.polyfit(df.steps, df[col], 1)
    xs = np.linspace(df.steps.min(), df.steps.max(), 50)
    ax.plot(xs, a + b * xs, color="red")
    r, p = stats.pearsonr(df.steps, df[col])
    ax.set_title(f"{label}\nr={r:.3f}, p={p:.2f}", fontsize=10)
    ax.set_xlabel("steps")
plt.tight_layout()
plt.savefig("outputs/fig_scatter_steps.png", dpi=150)
plt.close()


# ---------- 8. Dose-response across activity quartiles ----------

df["step_q"] = pd.qcut(df.steps, 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("\noutcome means by step quartile, with one-way ANOVA:")
rows = []
for col, label in outcomes.items():
    means = df.groupby("step_q", observed=True)[col].mean()
    f, p = stats.f_oneway(*[g[col].values for _, g in df.groupby("step_q", observed=True)])
    rows.append([label] + list(means.round(1)) + [round(p, 3)])
print(pd.DataFrame(rows, columns=["outcome", "Q1", "Q2", "Q3", "Q4", "ANOVA p"])
      .to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, label) in zip(axes.ravel(), outcomes.items()):
    ax.boxplot([g[col].values for _, g in df.groupby("step_q", observed=True)],
               tick_labels=["Q1", "Q2", "Q3", "Q4"])
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("step count quartile")
plt.tight_layout()
plt.savefig("outputs/fig_step_quartiles.png", dpi=150)
plt.close()


# ---------- 9. Drift over the observation period ----------

# the modelling showed later nights differ from earlier ones, so this
# checks whether the outcomes are stable across the six months
df["block"] = pd.cut(np.arange(len(df)), 3, labels=["first", "middle", "last"])
print("\noutcome means by third of the observation period:")
rows = []
for col, label in outcomes.items():
    means = df.groupby("block", observed=True)[col].mean()
    f, p = stats.f_oneway(*[g[col].values for _, g in df.groupby("block", observed=True)])
    rows.append([label] + list(means.round(1)) + [round(p, 4)])
print(pd.DataFrame(rows, columns=["outcome", "first", "middle", "last", "ANOVA p"])
      .to_string(index=False))

print("\nbinary outcome rates by third:")
print(df.groupby("block", observed=True)[["eff_poor", "waso_poor"]].mean().round(3))

print("\npredictor means by third:")
print(df.groupby("block", observed=True)[predictors].mean().round(1))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, label) in zip(axes.ravel(), outcomes.items()):
    ax.boxplot([g[col].values for _, g in df.groupby("block", observed=True)],
               tick_labels=["first", "middle", "last"])
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("third of observation period")
plt.tight_layout()
plt.savefig("outputs/fig_drift.png", dpi=150)
plt.close()

print("\nfigures saved to outputs/")
