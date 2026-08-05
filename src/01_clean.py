"""
Data cleaning and preparation.
Builds the night-level dataset from the raw activity and sleep files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

activity = pd.read_csv("data/raw/Activity-Didiconn-2025-11-01-2026-05-04.csv")
sleep = pd.read_csv("data/raw/Sleep-Didiconn-2025-11-01-2026-05-04.csv")


# ---------- 1. Initial inspection ----------

print("activity:", activity.shape)
print("sleep:", sleep.shape)
print(activity.head())
print(activity.dtypes)
print(sleep.dtypes)

print("\nmissing values per column:")
print(activity.isna().sum())
print(sleep.isna().sum())

print("\nduplicate rows:", activity.duplicated().sum(), sleep.duplicated().sum())
print("duplicate dates:", activity["Date"].duplicated().sum())
print("duplicate sleep episodes:",
      sleep.duplicated(subset=["Start Time", "End Time"]).sum())


# ---------- 2. Fix data types ----------

# Sleep Time Ratio is stored as text because of the % sign
activity["Date"] = pd.to_datetime(activity["Date"])
activity = activity.rename(columns={"Steps": "steps", "Calories(kcal)": "kcal"})

for c in ["Start Time", "End Time", "Falling Asleep Time", "Wake-up time"]:
    sleep[c] = pd.to_datetime(sleep[c])

sleep["eff"] = sleep["Sleep Time Ratio(%)"].str.rstrip("%").astype(float)
sleep["tst"] = sleep["Time Asleep(min)"]
sleep["deep"] = sleep["Sleep Stages - Deep Sleep(min)"]
sleep["rem"] = sleep["Sleep Stages - REM(min)"]
sleep["light"] = sleep["Sleep Stages - Light Sleep(min)"]
sleep["waso"] = sleep["Sleep Stages - Awake(min)"]

sleep["tib"] = (sleep["End Time"] - sleep["Start Time"]).dt.total_seconds() / 60
sleep["spt"] = (sleep["Wake-up time"] - sleep["Falling Asleep Time"]).dt.total_seconds() / 60
sleep["sol"] = (sleep["Falling Asleep Time"] - sleep["Start Time"]).dt.total_seconds() / 60


# ---------- 3. Descriptive statistics ----------

numeric = pd.DataFrame({
    "steps": activity.steps, "kcal": activity.kcal,
    "eff": sleep.eff, "tst": sleep.tst, "tib": sleep.tib,
    "sol": sleep.sol, "deep": sleep.deep, "rem": sleep.rem,
})

print("\ndescriptive statistics:")
print(numeric.describe().round(1))
print("\nskewness:")
print(numeric.skew().round(2))

numeric.hist(bins=25, figsize=(14, 8))
plt.tight_layout()
plt.savefig("outputs/fig_histograms.png", dpi=150)
plt.close()


# ---------- 4. Consistency checks ----------

print("\nefficiency vs computed tst/tib, max difference:",
      round((sleep.eff - 100 * sleep.tst / sleep.tib).abs().max(), 2))
print("stages sum to tst:", ((sleep.rem + sleep.light + sleep.deep) == sleep.tst).all())
print("awake column equals wake after sleep onset, max difference:",
      round((sleep.waso - (sleep.spt - sleep.tst)).abs().max(), 1))
print("sleep onset latency, distinct values:", sleep.sol.round(1).nunique(),
      "range:", sleep.sol.min(), "-", sleep.sol.max())


# ---------- 5. Outliers, IQR method ----------

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), numeric.columns):
    x = numeric[col].dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((x < lower) | (x > upper)).sum()
    print(f"{col}: Q1={q1:.1f} Q3={q3:.1f} bounds=({lower:.1f}, {upper:.1f}) outliers={n_out}")
    ax.boxplot(x, widths=0.5)
    ax.set_title(f"{col} ({n_out})")
    ax.set_xticks([])
plt.tight_layout()
plt.savefig("outputs/fig_boxplots.png", dpi=150)
plt.close()

# flagged values are real short nights, naps and inactive days, so nothing is removed
impossible = sleep[(sleep.tib < 30) | (sleep.tib > 960) | (sleep.eff < 20) | (sleep.eff > 100)]
print("impossible episodes:", len(impossible))


# ---------- 6. Assign episodes to nights ----------

# night d = daytime of day d plus the sleep that follows, using a noon-to-noon day
sleep["night"] = (sleep["Start Time"] - pd.Timedelta(hours=12)).dt.normalize()
print("\nepisodes per night:")
print(sleep.night.value_counts().value_counts().sort_index())

# longest episode of the night is the main sleep, the rest are naps
sleep = sleep.sort_values(["night", "tst"], ascending=[True, False])
sleep["is_main"] = sleep.groupby("night").cumcount() == 0
main = sleep[sleep.is_main]
naps = sleep[~sleep.is_main]
nap_min = naps.groupby("night").tst.sum().rename("nap_min")
print("main sleeps:", len(main), " naps:", len(naps))


# ---------- 7. Join ----------

# left join from a full calendar so nights without a sleep record stay visible
calendar = pd.DataFrame({"night": pd.date_range(main.night.min(), main.night.max())})

df = (calendar
      .merge(main[["night", "eff", "tst", "tib", "deep", "rem",
                   "Start Time", "Wake-up time"]], on="night", how="left")
      .merge(nap_min, on="night", how="left")
      .merge(activity.rename(columns={"Date": "night"}), on="night", how="left"))

df["nap_min"] = df.nap_min.fillna(0)
df = df.sort_values("night").reset_index(drop=True)

print("\nafter join:", df.shape)
print("nights without sleep record:", df.eff.isna().sum())
print(df[df.eff.isna()].night.dt.date.tolist())


# ---------- 8. Missing values ----------

# missing because the tracker was not worn, treated as MCAR
print("\nimputation strategies compared:")
print(pd.DataFrame({
    "drop": df.eff.dropna(),
    "mean": df.eff.fillna(df.eff.mean()),
    "median": df.eff.fillna(df.eff.median()),
    "ffill": df.eff.ffill(),
}).describe().round(2).loc[["count", "mean", "std", "min", "max"]])

# the missing values are the outcome itself, so the rows are dropped instead of imputed
print("lag-1 correlation of efficiency:", round(df.eff.corr(df.eff.shift(1)), 2))


# ---------- 9. Features and targets ----------

# bedtime on a continuous scale through midnight: 23:00 -> -1.0, 01:30 -> +1.5
bed_h = df["Start Time"].dt.hour + df["Start Time"].dt.minute / 60
df["bedtime"] = np.where(bed_h > 12, bed_h - 24, bed_h)
df["waketime"] = df["Wake-up time"].dt.hour + df["Wake-up time"].dt.minute / 60

# lags only valid when the previous row is really the night before
prev_ok = df.night.diff().dt.days.eq(1)
df["eff_lag1"] = df.eff.shift(1).where(prev_ok)
df["bedtime_lag1"] = df.bedtime.shift(1).where(prev_ok)
df["bedtime_sd7"] = df.bedtime.rolling(7, min_periods=5).std().shift(1)
df["weekend"] = df.night.dt.dayofweek.isin([4, 5]).astype(int)

df["eff_poor"] = np.where(df.eff.isna(), np.nan, (df.eff < 85).astype(float))
df["restorative_pct"] = 100 * (df.deep + df.rem) / df.tst

predictors = ["steps", "kcal", "eff_lag1", "bedtime_lag1", "bedtime_sd7", "weekend"]

print("\ncorrelation between predictors:")
print(df[predictors].corr().round(2))

# VIF
X = df[predictors].dropna()
Z = np.column_stack([np.ones(len(X)), ((X - X.mean()) / X.std()).values])
print("\nVIF:")
for i, name in enumerate(predictors, start=1):
    y, A = Z[:, i], np.delete(Z, i, axis=1)
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  {name}: {1 / (1 - r2):.2f}")


# ---------- 10. Final dataset ----------

required = ["eff"] + predictors
print("\nattrition:")
print("  all nights:", len(df))
running = df
for c in required:
    running = running.dropna(subset=[c])
    print(f"  after {c}: {len(running)}")

final = df.dropna(subset=required).copy()
print("\nfinal n:", len(final))
print("poor nights:", int(final.eff_poor.sum()),
      f"({100 * final.eff_poor.mean():.1f}%)")

keep = (["night", "eff", "eff_poor", "restorative_pct", "tst", "tib", "deep", "rem",
         "bedtime", "waketime", "nap_min"] + predictors)
final[keep].to_csv("data/processed/night_level.csv", index=False)

scaled = final[keep].copy()
scaled[predictors] = StandardScaler().fit_transform(final[predictors])
scaled.to_csv("data/processed/night_level_scaled.csv", index=False)

print("saved night_level.csv and night_level_scaled.csv")
