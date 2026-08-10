"""
Modelling. Six sleep quality outcomes predicted from the previous day's
activity and the previous night's sleep.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             r2_score, mean_absolute_error, mean_squared_error)

SEED = 42
df = pd.read_csv("data/processed/night_level.csv", parse_dates=["night"])
df = df.sort_values("night").reset_index(drop=True)

predictors = ["steps", "kcal", "eff_lag1", "bedtime_lag1", "bedtime_sd7", "weekend"]
sets = {
    "M0 prior sleep": ["eff_lag1"],
    "M1 activity": ["steps", "kcal"],
    "M2 both": predictors,
}

# six outcomes grouped by NSF category
continuous_outcomes = {
    "sleep efficiency (%)": "eff",
    "total sleep time (min)": "tst",
    "WASO (min)": "waso",
    "sleep latency (min)": "sol",
    "REM (% of TST)": "rem_pct",
    "deep sleep (% of TST)": "n3_pct",
}
binary_outcomes = {
    "poor efficiency (<85%)": "eff_poor",
    "high WASO (>=41 min)": "waso_poor",
}

tscv = TimeSeriesSplit(n_splits=5)
print("n =", len(df))


# ---------- 1. Regression on the six continuous outcomes ----------

def reg_models():
    return {
        "Linear regression": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
        "Decision tree": DecisionTreeRegressor(max_depth=3, random_state=SEED),
        "Random forest": RandomForestRegressor(n_estimators=300, max_depth=4,
                                               random_state=SEED),
        "Mean baseline": DummyRegressor(strategy="mean"),
    }


rows = []
for label, col in continuous_outcomes.items():
    y = df[col].values
    for set_name, cols in sets.items():
        X = df[cols].values
        for mname, model in reg_models().items():
            yt, yp = [], []
            for tr, te in tscv.split(X):
                model.fit(X[tr], y[tr])
                yt.extend(y[te])
                yp.extend(model.predict(X[te]))
            yt, yp = np.array(yt), np.array(yp)
            rows.append({
                "outcome": label, "predictors": set_name, "model": mname,
                "R2": r2_score(yt, yp),
                "RMSE": np.sqrt(mean_squared_error(yt, yp)),
                "MAE": mean_absolute_error(yt, yp),
            })

reg = pd.DataFrame(rows)
print("\n=== REGRESSION, forward-chaining cross-validation ===")
for label in continuous_outcomes:
    print(f"\n--- {label} ---")
    print(reg[reg.outcome == label].drop(columns=["outcome"])
          .round(3).to_string(index=False))
reg.to_csv("outputs/results_regression.csv", index=False)


# ---------- 2. Classification on the two binary outcomes ----------

def clf_models():
    return {
        "Logistic regression": Pipeline([("s", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=1000))]),
        "Naive Bayes": GaussianNB(),
        "Decision tree": DecisionTreeClassifier(max_depth=3, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=300, max_depth=4,
                                                random_state=SEED),
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
    }


def clf_metrics(yt, yp, ypr):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "AUC": roc_auc_score(yt, ypr) if len(set(yt)) > 1 else np.nan,
    }


rows, oof = [], {}
for label, col in binary_outcomes.items():
    y = df[col].astype(int).values
    print(f"\n{label}: {y.sum()} positive of {len(y)} ({100 * y.mean():.1f}%)")
    for set_name, cols in sets.items():
        X = df[cols].values
        for mname, model in clf_models().items():
            yt, yp, ypr = [], [], []
            for tr, te in tscv.split(X):
                model.fit(X[tr], y[tr])
                yt.extend(y[te])
                yp.extend(model.predict(X[te]))
                ypr.extend(model.predict_proba(X[te])[:, 1])
            yt, yp, ypr = np.array(yt), np.array(yp), np.array(ypr)
            m = clf_metrics(yt, yp, ypr)
            m.update({"outcome": label, "predictors": set_name, "model": mname})
            rows.append(m)
            oof[(label, set_name, mname)] = (yt, ypr)

clf = pd.DataFrame(rows)[["outcome", "predictors", "model", "accuracy",
                          "precision", "recall", "specificity", "AUC"]]
print("\n=== CLASSIFICATION, forward-chaining cross-validation ===")
for label in binary_outcomes:
    print(f"\n--- {label} ---")
    print(clf[clf.outcome == label].drop(columns=["outcome"])
          .round(3).to_string(index=False))
clf.to_csv("outputs/results_classification.csv", index=False)


# ---------- 3. Bootstrap confidence intervals for AUC ----------

print("\n=== AUC with 95% bootstrap CI, full predictor set ===")
rng = np.random.default_rng(SEED)
for label in binary_outcomes:
    for mname in clf_models():
        if mname == "Majority baseline":
            continue
        yt, ypr = oof[(label, "M2 both", mname)]
        aucs = []
        for _ in range(1000):
            i = rng.integers(0, len(yt), len(yt))
            if len(set(yt[i])) > 1:
                aucs.append(roc_auc_score(yt[i], ypr[i]))
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        print(f"  {label:<24} {mname:<20} {roc_auc_score(yt, ypr):.3f}  [{lo:.3f}, {hi:.3f}]")


# ---------- 4. ROC curves ----------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, label in zip(axes, binary_outcomes):
    for mname in clf_models():
        if mname == "Majority baseline":
            continue
        yt, ypr = oof[(label, "M2 both", mname)]
        fpr, tpr, _ = roc_curve(yt, ypr)
        ax.plot(fpr, tpr, label=f"{mname} ({roc_auc_score(yt, ypr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="chance (0.50)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(label)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/fig_roc.png", dpi=150)
plt.close()


# ---------- 5. R2 summary across outcomes ----------

best = (reg[reg.model != "Mean baseline"]
        .loc[lambda d: d.groupby("outcome").R2.idxmax()]
        .set_index("outcome"))
print("\n=== best R2 per outcome (negative means worse than predicting the mean) ===")
print(best[["predictors", "model", "R2", "RMSE"]].round(3).to_string())

plt.figure(figsize=(9, 5))
plt.barh(best.index, best.R2)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("best cross-validated $R^2$")
plt.title("Best model performance by outcome (0 = no better than the mean)")
plt.tight_layout()
plt.savefig("outputs/fig_r2_by_outcome.png", dpi=150)
plt.close()


# ---------- 5b. Effect of predictor count ----------

# M0 has 1 predictor, M1 has 2, M2 has 6, the baseline has none. If the
# predictors carried real information, more of them should help.
print("\n=== R2 by predictor set, linear regression vs mean baseline ===")
pivot = reg.pivot_table(index="outcome", columns=["model", "predictors"], values="R2")
print(pivot[["Linear regression", "Mean baseline"]].round(3).to_string())
pivot.to_csv("outputs/results_by_predictor_set.csv")

lin = reg[reg.model == "Linear regression"]
base = reg[reg.model == "Mean baseline"]
labels = list(continuous_outcomes)
x = np.arange(len(labels))
plt.figure(figsize=(11, 5.5))
for i, (sname, off) in enumerate(zip(sets, [-0.25, 0.0, 0.25])):
    vals = [lin[(lin.outcome == o) & (lin.predictors == sname)].R2.iloc[0] for o in labels]
    plt.bar(x + off, vals, width=0.25, label=f"{sname} ({len(sets[sname])} predictors)")
bvals = [base[(base.outcome == o) & (base.predictors == "M2 both")].R2.iloc[0] for o in labels]
plt.plot(x, bvals, "k_", markersize=28, markeredgewidth=2,
         label="mean baseline (0 predictors)")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(x, labels, rotation=20, ha="right", fontsize=8)
plt.ylabel("cross-validated $R^2$")
plt.title("More predictors give worse held-out performance")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/fig_predictor_count.png", dpi=150)
plt.close()


# ---------- 6. Tree depth sweep ----------

X = df[predictors].values
y = df["eff_poor"].astype(int).values
depths = range(1, 13)
tr_acc, te_acc = [], []
for d in depths:
    a, b = [], []
    for tr, te in tscv.split(X):
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X[tr], y[tr])
        a.append(accuracy_score(y[tr], m.predict(X[tr])))
        b.append(accuracy_score(y[te], m.predict(X[te])))
    tr_acc.append(np.mean(a))
    te_acc.append(np.mean(b))

plt.figure(figsize=(7, 5))
plt.plot(depths, tr_acc, "o-", label="training")
plt.plot(depths, te_acc, "s-", label="test")
plt.axhline(1 - y.mean(), color="red", linestyle="--", label="majority baseline")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("Decision tree complexity vs accuracy, poor efficiency")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/fig_depth_sweep.png", dpi=150)
plt.close()
print("\ndepth sweep, train:", np.round(tr_acc, 3).tolist())
print("depth sweep, test: ", np.round(te_acc, 3).tolist())


# ---------- 6b. Rolling window sweep ----------

# The outcomes drift over the six months, so a model trained on all earlier
# nights may be fitted to a period that no longer applies. This limits training
# to the most recent nights and measures how much that changes performance.
print("\n=== rolling window sweep ===")
windows = [30, 60, 90, None]          # None = expanding window, all earlier nights
rows = []
for label, col in continuous_outcomes.items():
    y = df[col].values
    X = df[predictors].values
    for w in windows:
        cv = TimeSeriesSplit(n_splits=5, max_train_size=w)
        for mname in ["Linear regression", "Mean baseline"]:
            model = reg_models()[mname]
            yt, yp = [], []
            for tr, te in cv.split(X):
                model.fit(X[tr], y[tr])
                yt.extend(y[te])
                yp.extend(model.predict(X[te]))
            rows.append({"outcome": label, "window": w if w else "all",
                         "model": mname, "R2": r2_score(yt, yp),
                         "RMSE": np.sqrt(mean_squared_error(yt, yp))})

win = pd.DataFrame(rows)
print("\nR2 by training window length:")
print(win.pivot_table(index="outcome", columns=["model", "window"], values="R2")
      .round(3).to_string())
win.to_csv("outputs/results_window.csv", index=False)

plt.figure(figsize=(9, 5.5))
for label in continuous_outcomes:
    sub = win[(win.outcome == label) & (win.model == "Linear regression")]
    plt.plot(range(len(windows)), sub.R2.values, "o-", label=label)
plt.xticks(range(len(windows)), ["30", "60", "90", "all"])
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("training window (nights)")
plt.ylabel("cross-validated $R^2$")
plt.title("Effect of limiting training to recent nights")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/fig_window_sweep.png", dpi=150)
plt.close()


# ---------- 7. Positive control ----------

# efficiency is time asleep divided by time in bed, so these two predictors
# determine the outcome. Recovering it confirms the pipeline works.
print("\n=== positive control: poor efficiency from time asleep and time in bed ===")
Xc = df[["tst", "tib"]].values
y = df["eff_poor"].astype(int).values
for mname, model in clf_models().items():
    if mname == "Majority baseline":
        continue
    yt, yp, ypr = [], [], []
    for tr, te in tscv.split(Xc):
        model.fit(Xc[tr], y[tr])
        yt.extend(y[te])
        yp.extend(model.predict(Xc[te]))
        ypr.extend(model.predict_proba(Xc[te])[:, 1])
    m = clf_metrics(np.array(yt), np.array(yp), np.array(ypr))
    print(f"  {mname:<20} accuracy {m['accuracy']:.3f}  AUC {m['AUC']:.3f}")


# ---------- 8. Linear coefficients on each outcome ----------

print("\n=== standardised linear regression coefficients (full data) ===")
coef_rows = []
for label, col in continuous_outcomes.items():
    pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
    pipe.fit(df[predictors], df[col])
    for p, c in zip(predictors, pipe.named_steps["m"].coef_):
        coef_rows.append({"outcome": label, "predictor": p, "coefficient": c})
coefs = pd.DataFrame(coef_rows).pivot(index="predictor", columns="outcome",
                                      values="coefficient")
print(coefs.round(2).to_string())
coefs.to_csv("outputs/coefficients.csv")

print("\nsaved results_regression.csv, results_classification.csv, coefficients.csv")
print("figures: fig_roc.png, fig_r2_by_outcome.png, fig_depth_sweep.png")
