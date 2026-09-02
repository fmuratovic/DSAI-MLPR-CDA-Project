# Daily Activity and Next-Night Sleep Quality

**Does daily step count and energy expenditure predict next-night sleep quality?**
An n-of-1 wearable study across six consensus-based sleep indicators.

Faris Muratović — Computational Data Analytics, Faculty of Electrical Engineering, University of Sarajevo
Supervised by Prof. Aida Branković

---

## Summary

This project tests whether a single adult's daily step count and energy expenditure predict
next-night sleep quality, using 151 consecutive nights of consumer wearable data. Six sleep
outcomes were selected from the sleep-quality literature — sleep efficiency, total sleep time,
wake after sleep onset (WASO), sleep onset latency, and REM and deep sleep as a share of total
sleep time — and modelled using logistic/linear regression, decision trees, random forests, and
naive Bayes, validated with time-aware (forward-chaining) cross-validation.

**Finding:** no model outperformed a simple baseline on any outcome. None of 36
predictor–outcome correlations were statistically significant. A positive control confirmed the
pipeline correctly detects a relationship when one exists (AUC 0.87), showing the null result
reflects an absence of signal in the data rather than a broken analysis. Separately, two sleep
continuity measures (WASO and sleep latency) changed significantly over the six-month period in
opposite directions, while sleep efficiency — the most commonly reported single metric — showed
no change at all, demonstrating that a single summary measure can conceal real underlying change.

Full detail, literature grounding, and discussion are in the report (see `report/`).

---

## Repository structure

```
.
├── data/
│   ├── raw/                    Original exported CSVs (activity + sleep)
│   └── processed/               Cleaned, joined, feature-engineered dataset
├── src/
│   ├── 01_clean.py               Data cleaning, joining, feature engineering
│   ├── 02_eda.py                 Exploratory analysis (distributions, correlations, drift)
│   └── 03_model.py               Modelling, cross-validation, positive control
├── outputs/                     Generated figures and result tables (from src/ scripts)
├── report/
│   ├── build_report_2col.js      Script that generates the final report
│   ├── build_poster.js           Script that generates the poster
│   ├── sleep_prediction_report_2col.docx/.pdf   Final report
│   └── poster.pptx/.pdf          Final poster
└── README.md
```

---

## Pipeline

Run in order from the repository root:

```bash
python src/01_clean.py     # raw CSVs -> data/processed/night_level.csv
python src/02_eda.py       # exploratory analysis -> outputs/fig_*.png
python src/03_model.py     # modelling and validation -> outputs/results_*.csv, fig_*.png
```

Each script reads from `data/` and writes its outputs to `outputs/` (and `data/processed/` for
the first script). Scripts must be run in order, as each depends on the previous step's output.

**Dependencies:** Python 3, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

---

## Methods overview

- **Unit of analysis:** one night, defined as the daytime of day *d* plus the sleep that follows
  it (noon-to-noon boundary).
- **Outcomes (6):** sleep efficiency, total sleep time, WASO, sleep onset latency, REM % and
  deep % of total sleep time — selected from the National Sleep Foundation consensus panel and a
  systematic review of sleep-quality measurement (see report references).
- **Predictors (6):** daily step count, daily energy expenditure, previous-night efficiency,
  previous-night bedtime, 7-night bedtime variability, and a weekend indicator. Nothing measured
  during the outcome night itself is used as a predictor.
- **Models:** linear/logistic regression (primary), decision tree, random forest, naive Bayes —
  each compared against an uninformative baseline under identical validation.
- **Validation:** blocked forward-chaining cross-validation (5 folds), with a rolling-window
  sensitivity analysis (30/60/90-night training windows).
- **Positive control:** the identical pipeline applied to two predictors that determine sleep
  efficiency by mathematical construction (time asleep, time in bed), to confirm the pipeline
  detects association when it exists.
- **Reporting standard:** TRIPOD+AI (prognostic prediction models).

Full methodological detail, sample-size justification, and literature grounding are in the report.

---

## Data

Two raw files, exported from a consumer sleep-tracking application:

| File | Rows | Contents |
|---|---|---|
| `Activity-Didiconn-*.csv` | 160 | Date, step count, estimated energy expenditure |
| `Sleep-Didiconn-*.csv` | 168 | One row per sleep episode: bedtime, wake time, sleep onset,
efficiency, total sleep time, stage minutes (awake/REM/light/deep) |

The dataset was provided by the course instructor for this coursework assignment. The device
type and form factor were not specified in the exported data. After cleaning and requiring a
full predictor history, 151 of 160 nights remain in the analysis dataset
(`data/processed/night_level.csv`).

---

## Outputs

- `outputs/fig_*.png` — all figures referenced in the report (distributions, correlation matrix,
  autocorrelation, model performance by predictor set, drift across the observation period, ROC
  curves, etc.)
- `outputs/results_*.csv` — full numeric results for regression and classification models,
  across all predictor sets and model families
- `outputs/coefficients.csv` — fitted logistic/linear regression coefficients per outcome

---

## Report and poster

- **Report:** `report/sleep_prediction_report_2col.pdf` — full write-up in Scientific
  Reports–style format (Introduction, Results, Discussion, Methods, Conclusion, Declarations),
  reported per TRIPOD+AI.
- **Poster:** `report/poster.pptx` — A3 landscape summary poster.

---

## Limitations

This is a single-participant, observational study; findings are indicative for this individual
only and are not generalisable. Full limitations, including measurement constraints of consumer
wearables and the statistical power of an n-of-1 design, are discussed in the report.
