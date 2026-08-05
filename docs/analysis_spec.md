# Analysis specification

Predicting next-night sleep quality from daily activity.
Written before modelling. Frozen once modelling begins; any later change is recorded in the
change log at the bottom, with the reason and the date.

Fixing these definitions in advance is what TRIPOD+AI asks for. It also protects the analysis:
if the outcome or the predictor list can be revised after seeing results, reported performance
is optimistic by an unknown amount.

---

## 1. Source data

Two files, one participant, exported for 2025-11-25 to 2026-05-04.

| File | Rows | Columns |
|---|---|---|
| `Activity-Didiconn-2025-11-01-2026-05-04.csv` | 160 | Date, Steps, Calories(kcal) |
| `Sleep-Didiconn-2025-11-01-2026-05-04.csv` | 168 | Start/End time, onset, wake, efficiency, minutes asleep, 4 stage columns |

Verified on load (all pass):

- reported efficiency equals minutes asleep / time in bed, to within 0.7 percentage points
- REM + Light + Deep equals minutes asleep exactly, on all 168 rows
- the Awake column equals (wake − onset) − minutes asleep, to within 1 minute,
  confirming it is wake-after-sleep-onset and not total time awake

## 2. Unit of analysis

One row per **night**. A night is labelled with the date of the day that precedes it, so
night `d` covers the daytime of day `d` and the sleep that follows it.

Episodes are assigned to nights on a **noon-to-noon** basis: an episode belongs to night `d`
if it starts between 12:00 on day `d` and 12:00 on day `d+1`. This keeps a 01:30 bedtime with
the evening it belongs to rather than splitting it onto the following calendar date.

Where a night contains more than one episode, the episode with the most minutes asleep is the
**main sleep** and supplies the outcome. Remaining episodes are treated as **naps** and their
minutes are summed into a predictor. 148 nights have one episode, 10 have two.

## 3. Outcome

**Primary — sleep efficiency (%) of the main sleep.**
Continuous, taken directly from the file. Modelled by linear regression.

**Secondary — poor night, defined as efficiency < 85%.**
Binary, modelled by logistic regression. The 85% threshold is the conventional cut-off used
in the insomnia literature; it was chosen before looking at the distribution, not from the
observed median. In this dataset it yields 35.8% poor nights, which is workable balance.

**Sensitivity — restorative percentage, (Deep + REM) / minutes asleep.**
Same predictors, same validation, reported separately. Included to show conclusions do not
depend on one definition of quality. Not primary, because consumer wrist devices stage sleep
far less reliably than they separate sleep from wake, and because slow-wave sleep is
front-loaded in the night while REM is back-loaded, so the ratio moves with sleep duration
for reasons unrelated to quality.

## 4. Predictors

All are measurable before the night begins. Nothing measured during the outcome window enters
the model.

| Predictor | Definition | Source |
|---|---|---|
| `steps` | step count, day `d` | activity file |
| `kcal` | calories, day `d` | activity file |
| `nap_min` | minutes asleep in non-main episodes, night `d` | sleep file |
| `eff_lag1` | efficiency of the previous night | sleep file, shifted |
| `bedtime_lag1` | previous night's bedtime, hours from midnight, negative before | sleep file, shifted |
| `bedtime_sd7` | SD of bedtime over the previous 7 nights | derived |
| `weekend` | 1 if night `d` is Friday or Saturday | calendar |

### Deliberately excluded

| Excluded | Reason |
|---|---|
| Wake-up time, tonight | closes the outcome window — the model would already know the answer |
| Falling-asleep time, tonight | sleep onset latency is itself a measure of sleep quality |
| Start time, tonight | only known once the person is in bed, too late to act on; `bedtime_lag1` carries almost the same information |
| Stage minutes, tonight | measured during the outcome window |
| `tst_lag1`, `waketime_lag1` | arithmetically redundant, see below |
| `steps_ma3` | 3-day rolling step mean, correlates 0.71 with same-day steps |

### The redundancy that forced two drops

Time in bed is wake time minus bedtime, and efficiency is minutes asleep divided by time in
bed. So `eff_lag1`, `tst_lag1`, `bedtime_lag1` and `waketime_lag1` are linked by arithmetic:
any three of them determine the fourth. Fitted together, the model can shift credit between
them freely and still produce identical predictions, so the estimated effects become unstable
and arbitrary. Measured variance inflation factors were 5.0, 88.3, 74.2 and 90.4.

Keeping two breaks the link. `eff_lag1` and `bedtime_lag1` were retained because they cover
different things — how well the previous night went, and when it happened.

Steps and calories were checked for the same problem and are acceptable: r = 0.647, variance
inflation factor about 1.9 with both present. Both are kept.

## 5. Sample size

151 nights survive to modelling (see attrition table in the cleaning output). At the 85%
threshold, 54 are poor nights.

The binding constraint is the rarer class, not the row count: the logistic model learns the
pattern mainly from the 54 poor nights. At the conventional 10 events per predictor, that
supports 5. Consecutive nights are also not fully independent, which reduces the effective
sample further.

**Both models are therefore capped at 6 predictors**, applied identically so the comparison
between them stays fair. The final list above contains exactly 6 (excluding `nap_min`, which
is reported as an optional sensitivity addition).

Predictors were chosen on physiological reasoning, before seeing performance. Selecting them
by testing which ones score best would inflate reported accuracy.

## 6. Models

Three nested models, fitted for each outcome:

| Model | Predictors | Question |
|---|---|---|
| M0 | `eff_lag1` only | how much is simple persistence? |
| M1 | `steps`, `kcal` | does daily activity predict anything on its own? |
| M2 | M1 + `eff_lag1`, `bedtime_lag1`, `bedtime_sd7`, `weekend` | does activity add anything beyond prior sleep? |

The comparison M1 versus M2 is the analysis that answers the project question. Change in R²
for the continuous outcome, change in area under the ROC curve for the binary one, with a
nested likelihood-ratio test.

**Note on M0.** M0 was specified as a persistence baseline on the assumption that sleep runs
in streaks. In this dataset it does not: the night-to-night correlation of efficiency is 0.04.
M0 is retained and reported anyway, because the absence of persistence is itself a finding
worth stating rather than hiding.

## 7. Validation

Nights are ordered in time and drawn from one participant, so a random shuffle would let the
model learn from nights that come after the ones it is tested on. Validation must respect
time order.

**Primary: blocked forward-chaining cross-validation.** Train on nights 1..k, test on the next
block, roll forward, average across folds. Reported with the number of folds and block size.

**Secondary: a single held-out final block**, the last 20% of nights, untouched until the
analysis is complete.

Bootstrap optimism correction is a reasonable alternative for a sample this size and may be
added, but it assumes exchangeable rows, which time-ordered data violates, so it is not
primary here.

## 8. Performance measures

Continuous outcome: R², root mean squared error, mean absolute error, all against a
mean-prediction baseline.

Binary outcome: area under the ROC curve with confidence interval, calibration slope and
intercept, calibration plot, Brier score.

Reporting a baseline alongside every figure is essential here. Given the correlations
observed, a model that appears adequate on RMSE alone may be doing nothing more than
predicting the mean of the outcome every night.

## 9. Explainability

If a non-linear model is added, SHAP values are reported per CLIX-M, with stability across
resamples explicitly assessed. The linear and logistic models are interpretable directly from
their coefficients and need no post-hoc method.

## 10. Known data quality issues

Carried into the limitations section of the report.

1. **Sleep onset latency is quantised.** All 168 values fall on an exact 2.5-minute grid from
   5.0 to 60.0 minutes; 149 land on the grid to the second, 167 within 2 seconds. Only 22
   distinct values occur, with a hard floor at 5 and a hard ceiling at 60. Stage columns are
   similarly rounded, mostly to multiples of 5.
2. **No night-to-night persistence.** Lag-1 correlation is 0.04 for efficiency and −0.008 for
   minutes asleep. Real sleep series almost always show some persistence.
3. Together, 1 and 2 suggest the file may be generated rather than device-exported. If nights
   were generated independently, no predictor can outperform the outcome mean, and a null
   result would reflect the data rather than the method. Query outstanding with course
   convenor.
4. **2025-11-25** records 1097 kcal against a basal floor of roughly 1310 seen on near-zero-step
   days, consistent with a partial first day of wear. That night is dropped anyway for lack of
   a lag-1 value.
5. **2026-02-06 and 2026-02-23** have no sleep episode at all.
6. Single participant. Findings do not generalise beyond this individual, and no external
   validation is possible.

---

## Change log

| Date | Change | Reason |
|---|---|---|
| (initial) | specification frozen | — |
| (initial) | cleaning restructured into CRISP-DM phases; added IQR outlier screen, missingness-mechanism classification, imputation-strategy comparison, skewness and log-transform assessment, and a standardised export | align the pipeline with the methods used throughout the course; no change to outcome or predictor definitions, and n remains 151 |
