# Delhivery Last-Mile Delay Intelligence Pipeline

This repository turns a messy export of 1,200 historical parcels from a 24-hub regional network into one connected delay-intelligence pipeline.
It answers three operational questions: how long a parcel will really take, which parcels will miss their promise early enough to act, and which hubs need which playbook.
Across the cleaned history 0.2317 of parcels missed their promise, so the pipeline exists to find those parcels before they are late rather than to count them afterwards.

## How to run

_Added in Part D._

## Data findings

### (i) Express is the service type that misses its promise most often

Express parcels are delayed at a rate of **0.4897** — 119 delayed out of 243 shipments — the highest of the three service types, against 0.2056 for Standard and 0.0857 for Economy. Source: `outputs/part_a_delay_by_group.csv`.

Nearly half of all Express bookings break their promise, so Express is where a proactive call or a hub escalation buys the most goodwill per call made. The mean Express delivery time of **36.8056** hours already sits above the Express promise, which means the average Express parcel is a near-miss rather than a comfortable success — the promise itself, not just execution, is part of the problem.

### (ii) The delay rate climbs steeply from Tier 1 to Tier 3

The delay rate rises from **0.04** at Tier 1 hubs to **0.215** at Tier 2 and **0.44** at Tier 3, on an identical 400 shipments per tier. Mean delivery hours move the same way: **48.5664**, **58.8194** and **63.4943**. Source: `outputs/part_a_delay_by_group.csv`.

Tier 3 hubs carry most of the network's delay risk on the same parcel volume, so extra staffing, buffer time and alert-handling capacity should be allocated to Tier 3 first and Tier 1 last. Because every tier ships the same 400 parcels here, the gap is about how the hubs operate rather than how much they handle.

### (iii) The delay rate rises sharply with distance band

Delay rate by distance band: **0.0219** for 0-200 km (6 of 274), **0.1368** for 201-400 km (77 of 563), **0.4364** for 401-600 km (120 of 275) and **0.8523** for 601+ km (75 of 88). Mean hours follow: **46.7559**, **55.966**, **64.266** and **72.2608**. Source: `outputs/part_a_delay_by_group.csv`.

Short lanes are close to reliable and need no routine intervention, while beyond 400 km the promise becomes the exception — in the 601+ band a parcel fails more often than it succeeds, so those lanes need either a longer promise or a different lane design, not more expediting.

### (iv) The three columns most strongly correlated with `delivery_hours`

Ranked by absolute correlation with `delivery_hours`, excluding `delivery_hours` itself, the three strongest columns are `promised_hours` at **+0.8372**, `distance_km` at **+0.4013** and `n_stops` at **+0.3247**. Source: `outputs/part_a_correlations.csv`.

`promised_hours` at +0.8372 means most of the variation in delivery time is simply which service class the customer bought, so the biggest lever is which services are offered on which lanes rather than day-to-day expediting. `distance_km` at +0.4013 makes lane length the largest physical driver after service mix, so re-cutting long lanes or fulfilling from a closer hub cuts hours directly. `n_stops` at +0.3247 shows rider route density adds real hours, so capping stops per route is a concrete action a hub manager can take this week.

### (v) Why `declared_value_inr` was log-transformed

On the median-filled column the mean is **INR 1,533.05** against a median of **INR 1,121.50**, a clear right skew; after the natural log the mean is **7.0277** against a median of **7.0224**, which is close to symmetric. Source: `outputs/part_a_checks.csv`.

A mean sitting far above the median means a handful of very high-value parcels were pulling the column, and the log puts an ordinary parcel and a rare expensive one on the same footing so the model does not read declared value mainly from those outliers.

## Model results (regression)

The table was split 840 / 180 / 180 rows for training, validation and test (`outputs/part_b_checks.csv`). The scaler used for the leakage check was fitted on the 840 training rows only: the largest absolute column mean of the scaled training rows is **0.0**, while the same figure for the scaled test rows is **0.2187**. A test-row mean that is clearly off zero is the proof that no test statistic was used to fit the scaler, which is what makes the test score below an honest estimate rather than a rehearsal.

All five candidates below were fitted on the training rows and scored on the same 180-row validation split (`outputs/part_b_metrics.csv`).

| model | MAE | MSE | RMSE | R² |
| --- | --- | --- | --- | --- |
| mean_baseline | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| linear_regression | 3.9222 | 23.9235 | 4.8912 | 0.9313 |
| decision_tree | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| random_forest | 4.2882 | 30.355 | 5.5095 | 0.9128 |
| gradient_boosting | 3.9166 | 24.1722 | 4.9165 | 0.9306 |

### The baseline explains nothing

Predicting the training mean for every parcel gives an R² of **-0.0212** and an RMSE of **18.8552** hours (`outputs/part_b_metrics.csv`). R² is the share of variation in delivery hours the model explains, so a slightly negative value means this rule explains none of it and does marginally worse than the validation rows' own average would. A planner working from "the average parcel takes about the average time" is off by **15.7952** hours on a typical parcel (MAE), which is far too coarse to commit to a 36-hour or 60-hour promise, and it sets the bar every model below has to clear to be worth running at all.

### Linear regression won on validation

`linear_regression` has the lowest validation RMSE at **4.8912** hours, just ahead of `gradient_boosting` at **4.9165**, then `random_forest` at **5.5095** and `decision_tree` at **6.2796** (`outputs/part_b_metrics.csv`). Its R² of **0.9313** means it explains most of the variation in delivery hours. In operational terms a typical estimate lands within about five hours of the real outcome, which is tight enough to plan a dispatch slot against a 36-hour Express promise, though not tight enough to treat a parcel sitting within five hours of its promise as safe.

### Shallow depths are high-bias, deep ones are high-variance

From `outputs/part_b_tree_depth.csv`, with the gap being `val_rmse` minus `train_rmse`:

| max_depth | train_rmse | val_rmse | gap |
| --- | --- | --- | --- |
| 1 | 11.6334 | 12.1324 | 0.4990 |
| 2 | 9.3687 | 10.4114 | 1.0427 |
| 3 | 7.5295 | 8.7033 | 1.1738 |
| 4 | 6.0999 | 7.3896 | 1.2897 |
| 6 | 4.7114 | 6.2796 | 1.5682 |
| 8 | 3.336 | 6.7901 | 3.4541 |
| 12 | 1.3613 | 7.354 | 5.9927 |
| unlimited | 0.0 | 7.8368 | 7.8368 |

Depths 1 to 3 are high-bias: the gap is small (0.4990 at depth 1) but both errors are high, so the tree is not failing to generalise, it is simply too simple to describe the data — a depth-1 tree splits on one threshold and is wrong by over 11 hours even on rows it trained on. Depths 8, 12 and unlimited are high-variance: `train_rmse` keeps falling to **3.336**, **1.3613** and **0.0** while `val_rmse` rises to **6.7901**, **7.354** and **7.8368**, widening the gap to **7.8368**. The unlimited tree reproduces its training rows exactly and is still the worst of the eight on unseen parcels, which is the clearest warning against trusting a training-set score. Depth **6** is the turning point and the depth used for the `decision_tree` candidate, with the lowest validation RMSE of **6.2796**.

### Learning rate 0.1 was best

With 100 trees held fixed (`outputs/part_b_gb_learning_rate.csv`), 0.1 wins at validation RMSE **4.9165** and R² **0.9306**, against **10.0018** / **0.7127** at 0.01 and **7.3946** / **0.8429** at 1.0. At 0.01 each tree's correction is shrunk to a hundredth of its size, so 100 trees are not enough to walk the ensemble down to the signal — it is still underfit when the tree budget runs out, which is why its R² is the worst of the three and barely better than a shallow tree. At 1.0 each tree applies its full correction, so the ensemble overshoots the target at every step and chases the last tree's residuals instead of settling into the minimum, giving a worse validation score than 0.1 despite having exactly the same capacity. The useful operating lesson is that the tree count and the learning rate have to be chosen together: a smaller rate is only safe if the number of trees rises to match.

### Cross-validation disagrees with validation, and the gap is small

The two rankings disagree on first place. By validation RMSE: `linear_regression` (**4.8912**) then `gradient_boosting` (**4.9165**). By 5-fold cross-validated mean R² on the 1,020 training-plus-validation rows: `gradient_boosting` (**0.9353**, std **0.0045**) then `linear_regression` (**0.9261**, std **0.0048**) (`outputs/part_b_cv.csv`).

Under the brief's rule the final model is the candidate with the lowest validation RMSE, so `linear_regression` stays the final model and is the model scored on the test rows.

The model I would ship is `gradient_boosting`. Cross-validation averages five folds across 1,020 rows, so every row serves as a held-out row once, whereas the validation ranking rests on a single 180-row split; and the validation gap that put `linear_regression` first is only 0.0253 hours — about a minute and a half, well inside what one unlucky split can produce. Both cross-validation standard deviations are small (**0.0045** and **0.0048**), so the fold-to-fold ordering is stable rather than noise. For a decision this close I would rather trust the estimate built from more held-out data, and I would re-run both on a second split before changing anything in production.

### The final model on the test rows

Scored once on the 180 untouched test rows, `linear_regression` gives MAE **3.5671**, MSE **20.4624**, RMSE **4.5235** and R² **0.9493** (`outputs/part_b_metrics.csv`). The test score is slightly better than its validation score, which says the model is not over-tuned to the validation split. For planners this means a dispatch-time estimate of delivery hours that is typically within about three and a half hours of the real outcome, so it is dependable for scheduling and capacity work but should not be the only signal used to decide whether a single parcel will break its promise — that is the classifier's job in Part C.

## Model results (classification)

The shipped configuration (`outputs/part_c_final_config.csv`) keeps 7 of the 17 columns with `SelectKBest` and feeds them to a random forest of 100 trees with no depth limit and `min_samples_leaf` 1, alerting when the delay probability reaches 0.5. It came from the `default_forest_same_7_features` row of the tuning table.

The four models below were fitted on the 840 training rows and scored on the 180 validation rows at the default 0.5 decision rule (`outputs/part_c_models.csv`).

| model | TN | FP | FN | TP | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind_all_on_time | 138 | 0 | 42 | 0 | 0.7667 | 0.0 | 0.0 | 0.0 |
| logistic_regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| decision_tree_depth_6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.747 |
| random_forest_default | 132 | 6 | 11 | 31 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

### The blind baseline scores 0.7667 and is still useless

Predicting "on time" for every parcel returns an accuracy of **0.7667**, but its recall is **0.0**: it catches no delayed parcel at all. The confusion cells say it plainly — TP **0** and FN **42**, so every one of the 42 delayed parcels in the validation split walks straight past it. The accuracy only looks respectable because **138** of the 180 validation parcels really were on time, so a rule that never raises an alarm is right about those and wrong about every parcel operations actually cares about. As an alerting system it has no value whatsoever: no proactive call is ever placed and no hub manager is ever warned, which is why accuracy alone must never be the headline metric on a problem where under a quarter of parcels are positive.

### The noise columns take 0.2245 of the forest's importance yet rank 8 or worse

The five bookkeeping columns carry a combined **0.2245** of the forest's impurity importance — roughly a fifth of it — while every one of them falls outside the top 7 by F-score: `label_print_batch` at rank **8**, `tape_roll_no` at **12**, `sticker_serial_mod` at **14**, `shelf_slot_no` at **16** and `scanner_build_no` at **17** (`outputs/part_c_feature_scores.csv`). Their F-scores are **4.5847**, **1.9052**, **1.2776**, **0.1393** and **0.0230**, all beneath 10, against **36.7281** for the weakest of the top seven — a gap of more than three times.

They still receive importance because impurity importance gives credit to columns with many distinct values even when they carry no signal. A label batch number spanning 1000 to 9999, a tape roll number up to 500 and a continuous sticker serial offer the tree hundreds of candidate split points, and with enough of them one will always happen to separate a few delayed parcels in the training rows by chance. The F-test is the better witness here because it asks a single question — do delayed and on-time parcels differ on this column — and answers no. The operational reading is that a feature-importance chart should never on its own justify collecting a field: these five are warehouse bookkeeping, and dropping them is what the selector does.

### Seven columns beat all seventeen

Validation F1 by number of selected columns (`outputs/part_c_selectk.csv`): **0.6988** at k=5, **0.8333** at k=7, **0.7857** at k=10, **0.8049** at k=12 and **0.7848** at k=17. The peak is at k=7 and the worst score is at k=5, so the table brackets the answer from both sides — five columns is too few and drops something that matters, while every k above 7 is readmitting bookkeeping columns and scores lower than 7 does. Using all 17 columns (**0.7848**) is worse than using the best 7 (**0.8333**), which is the concrete case for the selector staying in the pipeline rather than feeding the forest everything available.

### Tuning did not beat the untuned forest on validation

On the same 7 columns, validation F1 was **0.775** for the grid search and **0.7848** for the randomized search, against **0.8333** for the untuned forest (`outputs/part_c_tuning.csv`). The untuned forest won, which is why it is the shipped configuration.

That ordering is not evidence that tuning does not work. The two searches optimise cross-validated accuracy on the training rows — the grid reached **0.8952** and the randomized search **0.8905** across 135 and 45 fits — while the comparison above is F1 on 180 validation rows holding only 42 delayed parcels. Those are different targets measured on different data: accuracy rewards getting the 138 on-time parcels right, F1 only tracks the delayed ones, and on a base of 42 a handful of parcels moving between TP and FN is enough to reverse the order. The honest conclusion is that this validation split is too small to resolve a difference this fine, not that the searches were a waste. Both searches sensibly converged on `max_depth` 6, and on a larger labelled history — or with the searches scored on F1 rather than accuracy — the result could easily flip back.

### Threshold 0.5, chosen because 0.6 misses too many parcels

The operations rule takes the highest threshold whose validation recall still reaches 0.80, because a missed delayed parcel costs more than an unnecessary call. That is **0.5**, with validation precision **0.7955** and recall **0.8333** (`outputs/part_c_thresholds.csv`): of 42 delayed parcels it catches **35** and misses **7**, at the cost of **9** false alarms.

The next threshold up fails the rule outright. At **0.6** recall falls to **0.7619**, below the 0.80 floor, even though it would look better on the metrics usually quoted — precision rises to **0.8889** and F1 to **0.8205**, the best F1 in the table. Concretely, moving from 0.5 to 0.6 saves five wasted calls (FP **9** down to **4**) but lets three more delayed parcels through unwarned (FN **7** up to **10**). The rule rejects that trade deliberately, and it is the reason the alert threshold is not set by F1: on this problem the cheapest-looking threshold is not the one operations can live with.

### On the test rows, recall falls to 0.7561

Scored once on the 180 untouched test rows at threshold **0.5**, the final model gives precision **0.6889**, recall **0.7561** and F1 **0.7209** (`outputs/part_c_final_test.csv`). In plain operational terms, about 7 of every 10 flagged parcels really are delayed, and about 7.5 of every 10 delayed parcels get caught.

Test recall of **0.7561** is lower than the validation recall of **0.8333** that the threshold was chosen on — a drop of **0.0772** — and it is below the 0.80 the operations rule asks for. That is worth stating plainly rather than hiding: the threshold was selected to clear 0.80 on one 180-row validation split, and it did not hold on the test split. The test set contains only **41** delayed parcels, so a single parcel moves recall by about **0.0244**, and the entire shortfall is **31** caught instead of the **33** that would have reached **0.8049**. A gap of two or three parcels on a base of 41 is well within what one split can produce, so this is not evidence the model is broken — but it is also not a result to explain away. The correct response is not to re-tune the threshold against the test rows, which would destroy the only honest estimate in the repository; it is to hold 0.5, report recall as approximately 0.76 to 0.83 rather than a single confident figure, and re-measure recall on real outcomes once a few hundred more delivered parcels are available.

## Operations recommendation

_Added in Part D._

## Production plan

_Added in Part D._

## Limitations

_Added in Part D._
