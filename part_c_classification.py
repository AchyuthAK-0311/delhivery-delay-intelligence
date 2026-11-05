"""Part C: delay classifier and alert threshold. Reads data/modelling_table.csv, builds a
stratified 840/180/180 split, compares a blind baseline against three classifiers, measures
which columns carry signal, tunes a 7-feature forest with grid and randomized search, then
picks the probability threshold that satisfies the operations rule (catch at least 80% of
delayed parcels) and scores the test rows exactly once.
Writes outputs/part_c_models.csv, part_c_feature_scores.csv, part_c_selectk.csv,
part_c_tuning.csv, part_c_thresholds.csv, part_c_final_test.csv and part_c_final_config.csv."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

# task 1: only about 23% of parcels are delayed, so both splits stratify on the target —
# without it a split could hold too few delayed parcels to measure recall meaningfully
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


def confusion_metrics(y_true, y_pred):
    """Four confusion cells plus the four scores they imply; precision is 0 when nothing is flagged."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# task 2: the blind rule flags nothing, so it is the accuracy a model must beat to be worth
# running at all; the tree depth is chosen from the validation F1 column, never hard-coded
model_rows = []

blind_pred = np.zeros(len(y_val), dtype=int)
model_rows.append({"model": "blind_all_on_time", **confusion_metrics(y_val, blind_pred)})

logistic = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
logistic_metrics = confusion_metrics(y_val, logistic.predict(X_val))

tree_candidates = []
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    tree_candidates.append({"depth": d, "metrics": confusion_metrics(y_val, tree.predict(X_val))})
# highest validation F1 wins; a tie goes to the smaller depth, which is the simpler tree
best_tree = sorted(tree_candidates, key=lambda c: (-c["metrics"]["f1"], c["depth"]))[0]
best_tree_depth = best_tree["depth"]

forest_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
forest_metrics = confusion_metrics(y_val, forest_default.predict(X_val))

model_rows.append({"model": "logistic_regression", **logistic_metrics})
model_rows.append({"model": f"decision_tree_depth_{best_tree_depth}", **best_tree["metrics"]})
model_rows.append({"model": "random_forest_default", **forest_metrics})

models_df = pd.DataFrame(model_rows)
score_cols = ["accuracy", "precision", "recall", "f1"]
models_df[score_cols] = models_df[score_cols].round(4)
models_df.to_csv("outputs/part_c_models.csv", index=False)

# task 3: two independent views of signal — the forest's own importances and a univariate
# F-test — so the five bookkeeping columns can be shown to carry nothing by both measures
importance_forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)

feature_scores = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": selector_all.scores_,
        "rf_importance": importance_forest.feature_importances_,
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    }
)
feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="min").astype(int)
feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
feature_scores[["f_score", "rf_importance"]] = feature_scores[["f_score", "rf_importance"]].round(4)
feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline(
        [("selector", SelectKBest(f_classif, k=k)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))]
    ).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val), zero_division=0)})
selectk_df = pd.DataFrame(selectk_rows)
selectk_df["val_f1"] = selectk_df["val_f1"].round(4)
selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)

# task 4: the selector is fitted on the training rows only and then applied to the validation
# rows, so the F-test never sees a validation label; the test rows are untouched until task 7
SELECTOR_K = 7
selector = SelectKBest(f_classif, k=SELECTOR_K).fit(X_train, y_train)
X_train_sel = selector.transform(X_train)
X_val_sel = selector.transform(X_val)

param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
CV_FOLDS = 5
N_ITER = 9

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=CV_FOLDS).fit(X_train_sel, y_train)
randomized = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_grid, n_iter=N_ITER, cv=CV_FOLDS, random_state=42
).fit(X_train_sel, y_train)

grid_candidates = len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_leaf"])
default_params = {"max_depth": None, "min_samples_leaf": 1, "n_estimators": 100}
default_forest = RandomForestClassifier(random_state=42, **default_params).fit(X_train_sel, y_train)

# the parameters behind each row, kept as dictionaries so task 5 can rebuild the winner
# without parsing the string written to the CSV
candidate_params = {
    "grid_search": grid.best_params_,
    "randomized_search": randomized.best_params_,
    "default_forest_same_7_features": default_params,
}

tuning_df = pd.DataFrame(
    [
        {
            "search": "grid_search",
            "candidates": grid_candidates,
            "cv_folds": CV_FOLDS,
            "total_fits": grid_candidates * CV_FOLDS,
            "best_params": str(grid.best_params_),
            "best_cv_accuracy": grid.best_score_,
            "val_f1": f1_score(y_val, grid.best_estimator_.predict(X_val_sel), zero_division=0),
        },
        {
            "search": "randomized_search",
            "candidates": N_ITER,
            "cv_folds": CV_FOLDS,
            "total_fits": N_ITER * CV_FOLDS,
            "best_params": str(randomized.best_params_),
            "best_cv_accuracy": randomized.best_score_,
            "val_f1": f1_score(y_val, randomized.best_estimator_.predict(X_val_sel), zero_division=0),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,  # never cross-validated, so the column stays blank
            "val_f1": f1_score(y_val, default_forest.predict(X_val_sel), zero_division=0),
        },
    ]
)
tuning_df[["best_cv_accuracy", "val_f1"]] = tuning_df[["best_cv_accuracy", "val_f1"]].round(4)
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

# task 5: highest val_f1 wins; the stated tie order prefers the untuned forest, then the grid,
# then the randomized search, so a tie never rewards the more complicated search
TIE_ORDER = ["default_forest_same_7_features", "grid_search", "randomized_search"]
chosen_row = sorted(
    tuning_df.to_dict("records"), key=lambda r: (-r["val_f1"], TIE_ORDER.index(r["search"]))
)[0]
chosen_from = chosen_row["search"]
chosen_params = candidate_params[chosen_from]

final_model = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=SELECTOR_K)),
        ("model", RandomForestClassifier(random_state=42, **chosen_params)),
    ]
).fit(X_train, y_train)

# task 6: thresholds are written as exact one-decimal literals — a floating-point range would
# produce values like 0.30000000000000004 and change which rows clear the recall rule
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_proba = final_model.predict_proba(X_val)[:, 1]

threshold_rows = []
for t in THRESHOLDS:
    threshold_rows.append({"threshold": t, **confusion_metrics(y_val, (val_proba >= t).astype(int))})
thresholds_df = pd.DataFrame(threshold_rows)
thresholds_df[score_cols] = thresholds_df[score_cols].round(4)
thresholds_df.to_csv("outputs/part_c_thresholds.csv", index=False)

# a missed delayed parcel costs more than an unnecessary call, so recall is the binding
# constraint: take the strictest (highest) threshold that still catches 80% of delayed parcels
qualifying = thresholds_df[thresholds_df["recall"] >= 0.80]
chosen_threshold = float(qualifying["threshold"].max()) if len(qualifying) else 0.1

# ---------------------------------------------------------------------------
# task 7: FINAL TEST EVALUATION — the only place the test rows are used, scored once
# at the chosen threshold with the model already fitted on the training rows
# ---------------------------------------------------------------------------
test_proba = final_model.predict_proba(X_test)[:, 1]
test_metrics = confusion_metrics(y_test, (test_proba >= chosen_threshold).astype(int))

final_test_df = pd.DataFrame([{"threshold": chosen_threshold, **test_metrics}])
final_test_df[score_cols] = final_test_df[score_cols].round(4)
final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)

final_config_df = pd.DataFrame(
    [
        {
            "selector_k": SELECTOR_K,
            "n_estimators": chosen_params["n_estimators"],
            "max_depth": "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"],
            "min_samples_leaf": chosen_params["min_samples_leaf"],
            "threshold": chosen_threshold,
            "chosen_from": chosen_from,
        }
    ]
)
final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)
