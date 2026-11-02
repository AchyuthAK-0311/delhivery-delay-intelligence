"""Part B: delivery-hours regression. Reads data/modelling_table.csv, splits it into
training/validation/test, compares five models (a mean baseline, linear regression, a
decision tree, a random forest and gradient boosting) on validation RMSE, cross-validates
the two best, and scores the single winner on the test rows exactly once.
Writes outputs/part_b_checks.csv, part_b_tree_depth.csv, part_b_metrics.csv,
part_b_gb_learning_rate.csv and part_b_cv.csv."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["delivery_hours"]

# task 1: 70/15/15 split via two binary splits, no stratify (that is Part C's job on the
# classification target) — the brief pins random_state=42 on both calls for reproducibility
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_row(model_name, split, y_true, y_pred):
    return {
        "model": model_name,
        "split": split,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# task 2: the scaler is fitted on the 840 training rows only and reused as-is on the
# test rows, so the test-row check below only passes if no test-row statistic leaked in
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())

checks = pd.DataFrame(
    [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(train_scaled_max_abs_mean, 4)),
        ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4)),
    ],
    columns=["metric", "value"],
)
checks.to_csv("outputs/part_b_checks.csv", index=False)

# task 3: the mean baseline never sees a feature, so any model that beats it is using the
# features for something; it is scored on validation like every other candidate below
baseline_pred = np.full(len(y_val), y_train.mean())
baseline_metrics = metrics_row("mean_baseline", "val", y_val, baseline_pred)

# task 4: depth is tuned on validation RMSE, never hard-coded, so "unlimited" (max_depth=None)
# competes on equal footing with every finite depth
depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_depth_rows = []
for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tree_depth_rows.append(
        {
            "max_depth": "unlimited" if d is None else d,
            "train_rmse": rmse(y_train, tree.predict(X_train)),
            "val_rmse": rmse(y_val, tree.predict(X_val)),
        }
    )
tree_depth_df = pd.DataFrame(tree_depth_rows)
tree_depth_df[["train_rmse", "val_rmse"]] = tree_depth_df[["train_rmse", "val_rmse"]].round(4)
tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

# "unlimited" sorts after every finite depth so a tie only prefers it when no finite depth matches
depth_sort_key = tree_depth_df["max_depth"].apply(lambda v: float("inf") if v == "unlimited" else v)
best_row = tree_depth_df.assign(_depth_key=depth_sort_key).sort_values(["val_rmse", "_depth_key"]).iloc[0]
best_depth_label = best_row["max_depth"]
best_depth = None if best_depth_label == "unlimited" else int(best_depth_label)

# task 5: four candidates, each fitted on the training rows only and scored on validation;
# linear regression carries its own scaler in a Pipeline so it can be cross-validated safely later
def make_model(name):
    if name == "linear_regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "decision_tree":
        return DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    raise ValueError(name)


candidate_names = ["linear_regression", "decision_tree", "random_forest", "gradient_boosting"]
fitted_models = {}
val_rows = [baseline_metrics]
for name in candidate_names:
    model = make_model(name).fit(X_train, y_train)
    fitted_models[name] = model
    val_rows.append(metrics_row(name, "val", y_val, model.predict(X_val)))

metrics_df = pd.DataFrame(val_rows)

# task 6: same three learning rates, same n_estimators=100, fit fresh each time and scored on
# validation only — this experiment is independent of the task-5 gradient_boosting candidate
gb_lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    pred = gb.predict(X_val)
    gb_lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
gb_lr_df = pd.DataFrame(gb_lr_rows)
gb_lr_df[["val_rmse", "val_r2"]] = gb_lr_df[["val_rmse", "val_r2"]].round(4)
gb_lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# task 7: rank the four candidates by validation RMSE (never hard-coded) and cross-validate
# only the two best; the final model is simply the single best of the four
ranked = metrics_df[metrics_df["model"] != "mean_baseline"].sort_values("RMSE")
best_two = ranked["model"].iloc[:2].tolist()
final_name = ranked["model"].iloc[0]

# one KFold object is shared by both models so they see exactly the same five splits —
# otherwise a difference in mean_r2 could come from the split, not the model
X_cv = pd.concat([X_train, X_val], ignore_index=True)
y_cv = pd.concat([y_train, y_val], ignore_index=True)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for name in best_two:
    scores = cross_val_score(make_model(name), X_cv, y_cv, cv=kfold)  # default scoring is R2
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
cv_df = pd.DataFrame(cv_rows)
cv_df[["mean_r2", "std_r2"]] = cv_df[["mean_r2", "std_r2"]].round(4)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)

# the final model is already fitted on the training rows (task 5); it is scored on the
# test rows here for the first and only time
final_model = fitted_models[final_name]
test_row = metrics_row(final_name, "test", y_test, final_model.predict(X_test))
metrics_df = pd.concat([metrics_df, pd.DataFrame([test_row])], ignore_index=True)
metrics_df[["MAE", "MSE", "RMSE", "R2"]] = metrics_df[["MAE", "MSE", "RMSE", "R2"]].round(4)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
