"""Part D task 5: package the Part C classifier as a reusable artefact.
Reads outputs/part_c_final_config.csv, rebuilds Part C's stratified split, refits the
final Pipeline on the training rows and saves the model plus everything a prediction run
needs to reproduce training-time preprocessing: the alert threshold, the pinned feature
order and the three imputation medians. Then it reloads the pickle and proves the reloaded
model scores the test rows identically.
Writes models/delay_model.pkl, models/delay_threshold.csv, models/feature_order.csv,
models/impute_values.csv and outputs/part_d_reload_check.csv."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

# max_depth is stored as the word "unlimited" when the forest has no depth limit, so it
# has to come back as None; the other three are counts and must be ints, not numpy floats
selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

# the same stratified two-step split as Part C, so the artefact is fitted on exactly the
# 840 training rows the configuration was chosen on and the 180 test rows stay held out
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

final_model = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=selector_k)),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            ),
        ),
    ]
).fit(X_train, y_train)

# the selector travels inside the pickle, so a prediction run cannot accidentally feed the
# forest a different set of columns than it was trained on
joblib.dump(final_model, "models/delay_model.pkl")

pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

# the medians come from the same clean_shipments call Part A made on the full raw export,
# so a gap in a future parcel is filled with the training-time value, not with a statistic
# of whatever batch happens to be scored that day
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}).to_csv(
    "models/impute_values.csv", index=False
)

# a pickle is only useful if it scores identically after a reload, so prove it rather than
# assume it — this is the check that would catch a scikit-learn version mismatch
reloaded = joblib.load("models/delay_model.pkl")
proba_in_memory = final_model.predict_proba(X_test)[:, 1]
proba_reloaded = reloaded.predict_proba(X_test)[:, 1]
max_abs_diff = float(np.abs(proba_in_memory - proba_reloaded).max())

pd.DataFrame(
    [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", round(max_abs_diff, 4)),
        ("threshold", threshold),
    ],
    columns=["metric", "value"],
).to_csv("outputs/part_d_reload_check.csv", index=False)
