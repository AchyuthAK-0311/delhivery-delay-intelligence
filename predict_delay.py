"""Part D task 6: turn the parcels about to be dispatched into an operations action list.
Loads only data/new_shipments.csv, data/hubs.csv, the four artefacts in models/ and
outputs/hub_segments.csv — no training data and no second copy of the cleaning rules, which
come from src/shipment_prep.py so prediction preprocesses exactly as training did.
Writes outputs/new_shipment_actions.csv."""
import os

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")

model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_values = pd.read_csv("models/impute_values.csv")

# the stored medians are the training-time values, so a missing distance or weight in
# today's batch is never filled from today's batch
medians = dict(zip(impute_values["column"], impute_values["median"].astype(float)))

clean, _ = clean_shipments(new_raw, medians=medians)
features = build_features(clean, hubs)

# the pipeline's selector was fitted on columns in this exact order, so the order is read
# back from the artefact rather than assumed
X_new = features[feature_order]

# round first, then compare against the threshold, so the flag can never disagree with the
# probability printed next to it in the file
delay_probability = model.predict_proba(X_new)[:, 1].round(4)

actions = features[["shipment_id", "hub_id"]].copy()
actions["delay_probability"] = delay_probability
actions["flagged"] = (delay_probability >= threshold).astype(int)
actions = actions.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")

# an unflagged parcel costs nothing to leave alone; a flagged parcel at a hub that is
# already struggling needs a person, not a phone call
is_escalate_hub = actions["segment"] == "Escalate"
actions["action"] = "No action"
actions.loc[actions["flagged"] == 1, "action"] = "Proactive customer call"
actions.loc[(actions["flagged"] == 1) & is_escalate_hub, "action"] = "Escalate to hub manager"

actions = actions[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
actions.to_csv("outputs/new_shipment_actions.csv", index=False)
