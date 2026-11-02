"""Part A: builds the modelling table from the raw shipments export and writes the
evidence files the README's Data findings section quotes from.
Reads data/raw_shipments.csv and data/hubs.csv; writes data/modelling_table.csv plus
outputs/part_a_checks.csv, outputs/part_a_delay_by_group.csv and
outputs/part_a_correlations.csv. No later script repeats this cleaning logic — every
other script imports clean_shipments and build_features from src/shipment_prep.py."""
import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

raw_rows = len(raw)

# "Before" metrics are measured once, after duplicates are removed but before any
# missing value is filled or the sentinel weight is converted to a gap, so they
# describe the export exactly as operations handed it over.
dedup = raw.drop_duplicates().reset_index(drop=True)
duplicate_rows_removed = raw_rows - len(dedup)
rows_after_dedup = len(dedup)

missing_distance_km_before = int(dedup["distance_km"].isna().sum())
missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())  # excludes the 999.0 sentinel rows
sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())
missing_declared_value_before = int(dedup["declared_value"].isna().sum())
missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())
service_type_spellings_before = int(dedup["service_type"].nunique())
service_type_spellings_after = int(dedup["service_type"].str.strip().str.title().nunique())

clean, medians = clean_shipments(raw)
features = build_features(clean, hubs)
features["is_delayed"] = (features["delivery_hours"] > features["promised_hours"]).astype(int)

modelling_table = features.copy()
float_cols = modelling_table.select_dtypes(include=["float64", "float32"]).columns
modelling_table[float_cols] = modelling_table[float_cols].round(4)
modelling_table.to_csv("data/modelling_table.csv", index=False)

rows_after_hub_merge = len(features)
nulls_in_features_and_target = int(features[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())
delayed_count = int(features["is_delayed"].sum())
delay_rate = delayed_count / len(features)

# the four declared-value metrics are computed on the median-filled declared_value_inr
# column of the modelling table, not on the non-missing values alone (the brief notes
# that the two differ), so they are read straight from `features`
declared_value_mean_inr = features["declared_value_inr"].mean()
declared_value_median_inr = features["declared_value_inr"].median()
log_declared_value_mean = features["log_declared_value"].mean()
log_declared_value_median = features["log_declared_value"].median()

# bounds are computed from the unrounded quartiles and rounded only when written,
# per the brief, so the outlier count is not sensitive to display rounding
hours_q1 = features["delivery_hours"].quantile(0.25)
hours_q3 = features["delivery_hours"].quantile(0.75)
iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * iqr
hours_upper_bound = hours_q3 + 1.5 * iqr
hours_outlier_count = int(
    ((features["delivery_hours"] < hours_lower_bound) | (features["delivery_hours"] > hours_upper_bound)).sum()
)

checks = pd.DataFrame(
    [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("declared_value_mean_inr", declared_value_mean_inr),
        ("declared_value_median_inr", declared_value_median_inr),
        ("log_declared_value_mean", log_declared_value_mean),
        ("log_declared_value_median", log_declared_value_median),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", hours_q1),
        ("hours_q3", hours_q3),
        ("hours_lower_bound", hours_lower_bound),
        ("hours_upper_bound", hours_upper_bound),
        ("hours_outlier_count", hours_outlier_count),
    ],
    columns=["metric", "value"],
)
checks["value"] = checks["value"].apply(lambda v: round(v, 4) if isinstance(v, float) else v)
checks.to_csv("outputs/part_a_checks.csv", index=False)


def group_summary(dimension, series):
    g = features.groupby(series, observed=True)
    out = pd.DataFrame(
        {
            "dimension": dimension,
            "group": g.size().index.astype(str),
            "shipments": g.size().values,
            "delayed": g["is_delayed"].sum().values,
            "mean_hours": g["delivery_hours"].mean().values,
        }
    )
    out["delay_rate"] = (out["delayed"] / out["shipments"]).round(4)
    out["mean_hours"] = out["mean_hours"].round(4)
    return out[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]]


distance_band = pd.cut(
    features["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    include_lowest=True,
)

delay_by_group = pd.concat(
    [
        group_summary("service_type", features["service_type"]),
        group_summary("hub_tier", features["hub_tier"]),
        group_summary("distance_band", distance_band),
    ],
    ignore_index=True,
)
delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# row labels are kept as the first column (named "feature") so the matrix can be
# read back and indexed by column name, unlike a plain index=False dump
correlations = modelling_table.select_dtypes(include=[np.number]).corr().round(4)
correlations.index.name = "feature"
correlations.to_csv("outputs/part_a_correlations.csv")
