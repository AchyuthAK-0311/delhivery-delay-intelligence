"""Shared cleaning and feature logic for the Delay Intelligence capstone.
Every script (Parts A-D, package_model.py, predict_delay.py) imports clean_shipments
and build_features from here so the same rules run at training time and at
prediction time on parcels that have not been delivered yet."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    """Turn a messy shipments export into a clean table. Returns (clean_df, medians)."""
    df = raw.drop_duplicates().reset_index(drop=True)  # step 1: exact duplicate rows carry no new information

    df["service_type"] = df["service_type"].str.strip().str.title()  # step 2: collapse casing/whitespace variants to one spelling

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")  # step 3: missing payment mode is its own category, not dropped

    # step 4: declared_value text ("₹1,290") -> numeric declared_value_inr; the text column is no longer needed
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])  # step 5

    # step 6: the network accepts parcels up to 40 kg, so a reading above 100 is a scanner fault, not a real weight
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # step 7: medians are computed once on the full historical table and reused as-is on new parcels at
    # prediction time, so filling a gap never leaks information the model did not have at training time
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }
    for col, value in medians.items():
        df[col] = df[col].fillna(value)

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derived features. Never reads delivery_hours, so it
    also runs on parcels that have not been delivered yet (data/new_shipments.csv)."""
    df = clean.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id", how="left",
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
