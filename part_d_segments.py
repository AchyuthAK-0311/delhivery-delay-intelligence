"""Part D (tasks 1-4): hub operating segments. Reads data/modelling_table.csv and
data/hubs.csv, profiles each of the 24 hubs, clusters them into the three playbooks
operations actually runs, measures how well-separated the clusters are for K = 2 to 5,
and lists the hubs whose tier label disagrees with how they really behave.
Writes outputs/hub_profile.csv, part_d_separation.csv, hub_segments.csv,
segment_profile.csv and tier_vs_segment_mismatch.csv."""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# task 1: one row per hub. sort_index() makes the ascending hub_id order explicit rather
# than relying on groupby's default, because every later step depends on this row order
profile = (
    table.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .sort_index()
    .reset_index()
)

profile_out = profile.copy()
float_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
profile_out[float_cols] = profile_out[float_cols].round(4)
profile_out.to_csv("outputs/hub_profile.csv", index=False)

# task 2: K-Means measures straight-line distance, so the four columns are standardised
# first — delay_rate spans about 0 to 0.5 while mean_distance_km spans hundreds, and
# without scaling the distance column alone would decide every cluster
CLUSTER_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[CLUSTER_COLUMNS])

# K = 3 is fixed by the business: operations runs exactly three playbooks
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
labels = kmeans.labels_

# task 3: all 276 unordered pairs of the 24 hubs, measured on the scaled matrix
diffs = scaled[:, None, :] - scaled[None, :, :]
pair_distances = np.sqrt((diffs**2).sum(axis=2))
upper = np.triu_indices(len(scaled), k=1)
distances = pair_distances[upper]

separation_rows = []
for k in [2, 3, 4, 5]:
    # the same seed and n_init as task 2, so the K = 3 row here reproduces the labels
    # the segments below are built from
    k_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled).labels_
    same_cluster = k_labels[upper[0]] == k_labels[upper[1]]
    sizes = pd.Series(k_labels).value_counts().sort_values(ascending=False)
    separation_rows.append(
        {
            "K": k,
            "across_over_within": distances[~same_cluster].mean() / distances[same_cluster].mean(),
            "cluster_sizes": "-".join(str(s) for s in sizes),
        }
    )

separation = pd.DataFrame(separation_rows)
separation["across_over_within"] = separation["across_over_within"].round(4)
separation.to_csv("outputs/part_d_separation.csv", index=False)

# task 4: the cluster numbers K-Means hands back are arbitrary, so the names come from
# ranking the clusters by their hubs' mean delay_rate — lowest is Standard, highest Escalate
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]
cluster_delay_rate = profile.assign(cluster=labels).groupby("cluster")["delay_rate"].mean().sort_values()
cluster_to_segment = dict(zip(cluster_delay_rate.index, SEGMENT_NAMES))

segmented = profile.assign(cluster=labels)
segmented["segment"] = segmented["cluster"].map(cluster_to_segment)
segmented = segmented.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")

hub_segments = segmented[["hub_id", "segment", "hub_city", "hub_tier"]]
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

# means of the hub-level values, so every hub counts once regardless of its parcel count
segment_order = pd.Categorical(segmented["segment"], categories=SEGMENT_NAMES, ordered=True)
segment_profile = (
    segmented.groupby(segment_order, observed=True)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
    .reset_index(names="segment")
)
segment_profile[float_cols] = segment_profile[float_cols].round(4)
segment_profile.to_csv("outputs/segment_profile.csv", index=False)

# a tier label is only useful if it predicts behaviour; these are the hubs where it does not
TIER_TO_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = hub_segments[hub_segments["segment"] != hub_segments["hub_tier"].map(TIER_TO_SEGMENT)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
