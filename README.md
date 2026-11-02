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

_Added in Part B._

## Model results (classification)

_Added in Part C._

## Operations recommendation

_Added in Part D._

## Production plan

_Added in Part D._

## Limitations

_Added in Part D._
