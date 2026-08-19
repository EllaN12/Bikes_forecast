# Data Card — Bikeshop Orders Database (Synthetic)

**Project:** `Bikes_forecast`  
**Owner:** Ella Ndalla (ndallaella@gmail.com)  
**Profiling:** pandas profile of the committed data in this repository — 3 table(s)

A synthetic relational sales database for a fictitious bicycle manufacturer selling through independent bikeshops, used to forecast quarterly revenue with AutoARIMA and an LSTM. Three tables: bike catalog, bikeshop registry, and order lines.

## Provenance

- **Source:** Synthetic dataset generated in Python (Business Science teaching dataset lineage)
- **Snapshot:** Order history spanning 2011-2015 in the raw file, presented as 2020-2024
- **License / terms:** MIT, as part of this repository. No third-party restrictions — the data is generated, not collected.

### Collection

Fully synthetic. Records were generated programmatically; no real company, shop, customer, or transaction is represented. Raw source files are Excel workbooks in 00_data_raw/, loaded into a SQLite database at 01_database/bike_orders_database.sqlite via collect_data() in 02_SRC/database.py.

### Maintenance

Static. Regenerating or extending the series requires re-running the generator; there is no scheduled refresh.

## Labeling

No labels in the supervised sense. The forecasting target is monthly aggregated revenue, derived arithmetically as quantity x price and rolled up by summarize_by_time().

## Preprocessing

Date parsing; category and location strings split into component fields; monthly time-series aggregation by shop, category, and sub-category. For the LSTM, inputs are Min-Max scaled and windowed with a 3-month look-back across all 30 bikeshops simultaneously; the scaler is fit only on the training split to prevent leakage.

## Splits

Last 3 months held out as a true out-of-sample window for both models. Train/validation/test split is applied BEFORE scaling. Both models are scored on the same holdout via evaluate_arima_holdout() so MASE figures are comparable.

## Sensitive & Personal Data

None. The data is synthetic — no personal, financial, or commercially confidential information is present.

## Recommended Uses

- Demonstrating time-series forecasting workflow and MASE-based evaluation against a naive baseline.
- Teaching relational-to-time-series aggregation patterns.

## Discouraged Uses

- Drawing any conclusion about the real bicycle retail market.
- Treating the dollar error figures as calibrated to real revenue scales.

## Known Issues, Skews & Gaps

- The committed database is substantially smaller than the README states. Profiled counts: bikes 97 rows (README says 4,753), bikeshops 30 (README says 93), orderlines 15,644 (README says 109,514). The raw Excel files in 00_data_raw/ match the profiled numbers, so the README figures appear to describe a different or expanded generation run. The 30 bikeshops figure is consistent with the modeling code, which trains across 30 series.
- Missing values: none in any of the three tables.
- orderlines carries an unnamed index column inherited from the Excel export.
- Profiled order dates run 2011-01-07 to 2015-12-25, while the README and dashboard present the series as 2020-2024. The series is shifted for presentation, so absolute dates are not meaningful and the 'Q1 2025 forecast' is a relabeled Q1 2016.

## Profiled Schema

### `bikes`

**Rows:** 97  
**Columns:** 4  
**Duplicate rows:** 0  
**Source file:** `Bikes_forecast/01_database/bike_orders_database.sqlite`

> Full tables profiled.

| Field | Type | Null % | Distinct | Range / top values |
|---|---|---:|---:|---|
| `bike.id` | int64 | 0.00 | 97 | min 1 · median 49 · max 97 · mean 49 |
| `model` | object | 0.00 | 97 |  |
| `description` | object | 0.00 | 13 | Mountain - Cross Country Race - Carbon 14.43% · Road - Elite Road - Aluminum 11.34% · Road - Endurance Road - Carbon 11.34% |
| `price` | int64 | 0.00 | 53 | min 415 · median 3,200 · max 12,790 · mean 3,954 |

### `bikeshops`

**Rows:** 30  
**Columns:** 3  
**Duplicate rows:** 0  
**Source file:** `Bikes_forecast/01_database/bike_orders_database.sqlite`

> Full tables profiled.

| Field | Type | Null % | Distinct | Range / top values |
|---|---|---:|---:|---|
| `bikeshop.id` | int64 | 0.00 | 30 | min 1 · median 15.5 · max 30 · mean 15.5 |
| `bikeshop.name` | object | 0.00 | 30 |  |
| `location` | object | 0.00 | 30 |  |

### `orderlines`

**Rows:** 15,644  
**Columns:** 6  
**Duplicate rows:** 0  
**Source file:** `Bikes_forecast/01_database/bike_orders_database.sqlite`

> Full tables profiled.

| Field | Type | Null % | Distinct | Range / top values |
|---|---|---:|---:|---|
| `order.id` | int64 | 0.00 | 2,000 | min 1 · median 985.5 · max 2,000 · mean 998 |
| `order.line` | int64 | 0.00 | 30 | min 1 · median 7 · max 30 · mean 8.472 |
| `order.date` | object | 0.00 | 962 |  |
| `customer.id` | int64 | 0.00 | 30 | min 1 · median 10 · max 30 · mean 13.46 |
| `product.id` | int64 | 0.00 | 97 | min 1 · median 48 · max 97 · mean 49.48 |
| `quantity` | int64 | 0.00 | 10 | min 1 · median 1 · max 10 · mean 1.289 |

---

*Schema, null rates, cardinality, and distributions were computed directly from the committed data. Narrative sections are documented from the project README and source materials.*
