





# Model Card for Bikes Sales Forecast — AutoARIMA and TensorFlow LSTM

## Model Details

### Overview
Forecasts Q1 2025 revenue for a fictitious bicycle manufacturer distributing to 30 bikeshops, using AutoARIMA (univariate, per series) and a multivariate TensorFlow LSTM trained jointly across all shops. Built on 109,514 orderlines spanning 2020-2024 in SQLite, surfaced through a Streamlit dashboard on Google Cloud Run. Both models are scored against a naive last-value baseline on a true 3-month holdout. 

### Version

name: 1.0.0  

### Owners

* Ella Ndalla, ndallaella@gmail.com


### Licenses

* MIT

### References

* [https://bikes-forecast-6trxjbynbq-uc.a.run.app](https://bikes-forecast-6trxjbynbq-uc.a.run.app)


### Citations

* Ella Ndalla. Bikes Sales Forecast — AutoARIMA and TensorFlow LSTM. GitHub repository: Bikes_forecast.



## Considerations

### Users

* Sales and demand planning stakeholders (illustrative)

* Data scientists reviewing time-series forecasting practice


### Use Cases

* Monthly revenue forecasting at bikeshop, category, and sub-category level.

* Demonstration of MASE-based evaluation against a naive baseline for sparse retail series.


### Limitations

* Trained on synthetic data; absolute error figures carry no real-world meaning.

* LSTM predictions are pre-computed and committed to 03_outputs/ — the deployed app does not retrain and TensorFlow is not a runtime dependency, so forecasts are static until the training pipeline is re-run.

* MAPE is unusable on this dataset: near-zero bikeshop months produce percentage errors in the quadrillions. MASE is the only reliable headline metric here.

* 3-month look-back window and 30-epoch training cap limit the LSTM&#39;s ability to capture longer seasonal cycles.


### Tradeoffs

* LSTM wins on accuracy (MASE 0.703 vs 0.778) but requires GPU-class retraining and produces no confidence intervals; AutoARIMA is slower per-series but interpretable, CPU-friendly, and yields 95% prediction intervals.

* Joint multivariate training shares signal across shops but couples all 30 series to a single model artifact.


### Ethical Considerations

* Risk: Over-trusting point forecasts for inventory or staffing decisions when the LSTM provides no uncertainty bounds.
  * Mitigation Strategy: AutoARIMA 95% prediction intervals are surfaced alongside LSTM point forecasts in the dashboard; MASE vs naive is shown so users can see how much the model actually beats a trivial baseline.

* Risk: Silent staleness — committed predictions continue to serve after the underlying sales pattern changes.
  * Mitigation Strategy: Holdout evaluation scripts (evaluate_arima_holdout, compare_arima_lstm) are re-runnable to re-score before each refresh.

## Graphics



## Metrics

|Name|Value|
-----|------
|MASE (LSTM, mean over 30 bikeshops)|0.703|
|MASE (AutoARIMA)|0.778|
|MASE (Naive baseline)|1.000|
|MAE (LSTM)|$31,981|
|MAE (AutoARIMA)|$34,688|
|MAE (Naive)|$26,612|
|RMSE (LSTM)|$38,165|
|RMSE (AutoARIMA)|$39,689|

