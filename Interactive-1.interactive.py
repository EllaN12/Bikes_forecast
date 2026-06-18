# %% [markdown]
# # Bikes Forecast — LSTM + ARIMA comparison
# Run cells **in order** from the repo root (`Bikes_forecast/`).
# Opens as **Interactive-1** in Cursor's Python Interactive Window.

# %% Setup — paths (run this cell first)
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "03_SRC"))

print("Working directory:", os.getcwd())
print("03_SRC on path:", str(ROOT / "03_SRC"))

# %% Imports
import pickle
import warnings

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from database import collect_data, summarize_by_time
from forecasting import compare_arima_lstm, data_prep
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# %% Load & summarise data
df = collect_data()
df.to_csv(ROOT / "outputs/data.csv", index=False)

bike_sales_df = summarize_by_time(
    data=df,
    date_column="order_date",
    groups="bikeshop_name",
    value_column="total_price",
    rule="ME",
    kind="period",
    agg_func=np.sum,
    wide_format=True,
)
df1 = bike_sales_df
print(f"Shape: {df1.shape}  |  Shops: {len(df1.columns)}  |  Months: {len(df1)}")
df1.tail()

# %% Train / val / test split + scale
WINDOW_SIZE = 3
BATCH_SIZE = 9
shuffle_buffer_size = 90
n_features = len(df1.columns)

train_size = int(len(df1) * 0.8)
dev_size = int(len(df1) * 0.1)

train_raw = df1.iloc[:train_size]
dev_raw = df1.iloc[train_size : train_size + dev_size]
test_raw = df1.iloc[train_size + dev_size :]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw.values)
dev_scaled = scaler.transform(dev_raw.values)
test_scaled = scaler.transform(test_raw.values)

train_df = pd.DataFrame(train_scaled, index=train_raw.index, columns=df1.columns)
dev_df = pd.DataFrame(dev_scaled, index=dev_raw.index, columns=df1.columns)
test_df = pd.DataFrame(test_scaled, index=test_raw.index, columns=df1.columns)


def create_X_y(frame, window_size):
    x, y = [], []
    for i in range(len(frame) - window_size):
        x.append(frame.values[i : i + window_size])
        y.append(frame.values[i + window_size])
    return np.array(x), np.array(y)


X_train, y_train = create_X_y(train_df, WINDOW_SIZE)
X_val, y_val = create_X_y(dev_df, WINDOW_SIZE)
X_test, y_test = create_X_y(test_df, WINDOW_SIZE)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .batch(BATCH_SIZE)
    .shuffle(shuffle_buffer_size)
)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

print(f"Train windows: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

# %% Model builder (for Keras Tuner)
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(WINDOW_SIZE, n_features)))

    num_layers = hp.Int("num_layers", 1, 3)
    for i in range(num_layers):
        model.add(
            tf.keras.layers.LSTM(
                units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                activation=hp.Choice(f"lstm_activation_{i}", values=["tanh", "relu"]),
                return_sequences=i < num_layers - 1,
                recurrent_activation=hp.Choice(
                    f"recurrent_activation_{i}", values=["sigmoid", "tanh"]
                ),
            )
        )
        if hp.Boolean(f"dropout_{i}"):
            model.add(
                tf.keras.layers.Dropout(
                    rate=hp.Float(f"dropout_rate_{i}", 0.0, 0.5, step=0.1)
                )
            )

    for i in range(hp.Int("num_dense_layers", 0, 2)):
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(f"dense_units_{i}", min_value=16, max_value=128, step=16),
                activation=hp.Choice(f"dense_activation_{i}", values=["relu", "tanh"]),
            )
        )

    model.add(tf.keras.layers.Dense(units=n_features))

    optimizer_name = hp.Choice("optimizer", values=["adam", "sgd", "nadam"])
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        )
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=hp.Float("learning_rate", 1e-4, 1e-1, sampling="log"),
            momentum=hp.Float("momentum", 0.0, 0.99),
        )
    else:
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        )

    model.compile(
        loss=hp.Choice("loss", values=["huber", "mse", "mae"]),
        optimizer=optimizer,
        metrics=["mae", "mse"],
    )
    return model

# %% Hyperparameter search (slow — 30+ min)
tuner_path = ROOT / "outputs/hyperband"
tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory=str(tuner_path),
    project_name="lstm_time_series_tuning",
    overwrite=True,
)

tuner.search(train_dataset, epochs=30, validation_data=dev_dataset)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
for param, value in best_hps.values.items():
    print(f"  {param}: {value}")

# %% Train best model + evaluate
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, epochs=50, validation_data=dev_dataset, verbose=1)

model_path = ROOT / "outputs/time_series_model.h5"
model.save(model_path, overwrite=True)

test_loss, test_mae, test_mse = model.evaluate(test_dataset)
print(f"Test loss: {test_loss:.4f}  |  MAE: {test_mae:.4f}  |  MSE: {test_mse:.4f}")

# %% Predict + save LSTM outputs
X_test_all = np.concatenate([x for x, _ in test_dataset], axis=0)
y_pred = model.predict(X_test_all)
y_pred_inverse = scaler.inverse_transform(y_pred)

pred_df = pd.DataFrame(y_pred_inverse, columns=df1.columns)
pred_path = ROOT / "outputs/Multivariate_time_series_predictions"
pred_df.to_pickle(pred_path)
print(f"Predictions saved → {pred_path}")
pred_df.tail()

# %% ARIMA vs LSTM comparison
# Use last h=3 forecast rows as LSTM input for data_prep
h = 3
lstm_df = pred_df.tail(h).copy()
lstm_df.index = df1.index[-h:]

bike_shop_predictions = data_prep(
    data=collect_data(),
    group="bikeshop_name",
    h=h,
    LSTM_df=lstm_df,
)

comparison_df = compare_arima_lstm(
    prediction_df=bike_shop_predictions,
    train_actuals=bike_sales_df,
    group="bikeshop_name",
    seasonal_period=1,
)

comparison_path = ROOT / "outputs/arima_vs_lstm_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print(comparison_df.head(10))
print("\nMean metrics by model (lower is better, MASE < 1 beats naive):")
summary = comparison_df.groupby("model")[["MAE", "RMSE", "MAPE", "MASE"]].mean()
print(summary)

# Save text summary for outputs/Multivariate_forecasting.txt
summary_path = ROOT / "outputs/Multivariate_forecasting.txt"
with open(summary_path, "w") as f:
    f.write("LSTM test metrics\n")
    f.write(f"  loss={test_loss:.6f}  MAE={test_mae:.6f}  MSE={test_mse:.6f}\n\n")
    f.write("Best hyperparameters\n")
    for param, value in best_hps.values.items():
        f.write(f"  {param}: {value}\n")
    f.write("\nARIMA vs LSTM comparison (mean by model)\n")
    f.write(summary.to_string())
    f.write("\n")
print(f"\nSummary written → {summary_path}")
