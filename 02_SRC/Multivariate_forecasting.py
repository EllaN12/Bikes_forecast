# %% Setup
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "03_SRC"))

from database import collect_data, summarize_by_time, ROOT, OUTPUT_DIR

# %%
import h5py
from forecasting import compare_arima_lstm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#Forecasting using LSTM BY Category
# %%
df = collect_data()
df.to_csv(OUTPUT_DIR / "data.csv", index=False)


Cat_1_bikes_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    groups= 'category_1',
    value_column = "total_price",
    rule = "ME",
    kind = "period",
    agg_func = np.sum,
    wide_format = True,
    #fillna = np.nan
)

Cat_1_bikes_sales_df

Cat_2_bikes_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    groups= 'category_2',
    value_column = "total_price",
    rule = "ME",
    kind = "period",
    agg_func = np.sum,
    wide_format = True,
    #fillna = np.nan
)

Cat_2_bikes_sales_df

df1 = Cat_1_bikes_sales_df.merge(Cat_2_bikes_sales_df, on = 'order_date', how = 'outer')


bike_sales_df = summarize_by_time(
    data = df,
    date_column= "order_date",
    groups= 'bikeshop_name',
    value_column= "total_price",
    rule = "ME",
    kind = 'period',
    agg_func = np.sum,
    wide_format = True

)
df1 = bike_sales_df 

# Define parameters
n_features = len(df1.columns) # number of features/category of bikes

WINDOW_SIZE = 3
BATCH_SIZE = 9


# split the data into train, val, and test BEFORE scaling to prevent data leakage
train_size = int(len(df1) * 0.8)
dev_size = int(len(df1) * 0.1)

train_raw = df1.iloc[:train_size]
dev_raw = df1.iloc[train_size:train_size + dev_size]
test_raw = df1.iloc[train_size + dev_size:]

# instantiate MinMaxScaler and fit ONLY on training data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw.values)
dev_scaled = scaler.transform(dev_raw.values)
test_scaled = scaler.transform(test_raw.values)

train_df = pd.DataFrame(train_scaled, index=train_raw.index, columns=df1.columns)
dev_df = pd.DataFrame(dev_scaled, index=dev_raw.index, columns=df1.columns)
test_df = pd.DataFrame(test_scaled, index=test_raw.index, columns=df1.columns)


def create_X_y(df, WINDOW_SIZE):
    X = []
    y = []
    for i in range(len(df) - WINDOW_SIZE):
        X.append(df.values[i:i+WINDOW_SIZE])
        y.append(df.values[i+WINDOW_SIZE])
    return np.array(X), np.array(y)


X_train, y_train = create_X_y(train_df, WINDOW_SIZE)
X_val, y_val = create_X_y(dev_df, WINDOW_SIZE)
X_test, y_test = create_X_y(test_df, WINDOW_SIZE)

# Creating tensorFlow datasets
# shuffle buffer must not exceed dataset size — val/test are never shuffled
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE)
dev_dataset   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)






print("Current working directory:", os.getcwd())

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Fixed lightweight architecture — bypasses Keras Tuner for fast execution.
# Tuner search over 93 bikeshop features takes hours on CPU; this gives a
# valid trained model in under 2 minutes and preserves the full pipeline.
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, n_features)),
    tf.keras.layers.LSTM(32, activation="tanh"),
    tf.keras.layers.Dense(n_features),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae", "mse"])

history = model.fit(
    train_dataset, epochs=30,
    validation_data=dev_dataset,
    callbacks=[early_stop],
    verbose=1,
)

print(model.summary())

model_path = OUTPUT_DIR / "time_series_model.h5"
model.save(str(model_path), overwrite=True)
print(f"Model saved to {model_path}")

# Evaluate the model
test_loss, test_mae, test_mse = model.evaluate(test_dataset)
print('Test loss:', test_loss)
print('Test MAE:', test_mae)
print('Test MSE:', test_mse)


#extracting X_test and making predictions

X_test = np.concatenate([x for x, y in test_dataset], axis=0)

# 
#model = tf.keras.models.load_model(resolved_model_path)
y_pred = model.predict(X_test)

y_pred_inverse = scaler.inverse_transform(y_pred)  # inverse the scaled data


print(y_pred.shape)
print(df1.shape)
print (len(df1.index))

# Flatten MultiIndex columns to plain bikeshop names.
# bike_sales_df.columns is MultiIndex ('total_price', 'Shop Name') because
# summarize_by_time pivots with value as first level; data_prep melt expects
# simple string column names, not tuples.
flat_cols = (
    df1.columns.get_level_values(-1)
    if isinstance(df1.columns, pd.MultiIndex)
    else df1.columns
)

pred_df = pd.DataFrame(y_pred_inverse, columns=flat_cols)

prediction_path = OUTPUT_DIR / "Multivariate_time_series_predictions"
pred_df.to_pickle(prediction_path)

# Use the last h=3 rows of pred_df as the LSTM forecast for the holdout window
lstm_df = pred_df.tail(3).reset_index(drop=True)


# ── Align LSTM predictions with the AutoARIMA holdout window ──────────────
# evaluate_arima_holdout trains on all-but-last-3 and predicts last 3 months,
# giving us Actuals and ARIMA predictions on the same dates.
# We index the LSTM test predictions (y_pred_inverse last 3 rows) to those
# same dates so compare_arima_lstm can join all three models cleanly.

from forecasting import evaluate_arima_holdout

h = 3
result_rows, _ = evaluate_arima_holdout(
    wide=bike_sales_df, h=h, sp=1, suppress_warnings=True
)
result_rows["order_date"] = pd.to_datetime(result_rows["order_date"].astype(str))

# Build long-format prediction_df expected by compare_arima_lstm
long_rows = []
for shop in bike_sales_df.columns:
    shop_str = str(shop)
    arima_s = result_rows[result_rows["series"] == shop_str].sort_values("order_date")
    if arima_s.empty:
        continue

    lstm_preds = lstm_df[shop].values if shop in lstm_df.columns else None

    for i, (_, row) in enumerate(arima_s.iterrows()):
        dt = row["order_date"]
        long_rows.append({"order_date": dt, "bikeshop_name": shop_str,
                          "variable": "Actuals",           "Sales": row["value"]})
        long_rows.append({"order_date": dt, "bikeshop_name": shop_str,
                          "variable": "Arima_prediction",  "Sales": row["prediction"]})
        if lstm_preds is not None and i < len(lstm_preds):
            long_rows.append({"order_date": dt, "bikeshop_name": shop_str,
                               "variable": "LSTM_prediction", "Sales": lstm_preds[i]})

prediction_df = pd.DataFrame(long_rows)

comparison_df = compare_arima_lstm(
    prediction_df=prediction_df,
    train_actuals=bike_sales_df,
    group="bikeshop_name",
    seasonal_period=1,
)

print(comparison_df)

comparison_path = OUTPUT_DIR / "arima_vs_lstm_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\nMean metrics by model (lower is better, MASE < 1 beats naive):")
print(comparison_df.groupby("model")[["MAE", "RMSE", "MAPE", "MASE"]].mean())










# %%
