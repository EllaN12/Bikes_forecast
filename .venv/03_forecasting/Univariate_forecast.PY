#data
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time

#Analysis 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Forecasting
#from sktime.forecasting.arima import AutoARIMA
#from tqdm import tqdm

#forecasting

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.optimizers import Nadam, Adam
import keras_tuner as kt

import pickle




#Forecasting using LSTM

df = collect_data()
total_bikes_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = False,
    #fillna = np.nan
)

total_bikes_sales_df.count()

# Define parameters
WINDOW_SIZE = 3
BATCH_SIZE = 2
shuffle_buffer_size = 4

df1 = total_bikes_sales_df


# instantiate MinMaxScaler
scaler = MinMaxScaler()

#scale data
scaled_values = scaler.fit_transform(df1.values.reshape(-1,1))

#cobvert to dataframe
df2 = pd.DataFrame(scaled_values, index= df1.index, columns=["scaled_total_price"])
df2 


# Split the data into training and validation and test sets

train_size = int(len(df2) * 0.8)
dev_size = int(len(df2) * 0.1)

train_df = df2.iloc[:train_size]
dev_df = df2.iloc[train_size:train_size + dev_size]
test_df = df2.iloc[train_size + dev_size:]


# funciton to create X anf y
def create_X_y(df, WINDOW_SIZE):
    X = []
    y = []
    for i in range(len(df) - WINDOW_SIZE):
        X.append(df['scaled_total_price'].values[i:i+WINDOW_SIZE])
        y.append(df['scaled_total_price'].values[i+WINDOW_SIZE])
    return np.array(X), np.array(y)


X_train, y_train = create_X_y(train_df, WINDOW_SIZE)
X_val, y_val = create_X_y(dev_df, WINDOW_SIZE)
X_test, y_test = create_X_y(test_df, WINDOW_SIZE)




# Creating tensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# build the model with the optimal hyperparameters and train it on the data
def build_model(hp):
    
    model = tf.keras.models.Sequential()
    
    # building the model input layer
    model.add(tf.keras.layers.Input( shape = (WINDOW_SIZE, 1)))
        
    
    # building the LSTM Layer
    for i in range(hp.Int('num_layers', 1, 2)):
        model.add(tf.keras.layers.LSTM(
                    units=hp.Int(f'units_{i}', min_value=16, max_value=64, step=16),
        activation= hp.Choice(f'lstm_activation_{i}', values =  ['tanh', 'relu']),
        return_sequences = i < hp.Int('num_layers', 1,2) - 1,
        recurrent_activation= hp.Choice(f'recurrent_activation_{i}', values = ['sigmoid', 'tanh']) 
        ))
            
        if hp.Boolean(f'dropout_{i}'):
            model.add(tf.keras.layers.Dropout(
                rate= hp.Float(f'dropout_rate_{i}', 0.0, 0.5, step=0.1)
            ))
            

# Build the dense layer
    for i in range(hp.Int('num_dense_layers', 0, 2)):
        model.add(tf.keras.layers.Dense(
            units= hp.Int(f'dense_units_{i}', min_value=8, max_value=32, step=8),
            activation = hp.Choice(f'dense_activation_{i}', values = ['relu', 'tanh'])
            ))
    
# output layer
    model.add(tf.keras.layers.Dense(units = 1))

# OPtimzer selection

    optimizer = hp.Choice('optimizer', values = ['adam', 'sgd',  'nadam'])


    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate= hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
        
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate= hp.Float('learning_rate', 1e-4, 1e-1, sampling='log'),
            momentum= hp.Float('momentum', 0.0, 0.99))
    
    elif optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate= hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
        

# Compile the model
    model.compile(
        loss= hp.Choice('loss', values = ['huber', 'mse', 'mae']),
        optimizer= optimizer,
        metrics=["mae", "mse"])
    
    return model


tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory="/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/hyperband",
    project_name="univariate_time_series_tuning",
    overwrite=True
    )

# Perform hyperparameter search
tuner.search(train_dataset, epochs=30, validation_data= dev_dataset)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]



model = tuner.hypermodel.build(/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/best_hyperparameters.pkl)
history = model.fit(train_dataset, epochs=50, validation_data= dev_dataset, verbose=1)


#load the best hyperparameters
file_path = "/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/best_hyperparameters.pkl"
with open(file_path, 'wb') as f:
    pickle.dump(best_hps, f)
    
# loadin the best hyperparameters

with open(file_path, 'rb') as f:
    hyp = pickle.load(f)



model.save ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/univariate_time_series_model.h5", overwrite = True)
print(model.summary())
# Evaluate the model
test_loss, test_mae, test_mse = model.evaluate(test_dataset)
print('Validation loss:', test_loss)
print('Validation MAE:', test_mae)
print('Validation MSE:', test_mse)

# Print the best hyperparameters
print("\nBest hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")





#estracting X_test and making predictions

test_data = (X_test, y_test)

#X_test = np.concatenate([x for x, _ in test_data], axis=0)
X_test = test_data[0]
y_pred = model.predict(X_test)

y_test.shape

y_pred.shape


actual_df = df1.copy()  # copy the original data
#y_pred_inverse = scaler.inverse_transform(y_pred)  # inverse the scaled data



print(y_pred.shape)
print(df1.shape)
print (len(df1.index))


# inverse the scaled data both actual and predictions
#actual_inverse = scaler.inverse_transform(df1.values)
pred_inverse = scaler.inverse_transform(y_pred)


# create a dataframe with the actual values
actual_df = df1.copy()

univariate_prediction_df = pd.DataFrame(
    pred_inverse,
    columns = df1.columns
)

univariate_prediction_df.astype(int)

univariate_prediction_df.to_pickle("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Univariate_predictions.pkl")



# to rerun a model 

model = tf.keras.models.load_model("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/univariate_time_series_model.h5")

print (model.summary())

test_data = (X_test, y_test)

#X_test = np.concatenate([x for x, _ in test_data], axis=0)
X_test = test_data[0]
y_pred = model.predict(X_test)

pred_inverse = scaler.inverse_transform(y_pred)


# create a dataframe with the actual values
actual_df = df1.copy()

univariate_prediction_df = pd.DataFrame(
    pred_inverse,
    columns = df1.columns
)


def univariate_forecasting: 
    