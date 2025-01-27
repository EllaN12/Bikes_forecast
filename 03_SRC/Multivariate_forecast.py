
#%%
import os
from my_pandas_extensions.database import collect_data
from my_pandas_extensions.database import summarize_by_time
from my_pandas_extensions.forecasting import data_prep 
import pickle
import h5py

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
from tensorflow.keras.optimizers import Nadam
import keras_tuner as kt

#Forecasting using LSTM BY Category

df = collect_data()
df.to_csv('04_artifacts/data.csv')


Cat_1_bikes_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    groups= 'category_1',
    value_column = "total_price",
    rule = "M",
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
    rule = "M",
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
    rule = "M",
    kind = 'preiod',
    agg_func = np.sum,
    wide_format = True

)
df1 = bike_sales_df 

# Define parameters
n_features = len(df1.columns) # number of features/category of bikes

WINDOW_SIZE = 3
BATCH_SIZE = 9
shuffle_buffer_size = 90


# instantiate MinMaxScaler
scaler = MinMaxScaler()

#scale data
scaled_values = scaler.fit_transform(df1.values)

df2 = pd.DataFrame(scaled_values, index= df1.index, columns=df1.columns)
df2 


# split the data into train and val and test

train_size = int(len(df2) * 0.8)
dev_size = int(len(df2) * 0.1)

train_df = df2.iloc[:train_size]
dev_df = df2.iloc[train_size:train_size + dev_size]
test_df = df2.iloc[train_size + dev_size:]


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
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)


# Define the model architecture

def build_model(hp):
    
    model = tf.keras.models.Sequential()
    
    # building the model input layer
    model.add(tf.keras.layers.Input(
        shape= (WINDOW_SIZE, n_features)
    ))
        
    
    # building the LSTM Layer
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(tf.keras.layers.LSTM(
                    units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
        activation= hp.Choice(f'lstm_activation_{i}', values =  ['tanh', 'relu']),
        return_sequences = i < hp.Int('num_layers', 1,3) - 1,
        recurrent_activation= hp.Choice(f'recurrent_activation_{i}', values = ['sigmoid', 'tanh']) 
        ))
            
        if hp.Boolean(f'dropout_{i}'):
            model.add(tf.keras.layers.Dropout(
                rate= hp.Float(f'dropout_rate_{i}', 0.0, 0.5, step=0.1)
            ))
            

# Build the dense layer
    for i in range(hp.Int('num_dense_layers', 0, 2)):
        model.add(tf.keras.layers.Dense(
            units= hp.Int(f'dense_units_{i}', min_value=16, max_value=128, step=16),
            activation = hp.Choice(f'dense_activation_{i}', values = ['relu', 'tanh'])
            ))
    
# output layer
    model.add(tf.keras.layers.Dense(units = n_features))

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




print("Current working directory:", os.getcwd())
tuner_path = '04_artifacts/hyperband'
resolved_tuner_path = os.path.abspath(tuner_path)
print("Resolved path to hyperband and model files:", tuner_path)


tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory= resolved_tuner_path,
    project_name="lstm_time_series_tuning",
    overwrite=True
    )

# Perform hyperparameter search
tuner.search(train_dataset, epochs=30, validation_data= dev_dataset)


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, epochs=50, validation_data= dev_dataset, verbose=1)

print("Current working directory:", os.getcwd())

model_path = "04_artifacts/time_series_model.h5"
model.save (model_path, overwrite = True)
resolved_model_path = os.path.abspath(model_path)
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


#extracting X_test and making predictions

X_test = np.concatenate([x for x, y in test_dataset], axis=0)

# 
#model = tf.keras.models.load_model(resolved_model_path)
y_pred = model.predict(X_test)

y_pred_inverse = scaler.inverse_transform(y_pred)  # inverse the scaled data


print(y_pred.shape)
print(df1.shape)
print (len(df1.index))


#create a dataframe with the predictions
pred_df = pd.DataFrame(
    y_pred_inverse,
    columns = df1.columns
)


pred_df = pd.DataFrame(
    y_pred_inverse,
    columns = df1.columns,
    
)

prediction_path = '04_artifacts/Multivariate_time_series_predictions'
resolved_prediction_path = os.path.abspath(prediction_path)
pred_df.to_pickle(resolved_prediction_path)








bikeshop_predictions = '04_artifacts/bikeshop_prediction.pkl'
resolved_multi_var_path = os.path.abspath(bikeshop_predictions)


with open(resolved_multi_var_path, 'rb') as f:
    bikeshop_predictions = pickle.load(f)



lstm_df = bikeshop_predictions[bikeshop_predictions['variable'] == 'LSTM_prediction']

lstm_df = lstm_df.dropna(subset=['Sales'])

lstm_df.drop(columns=['ci_lower', 'ci_upper',	'variable' ])

lstm_df = lstm_df.pivot(
    columns = 'bikeshop_name',
    values = 'Sales',
    index = 'order_date'

)\
    .reset_index()\
        .drop(columns = ['order_date'])


df = collect_data()

bike_shop_predictions = data_prep(
    data = df,
    group = 'bikeshop_name',
    h = 3,
    LSTM_df= lstm_df
)









"""""

shops_predict = '04_artifacts/bishops_time_series_predictions.pkl'
resolved_state_var_path = os.path.abspath(shops_predict)

with open(resolved_state_var_path, 'rb') as f:
        shops_predictions = pickle.load(f)
        print()

"""



