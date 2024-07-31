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
from tensorflow.keras.optimizers import Nadam
import keras_tuner as kt

#Forecasting using LSTM

df = collect_data()
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


tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=50,
    factor=3,
    directory="/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/hyperband",
    project_name="bikeshop_lstm_time_series_tuning",
    overwrite=True
    )

# Perform hyperparameter search
tuner.search(train_dataset, epochs=30, validation_data= dev_dataset)


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset, epochs=50, validation_data= dev_dataset, verbose=1)
model.save ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/bikeshop_time_series_model.h5", overwrite = True)
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
#model = tf.keras.models.load_model("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Multivariate_time_series_model.h5")
y_pred = model.predict(X_test)


y_pred.shape


actual_df = df1.copy()  # copy the original data
y_pred_inverse = scaler.inverse_transform(y_pred)  # inverse the scaled data



print(y_pred.shape)
print(df1.shape)
print (len(df1.index))


#create a dataframe with the predictions
pred_df = pd.DataFrame(
    y_pred,
    columns = df1.columns
)


# inverse the scaled data both actual and predictions
#actual_inverse = scaler.inverse_transform(df1.values)
pred_inverse = scaler.inverse_transform(pred_df.values)


# create a dataframe with the actual values
actual_df = df1.copy()

pred_df = pd.DataFrame(
    y_pred_inverse,
    columns = df1.columns,
    
)

pred_df.to_pickle("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Multivariate_time_series_predictions.pkl")



def LSTM_forecast(group, WINDOW_SIZE, BATCH_SIZE, shuffle_buffer_size, train_rate, dev_rate):
    df = collect_data()
    
    if group == 'category_1' or group == 'category_2':
        cat_1 = summarize_by_time(
        data = df,
        date_column = "order_date",
        groups= 'category_1',
        value_column = "total_price",
        rule = "M",
        kind = "period",
        agg_func = np.sum,
        wide_format = True)
        
        cat_2 = summarize_by_time(
        data = df,
        date_column = "order_date",
        groups= 'category_2',
        value_column = "total_price",
        rule = "M",
        kind = "period",
        agg_func = np.sum,
        wide_format = True)
        
        df1 = cat_1.merge(cat_2, on = 'order_date', how = 'outer')
    else:
        df1 = summarize_by_time(
        data = df,
        date_column = "order_date",
        groups= group,
        value_column = "total_price",
        rule = "M",
        kind = "period",
        agg_func = np.sum,
        wide_format = True)
    
    
    # Define parameters
    n_features = len(df1.columns) # number of features/category of bikes
    
   
    #scale data
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df1.values)
    
    #create a dataframe with the scaled values
    df2 = pd.DataFrame(scaled_values, index= df1.index, columns=df1.columns)
    
    
    # split the data into train and val and test

    train_size = int(len(df2) * train_rate)
    dev_size = int(len(df2) * dev_rate)

    train_df = df2.iloc[:train_size]
    dev_df = df2.iloc[train_size:train_size + dev_size]
    test_df = df2.iloc[train_size + dev_size:]
    
    
   
    # function to create X and y
    def create_X_y(df, WINDOW_SIZE):
        X = []
        y = []
        for i in range(len(df) - WINDOW_SIZE):
            X.append(df.values[i:i+WINDOW_SIZE])
            y.append(df.values[i+WINDOW_SIZE])
        return np.array(X), np.array(y)

    # create X and y for train, val and test datasets

    X_train, y_train = create_X_y(train_df, WINDOW_SIZE)
    X_val, y_val = create_X_y(dev_df, WINDOW_SIZE)
    X_test, y_test = create_X_y(test_df, WINDOW_SIZE)
    

    # Creating tensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
    dev_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    
    
    # Import the model
    if group == 'category_1' or group == 'category_2':
        model_path = ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/Multivariate_time_series_model.h5")
    elif group == 'bikeshop_name':
        model_path = ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/bikeshop_time_series_model.h5")
    else:
        model_path = ("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/univariate_time_series_model.h5")        
      
    model = tf.keras.models.load_model(model_path)
    
    
    # make predictions
    X_test = np.concatenate([x for x, y in test_dataset], axis=0)
    y_pred = model.predict(X_test)
        
    # create a dataframe with the predictions
    pred_inverse = scaler.inverse_transform(y_pred)
    LSTM_prediction_df = pd.DataFrame(
    pred_inverse,
    #index=test_df.index[-len(pred_inverse):],
    columns = df1.columns)

    return LSTM_prediction_df





multivariate_forecast('category_2', 3, 9, 90, 0.8, 0.1)




#forecasting by Bikeshop name
df = collect_data()
bikeshops_sales_df = summarize_by_time(
    data = df,
    date_column = "order_date",
    groups= 'bikeshop_name',
    value_column = "total_price",
    rule = "M",
    kind = "period",
    agg_func = np.sum,
    wide_format = True,
    #fillna = np.nan
)
df1 = bikeshops_sales_df.copy()

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

# function to create X and y
def create_X_y(df, WINDOW_SIZE):
    X = []
    y = []
    for i in range(len(df) - WINDOW_SIZE):
        X.append(df.values[i:i+WINDOW_SIZE])
        y.append(df.values[i+WINDOW_SIZE])
    return np.array(X), np.array(y)

# create X and y for train, val and test datasets

X_train, y_train = create_X_y(train_df, WINDOW_SIZE)
X_val, y_val = create_X_y(dev_df, WINDOW_SIZE)
X_test, y_test = create_X_y(test_df, WINDOW_SIZE)


# Creating tensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).shuffle(shuffle_buffer_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)


# Import the model
    
model = tf.keras.models.load_model("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/bikeshop_time_series_model.h5")

# make predictions
#X_test = np.concatenate([x for x, y in test_dataset], axis=0)
#y_pred = model.predict(test_dataset)
y_pred = model.predict(test_dataset.map(lambda x, y: x))

# create a dataframe with the predictions
pred_inverse = scaler.inverse_transform(y_pred)
bikeshop_prediction_df = pd.DataFrame(
pred_inverse,
columns = df1.columns)

bikeshop_prediction_df

bikeshop_prediction_df.to_pickle("/Users/ellandalla/Desktop/Bike_Sales_Forecasting/venv/03_artifacts/bikeshops_predictions.pkl")