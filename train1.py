import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import datetime as dt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler,StandardScaler



import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import LSTM,Activation,Dropout
from tensorflow.keras.optimizers import Adam

df=pd.read_csv('dataset/detect_data1.csv')

cols=list(df)[1:3]

datelist_train=list(df['DateTime'])


print('Training set shape == {}'.format(df.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print('Featured selected: {}'.format(cols))

dataset_train = df[cols].astype(str)


dataset_train = dataset_train.astype(float)

# Using multiple features (predictors)
training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))

print(training_set)


sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
training_set_scaled=sc_predict.fit_transform(training_set[:,0:2])

print("training_set_scaled",training_set_scaled)


print(dataset_train.shape)

X_train = []
y_train = []


n_future = 700 # Number of days we want top predict into the future
n_past = 1   # one per hour to predict 

for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))

print(len(dataset_train))


model = Sequential()

# Adding 1st LSTM layer
model.add(LSTM(units=128, return_sequences=True, input_shape=(n_past, dataset_train.shape[1])))

# Adding 2nd LSTM layer
model.add(LSTM(units=20, return_sequences=False))

# Adding Dropout
model.add(Dropout(0.2))



# Output layer
model.add(Dense(units=2, activation='linear'))

# Compiling the Neural Network
model.compile(optimizer = Adam(learning_rate=0.001), loss='mean_squared_error')


model.summary()

#history = model.fit(X_train, y_train, shuffle=True, epochs=200, validation_split=0.2, verbose=2, batch_size=256)


#model.save("./weight")


