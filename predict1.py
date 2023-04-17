import numpy as np
import tensorflow as tf
from tensorflow import keras
from operator import concat
from pickletools import optimize
from tabnanny import verbose
from this import d
from unicodedata import name
from importlib_metadata import metadata
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler,StandardScaler
import time

model=keras.models.load_model('./weight')


df=pd.read_csv('dataset/detect_data1.csv')

cols=list(df)[1:3]

datelist_train=list(df['DateTime'])

#print(datelist_train[-1])

dataset_train = df[cols].astype(str)

dataset_train = dataset_train.astype(float)

# Using multiple features (predictors)
training_set = dataset_train.values

sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)


#print(len(training_set_scaled)) #711
sc_predict = StandardScaler()
training_set_scaled=sc_predict.fit_transform(training_set[:, 0:2])


X_train = []
y_train = []

n_future =700  # Number of days we want top predict into the future
n_past = 1 # Number of past days we want to use to predict the future

# - n_future +1
for i in range(n_past, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])


X_train, y_train = np.array(X_train), np.array(y_train)


for i in range(0,n_future):
    
    
    datelist_future = pd.date_range(datelist_train[i], periods=n_future, freq='H').tolist()


datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())
    
#print(len(datelist_future))
        
predictions_future = model.predict(X_train[-n_future:])

y_pred_future = sc_predict.inverse_transform(predictions_future)

print(len(y_pred_future))

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Temperature','Humidity']).set_index(pd.Series(datelist_future))


print(PREDICTIONS_FUTURE.head(700))


#PREDICTIONS_FUTURE.to_csv('./out.csv')
