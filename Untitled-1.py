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
######################torch
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import transforms, datasets
############################# keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM,Activation,Dropout



df=pd.read_csv("dataset/weatherHistory.csv")


print(df.head())
print(df.isnull().sum())



def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    
    n_vars=1 if type(data) is list else data.shape[1]
    df=pd.DataFrame(data)
    cols,names=list(),list()
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)] 
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('var%d(t)' % (j+1)) for j in range(n_vars)] 
        else:
            names+=[('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)] 
    agg=pd.concat(cols,axis=1)
    agg.columns=names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

df=df.set_index(pd.DatetimeIndex(df['Date']))

df=df.drop('Date',axis=1)

df.index.name='date'

print(df.head())

data=df.values

print(df.Temperature.min(),df.Temperature.max())

data=data.astype('float32')
scaler=MinMaxScaler(feature_range=(-1,1))
scaled=scaler.fit_transform(data)

n_hours=3
n_feature=2

# as at 6th no. from last is temp. our o/p var
reframed=series_to_supervised(scaled,n_hours,1)
print(reframed.shape)
print(reframed)

values=reframed.values

n_train_hours=365*24

train=values[:n_train_hours,:]
test=values[n_train_hours:,:]

n_obs=n_hours*n_feature

trainX,trainY=train[:,:n_obs],train[:,0]
testX,testY=test[:,:n_obs],test[:,0]
print(trainY)
#print(testY)

trainX=trainX.reshape((trainX.shape[0],n_hours,n_feature))
testX=testX.reshape((testX.shape[0],n_hours,n_feature))
print(trainX.shape)


#############################LSTM Model


hidden_nodes=int((trainX.shape[1]*trainX.shape[2]))

model=Sequential()
model.add(LSTM(30,input_shape=(trainX.shape[1],trainX.shape[2])))## input_shape=(no. of i/p, dimension), result=result=(1,50)
model.add(Dense(256,name='FC1'))
model.add(Activation("relu"))
model.add(Dropout(0.2))
#model.add(Dense(128,name='FC2'))
#model.add(Activation("relu"))
model.add(Dense(1,name="out_layer"))
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=tf.keras.metrics.MeanSquaredError())


print(model.summary())



history=model.fit(trainX,trainY,epochs=100,batch_size=128,validation_data=(testX,testY),verbose=1,shuffle=False)
model.save('LSTM_model/LSTM.h5')
model.save_weights("LSTM_model/LSTM.weight")
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
#plt.show()

# make a prediction
testY_predicted=model.predict(testX)
testX=testX.reshape((testX.shape[0],n_hours*n_feature))

print(testY_predicted)

print(testY)



plt.plot(testY[:100],label='Actual')
plt.plot(testY_predicted[:100],label="Predicted")

plt.legend()
plt.show()
plt.savefig("LSTM/one_hidden_layer.png",dpi='figure',format=None,metadata=None,bbox_inches=None,pad_inches=0.1)
















model = Sequential()

# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))

# Adding 2nd LSTM layer
model.add(LSTM(units=10, return_sequences=False))

# Adding Dropout
model.add(Dropout(0.25))

# Output layer
model.add(Dense(units=1, activation='linear'))

# Compiling the Neural Network
model.compile(optimizer = Adam(learning_rate=0.001), loss='mean_squared_error')


history = model.fit(X_train, y_train, shuffle=True, epochs=100, validation_split=0.2, verbose=1, batch_size=32)


model.save('LSTM.H5')