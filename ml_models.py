# -*- coding: utf-8 -*-
"""
@author: dimkonto
"""


from math import sqrt
import pandas as pd
import numpy as np
from matplotlib import pyplot as pp
from sklearn import preprocessing as prc
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


path = r'D:\Datasets\hpc\hpc.csv'
dataset = pd.read_csv(path,header=0,infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

#print(dataset.head())

#Convert dataset to supervised learning (lagged observations by shifting)

#dataset['active_power-1']=dataset['Global_active_power'].shift(1)
#print(dataset.head())

#Numpy 2D array to supervised dataframe
#data: values of a pandas dataframe
#n_in: input columns
#n_out: output columns
#sampling: parameter that changes the sampling of columns (defaults to 1: the original dataset sampling)
def series_to_supervised(data,n_in=1,n_out=1,sampling=1,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols,names = list(), list()
    
    for i in range(n_in,0,-1):
        cols.append(df.shift(i*sampling))
        names+=[('var%d(t-%d)'%(j+1,i))for j in range(n_vars)]
    for i in range(0,n_out):
        cols.append(df.shift(-i*sampling))
        if i==0:
            names += [('var%d(t)'%(j+1))for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)'%(j+1,i))for j in range(n_vars)]    
    agg=pd.concat(cols,axis=1)
    agg.columns=names
    
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#All columns to values (all features used)
#values = dataset.values
#print(values)

#Selected Features to be used
#values = dataset[['Global_active_power','Global_intensity']].values
#print(values)

#For one feature to be used    
values = dataset['Global_active_power'].values
values = values.reshape(-1,1)
print(values)

#Prepare data (Scale)
scaler = prc.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

print(scaled)

#For all features (scaled should be used as data in function). 
reframed = series_to_supervised(scaled,2,1,1)
#Drop Unnecessary Columns (For Multiple Features)
#reframed.drop(reframed.columns[[5]],axis=1,inplace=True)

print(reframed.head())
print(reframed.shape)

#Fit Model (Train & Test) Split
values = reframed.values
n_train_minutes = 3*365*24*60
train = values[:n_train_minutes,:]
test = values[n_train_minutes:,:]
#print(train[:,-1])

#Separate input X and output Y for train and test
train_X, train_y = train[:,:-1], train[:,-1] #input all t-1 columns, output the t current column
test_X, test_y = test[:,:-1], test[:,-1]
#LSTMs and CNN want 3D format of data input
#train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
#test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#Design Baseline LSTM
def baseline_LSTM():
    nn = Sequential()
    #First Layer
    nn.add(LSTM(100,input_shape=(train_X.shape[1], train_X.shape[2])))
    #Output Layer
    nn.add(Dense(1))
    #Compile, choose loss and optimizer metrics
    nn.compile(loss='mae',optimizer='adam')
    return nn

#Design Stacked LSTM
def stacked_LSTM():
    nn = Sequential()
    nn.add(LSTM(100,activation='relu',return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    nn.add(LSTM(100,activation='relu'))
#    nn.add(Dropout(rate=0.2))
    nn.add(Dense(1))
    nn.compile(loss='mae',optimizer='adam')
    return nn

#Design Bidirectional LSTM
def bidirectional_LSTM():
    nn = Sequential()
    #First Layer
    nn.add(Bidirectional(LSTM(100,activation='relu'),input_shape=(train_X.shape[1], train_X.shape[2])))
    #Output Layer
    nn.add(Dense(1))
    #Compile, choose loss and optimizer metrics
    nn.compile(loss='mae',optimizer='adam')
    return nn

#Design Baseline CNN
def baseline_CNN():
    nn = Sequential()
    nn.add(Conv1D(filters=64, kernel_size=2,padding='same', activation='relu',input_shape=(train_X.shape[1], train_X.shape[2])))
    nn.add(MaxPooling1D(pool_size=2,padding='same'))
    nn.add(Flatten())
    nn.add(Dense(100,activation='relu'))
    nn.add(Dense(1))
    nn.compile(loss='mae',optimizer='adam')
    return nn

def baseline_MLP():
    nn=Sequential()
    nn.add(Dense(100,activation='relu',input_dim=2)) #2 time steps
    nn.add(Dense(1))
    nn.compile(loss='mae',optimizer='adam')
    return nn
    

#Build Baseline_LSTM
#nn=baseline_LSTM()

#Build Stacked LSTM
#nn=stacked_LSTM()
    
#Build Bidirectional LSTM
#nn=bidirectional_LSTM()
    
#Build Baseline CNN
#nn=baseline_CNN()

#Build Baseline MLP
nn=baseline_MLP()

#Fit Baseline_LSTM
history=nn.fit(train_X,train_y, epochs=8, batch_size=72, validation_data=(test_X,test_y), verbose=2, shuffle=False)

#Save Baseline Model
#nn.save(r'D:\Datasets\hpc\hpcLSTM.h5')
#nn.save(r'D:\Datasets\hpc\hpcLSTM_v2.h5')


#Save Stacked Model
#nn.save(r'D:\Datasets\hpc\hpcLSTM_Stacked.h5')

#Save Bidirectional Model
#nn.save(r'D:\Datasets\hpc\hpcLSTM_BD.h5')

#Save Baseline CNN
#nn.save(r'D:\Datasets\hpc\hpc_CNN.h5')

#Save Baseline MLP
nn.save(r'D:\Datasets\hpc\hpc_MLP.h5')
    
#Load Model(Only for fast experiments)
#nn=load_model(r'D:\Datasets\hpc\hpcLSTM.h5')
#nn=load_model(r'D:\Datasets\hpc\hpcLSTM_v2.h5')
#nn=load_model(r'D:\Datasets\hpc\hpc_CNN.h5')
#nn=load_model(r'D:\Datasets\hpc\hpc_MLP.h5')     

#Estimate Performance (extra)
trainScore = nn.evaluate(train_X, train_y, verbose=0)
print(trainScore)
testScore = nn.evaluate(test_X, test_y, verbose=0)
print(testScore)

#Make Predictions (ext)
trainPredict = nn.predict(train_X)
testPredict = nn.predict(test_X)
print(trainPredict.shape)
print(testPredict.shape)

#invert predictions (ext) 
trainPredict = scaler.inverse_transform(trainPredict) #1 column many rows
train_y = scaler.inverse_transform([train_y]) #1 row many columns 
testPredict = scaler.inverse_transform(testPredict)
test_y = scaler.inverse_transform([test_y])

#print(train_y[0,:])
print(trainPredict.shape)

# plot baseline and predictions ON TRAIN
pp.plot(trainPredict[:400,0],color='blue',label='Predicted')
pp.plot(train_y[0,:400],color='red', label='Train')
pp.legend()
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTrain.png',dpi=100,bbox_inches="tight")
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTrain_V2.png',dpi=100,bbox_inches="tight")
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTrain_Stacked.png',dpi=100,bbox_inches="tight") 
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTrain_BD.png',dpi=100,bbox_inches="tight")
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTrain_CNN.png',dpi=100,bbox_inches="tight")
pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTrain_MLP.png',dpi=100,bbox_inches="tight")
pp.show()

# plot baseline and predictions ON TEST
pp.plot(testPredict[:400,0],color='blue',label='Predicted')
pp.plot(test_y[0,:400],color='red', label='Test')
pp.legend()
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTest.png',dpi=100,bbox_inches="tight")
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTest_V2.png',dpi=100,bbox_inches="tight")  
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTest_Stacked.png',dpi=100,bbox_inches="tight")
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTest_BD.png',dpi=100,bbox_inches="tight") 
#pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTest_CNN.png',dpi=100,bbox_inches="tight") 
pp.savefig(r'D:\Datasets\hpc\charts\PredictionOnTest_MLP.png',dpi=100,bbox_inches="tight") 
pp.show()

#Plot Loss
pp.plot(history.history['loss'], label='train')
pp.plot(history.history['val_loss'], label='test')
pp.legend()
#pp.savefig(r'D:\Datasets\hpc\charts\LossPlot.png',dpi=100,bbox_inches="tight") 
#pp.savefig(r'D:\Datasets\hpc\charts\LossPlot_V2.png',dpi=100,bbox_inches="tight")
#pp.savefig(r'D:\Datasets\hpc\charts\LossPlot_Stacked.png',dpi=100,bbox_inches="tight") 
#pp.savefig(r'D:\Datasets\hpc\charts\LossPlot_BD.png',dpi=100,bbox_inches="tight")
#pp.savefig(r'D:\Datasets\hpc\charts\LossPlot_CNN.png',dpi=100,bbox_inches="tight")
pp.savefig(r'D:\Datasets\hpc\charts\LossPlot_MLP.png',dpi=100,bbox_inches="tight")
pp.show()




#values1 = [x for x in range(10)]
#Pick column or columns from dataset if you don't want to include them in the original dataframe
#values2lst= dataset['Global_active_power'].tolist()
#data = series_to_supervised(values2lst,3)
#print(data)
    
