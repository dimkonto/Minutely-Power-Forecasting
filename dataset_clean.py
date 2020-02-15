# -*- coding: utf-8 -*-
"""
@author: dimkonto
"""
import pandas as pd
import numpy as np

#load dataset
path = r'D:\Datasets\hpc\household_power_consumption.txt'
dataset = pd.read_csv(path,sep=';',header=0,low_memory=False,infer_datetime_format=True,parse_dates={'datetime':[0,1]},index_col=['datetime'])

#print shape and a few records
print(dataset.shape)
print(dataset.head())

#change missing values
dataset.replace('?',np.NaN, inplace=True)

#new col for rest of remaining house Wh
values= dataset.values.astype('float32')
dataset['sub_metering_4'] = (values[:,0]*1000/60)-(values[:,4]+values[:,5]+values[:,6])

#new col for global active energy
#values= dataset.values.astype('float32')
#dataset['Global_active_energy'] = (values[:,0]*1000/60)

#save as new csv
dataset.to_csv(r'D:\Datasets\hpc\hpc.csv')

print(dataset.shape)
print(dataset.head())