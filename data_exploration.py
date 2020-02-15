# -*- coding: utf-8 -*-
"""
@author: dimkonto
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as pp

"""
#Preprocessing/Manipulation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

#Linear Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Non-Linear Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Regression Linear
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#Regression Non-Linear
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#Validation/Test set config
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#Group a list of ML steps in a pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
#Ensemble methods to improve accuracy
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
#Find best parameters
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
#Ways to save and load
from pickle import dump
from pickle import load
#from sklearn.externals.joblib import dump #JOBLIB DEPRECATED
#from sklearn.externals.joblib import load

"""


"""
# DEEP LEARNING
import theano
from theano import tensor
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.constraints import maxnorm
from keras.optimizers import SGD

from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.models import model_from_json
import os
from keras.callbacks import ModelCheckpoint

from keras.callbacks import LearningRateScheduler

"""

path = r'D:\Datasets\hpc\hpc.csv'
dataset = pd.read_csv(path,header=0,infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

print(dataset.head())
print(dataset.dtypes)

#Data Correlation
print(dataset.corr(method='pearson'))
#Plot it
correlations = dataset.corr()
# plot correlation matrix
fig = pp.figure(figsize=(22, 22))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataset.columns)
ax.set_yticklabels(dataset.columns)
pp.savefig(r'D:\Datasets\hpc\charts\correlation.png',dpi=100,bbox_inches="tight")
pp.show()


#Histogram
#dataset.hist(figsize=(20,10))
#pp.savefig(r'D:\Datasets\hpc\charts\histogram.png',dpi=100,bbox_inches="tight")
#pp.show()

#Box Plots
#dataset.plot(kind='box',subplots=True,layout=(3,3),sharex=False, figsize=(20,10))
#pp.savefig(r'D:\Datasets\hpc\charts\boxplt.png',dpi=100,bbox_inches="tight")
#pp.show()

#Rescale (do that though only on input vars)
#scaler=MinMaxScaler(feature_range=(0,1))
#rescaled = scaler.fit_transform(dataset.values)
#print(rescaled)


#line plots for timeseries
#pp.figure(figsize=(20,20))

#for i in range(len(dataset.columns)):
#    pp.subplot(len(dataset.columns),1,i+1)
#    name = dataset.columns[i]
#    pp.plot(dataset[name])
#    pp.title(name,y=0)
#pp.savefig(r'D:\Datasets\hpc\charts\lineplts.png',dpi=100,bbox_inches="tight")    
#pp.show()

#line plots for timeseries (Study Active power each year to see any patterns patterns)
#years=['2007','2008','2009','2010']
#pp.figure(figsize=(20,20))

#for i in range(len(years)):
#    pp.subplot(len(years),1,i+1)
    #determine year
#    year=years[i]
#    result=dataset[str(year)]
    
#    pp.plot(result['Global_active_power'])
#    pp.title(str(year),y=0,loc='left')
#pp.savefig(r'D:\Datasets\hpc\charts\APperyear.png',dpi=100,bbox_inches="tight")
#pp.show()

#line plots for timeseries (Study Active power each month of a certain year)
#months=[x for x in range(1,13)]
#pp.figure(figsize=(30,30))

#for i in range(len(months)):
#    pp.subplot(len(months),1,i+1)
    #determine month
#    month = '2007-'+str(months[i])
#    result=dataset[month]
    
#    pp.plot(result['Global_active_power'])
#    pp.title(str(month),y=0,loc='left')
#pp.savefig(r'D:\Datasets\hpc\charts\APpermonth2007.png',dpi=100,bbox_inches="tight")
#pp.show()

#line plots for timeseries (Study Active power each day) (plots first 20 days of a month)
#days=[y for y in range(1,20)]
#pp.figure(figsize=(40,40))

#for i in range(len(days)):
#    pp.subplot(len(days),1,i+1)
    #determine day
#    day = '2007-01-'+str(days[i])
#    result=dataset[day]
    
#    pp.plot(result['Global_active_power'])
#    pp.title(str(day),y=0,loc='left')
#pp.savefig(r'D:\Datasets\hpc\charts\AP20days.png',dpi=100,bbox_inches="tight")
#pp.show()

# Find probability distribution of variables (columns) with histogram
#pp.figure(figsize=(20,20))

#for i in range(len(dataset.columns)):
#    pp.subplot(len(dataset.columns),1,i+1)
    #determine histogram bins
#    name=dataset.columns[i]
#    dataset[name].hist(bins=100)
    
#    pp.title(name,y=0)
#pp.savefig(r'D:\Datasets\hpc\charts\distribution.png',dpi=100,bbox_inches="tight")    
#pp.show()

#Investigate distribution of one column for each year of data
#years=['2007','2008','2009','2010']
#pp.figure(figsize=(20,20))

#for i in range(len(years)):
#    ax=pp.subplot(len(years),1,i+1)
    #determine year & set up bins for hstogram
#    year=years[i]
#    result=dataset[str(year)]
#    result['Global_active_power'].hist(bins=100) #for that column
#    ax.set_xlim(0,5) #zoom in
    #Title
#    pp.title(str(year),y=0,loc='right')
#pp.savefig(r'D:\Datasets\hpc\charts\APyearlydist.png',dpi=100,bbox_inches="tight")    
#pp.show()

#Investigate distribution for each month of the year
#months=[x for x in range(1,13)]
#pp.figure(figsize=(30,30))

#for i in range(len(months)):
#    ax=pp.subplot(len(months),1,i+1)
    #determine month
#    month = '2007-'+str(months[i])
#    result=dataset[month]
#    result['Global_active_power'].hist(bins=100)
#    ax.set_xlim(0,5)

#    pp.title(str(month),y=0,loc='right')
#pp.savefig(r'D:\Datasets\hpc\charts\APmonthlydist.png',dpi=100,bbox_inches="tight")    
#pp.show()
    


    