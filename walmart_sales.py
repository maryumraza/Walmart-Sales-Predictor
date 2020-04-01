# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:57:28 2020

@author: uni tech
"""


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



# Initializing datasets
train=pd.read_csv('train.csv')
test=pd.read_csv('test_walmart.csv')
features=pd.read_csv('features.csv')
stores=pd.read_csv('stores.csv')

# Mergign train and features datasets
df= pd.merge(features, train, on=['Store', 'Date', 'IsHoliday'], how='inner')

# One Hot Encoding categorical data
one_hot=pd.get_dummies(stores['Type'])
stores=stores.drop('Type', axis=1)
stores = stores.join(one_hot)




df = pd.merge(df, stores, on=['Store'], how='inner')

# Separating date, month, and year from Date
df['Date']=pd.to_datetime(df['Date'])
df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month
del df['Date']


holiday= pd.get_dummies(df['IsHoliday'])
df= df.drop('IsHoliday', axis=1)
df= df.join(holiday)


# Fixing null values in markdown with the help of imputer class
se= SimpleImputer()
markdown= pd.DataFrame(se.fit_transform(df[['MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']]),columns=['MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
df= df.drop(['MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1)

df = pd.concat([df,markdown], axis=1)


X = np.array(df.drop(columns='Weekly_Sales'))
y= np.array(df['Weekly_Sales']).reshape(-1,1)


# Normalizing inputs and outputs
scalar= preprocessing.MinMaxScaler()
X= scalar.fit_transform(X)
y= scalar.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Defining functions for regression
def linear_reg():
    clf= LinearRegression()
    return clf
    
    
def svm_reg():
    clf= SVR(kernel='rbf', degree=3, gamma='scale')
    return clf

def decision_tree():
    clf=DecisionTreeRegressor(criterion='mse',splitter='best')
    return clf

def random_forest():
    clf= RandomForestRegressor(n_estimators=5, criterion='mse')
    return clf

lr_ = linear_reg()
svm_ = svm_reg()
dt_ = decision_tree()
rf_ = random_forest()

models = [lr_ , dt_, svm_ , rf_]
for model in models:
    y_train = y_train.ravel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(score)
 


    


