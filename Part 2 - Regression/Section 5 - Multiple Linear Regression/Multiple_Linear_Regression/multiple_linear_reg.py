# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:22:30 2019

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values


#encoding categoryical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#avoiding the dummy variable trap
X = X[:,1:]

#splitting dataset into training and testset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y , test_size = 0.2 , random_state = 0)

#feature scaling -- means to make age and salary column values range same
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#fitting multiple linear regression in training set
from sklearn.linear_model import LinearRegression
linearregressor = LinearRegression()
linearregressor.fit(X_train,y_train)


#predicting values of train
y_predict = linearregressor.predict(X_test)

#building optimal model using backward elimination and used r formula
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int) ,values = X , axis =1)
X_Opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.ols(formula = 'y ~ X_Opt', data = dataset).fit() #ordinary least square
regressor_OLS.summary()
X_Opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.ols(formula = 'y ~ X_Opt', data = dataset).fit() #ordinary least square
regressor_OLS.summary()
X_Opt = X[:,[0,3,4,5]]
regressor_OLS = sm.ols(formula = 'y ~ X_Opt', data = dataset).fit() #ordinary least square
regressor_OLS.summary()
X_Opt = X[:,[0,3,5]]
regressor_OLS = sm.ols(formula = 'y ~ X_Opt', data = dataset).fit() #ordinary least square
regressor_OLS.summary()
X_Opt = X[:,[0,3]]
regressor_OLS = sm.ols(formula = 'y ~ X_Opt', data = dataset).fit() #ordinary least square
regressor_OLS.summary()


# this is how im testing my model with xopt 
X_train1,X_test1,y_train1,y_test1 = train_test_split(X_Opt, y , test_size = 0.2 , random_state = 0)

linearregressor.fit(X_train1,y_train1)

y_predict1 = linearregressor.predict(X_test1)