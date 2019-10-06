

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:49:39 2019

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#splitting dataset into training and testset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y , test_size = 1/3 , random_state = 0)

#feature scaling -- means to make age and salary column values range same
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)￼
X_test = sc_X.transform(X_test)"""

#fit slregrr in train model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting test set results
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, y_train , color = "red")
plt.plot(X_train , regressor.predict(X_train) , color = "blue")
plt.title('Sal vs Exp(Train set)')
plt.xlabel('yrs of exp')
plt.ylabel('salary')
plt.show()

#visualising the test set results
plt.scatter(X_test, y_test , color = "red")
plt.plot(X_train , regressor.predict(X_train) , color = "blue")
plt.title('Sal vs Exp(Test set)')
plt.xlabel('yrs of exp')
plt.ylabel('salary')
plt.show()