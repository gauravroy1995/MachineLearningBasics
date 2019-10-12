# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:34:46 2019

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,2].values


#splitting dataset into training and testset
# =============================================================================
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X, y , test_size = 0.2 , random_state = 0)
# =============================================================================

#feature scaling -- means to make age and salary column values range same
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#fit linear regressor model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# fit poly model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =8)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#visualising linear model
plt.scatter(X,y,color ="red")
plt.plot(X,lin_reg.predict(X),color = "blue")
plt.title('linear')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualising poly reg
plt.scatter(X,y,color ="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = "blue")
plt.title('linear')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#predict salary of lin first nad then poly
lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
