# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:03:55 2019

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

#feature scaling -- means to make age and salary column values range same or in this case the companies postion and salaries
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_Y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_Y.fit_transform(np.reshape(y,(10,1)))
# =============================================================================



#fit Random forest tree model regressor model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(X,y)



#to compute value 
y_pred = regressor.predict([[6.5]])

#visualising in actuall higher resolution
X_grid = np.arange(min(X) , max(X) , 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color ="red")
plt.plot(X_grid,regressor.predict(X_grid),color = "blue")
plt.title('Forest tree')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()