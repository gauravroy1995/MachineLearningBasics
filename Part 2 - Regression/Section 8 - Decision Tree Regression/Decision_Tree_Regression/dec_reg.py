# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:18:32 2019

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



#fit Decision tree model regressor model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)



#visualising linear model
plt.scatter(X,y,color ="red")
plt.plot(X,regressor.predict(X),color = "blue")
plt.title('Dec tree')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualising in actuall higher resolution
X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color ="red")
plt.plot(X_grid,regressor.predict(X_grid),color = "blue")
plt.title('Dec tree')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()




#predict salary by inversing the  feature scaling values
regressor.predict(sc_X.transform([[6.5]])) #try this and you will get the transformed value, so now in nextstep we will extract the value
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

