#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:14:21 2019

@author: yanndebain
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
dataSet = pd.read_csv('/Users/yanndebain/MEGA/MEGAsync/Code/Data Science/ML/Polynomial Regression/Position_Salaries.csv')

X = dataSet.iloc[:, 1:-1].values #independant variables
y = dataSet.iloc[:, -1].values #dependant variables

# Polynomial features
polyFeat = PolynomialFeatures(degree = 4)
X_poly = polyFeat.fit_transform(X)


#Training set and test set
X_plt, X_plt_t, y_plt, y_plt_t = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.2, random_state = 0)

# Building model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test model
y_pred = regressor.predict(X_test)
#regressor.predict([[15]])


# Data visualization
#plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_plt, y_plt, color = 'green')
plt.plot(X, regressor.predict(X_poly), color = 'blue')
plt.title('Position vs Experience')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()