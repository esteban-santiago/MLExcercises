#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:53:36 2022

@author: esantiago
"""
#When the relation is not linear
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

np.random.seed(42)

n_samples = 100

X = np.linspace(0, 10, 100)

rng = np.random.randn(n_samples) * 100

y = X ** 3 + rng + 100

plt.Figure(figsize=(10,8))
plt.scatter(X.reshape(-1,1), y)


#Linear Regression
lr = LinearRegression()
lr.fit(X.reshape(-1,1), y)
model_pred = lr.predict(X.reshape(-1,1))

plt.figure(figsize=(10,8))
plt.scatter(X, y)
plt.plot(X, model_pred)
print(r2_score(y, model_pred))

#Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X.reshape(-1,1))
lr2 = LinearRegression()
lr2.fit(X_poly, y.reshape(-1,1))
y_pred = lr2.predict(X_poly)

plt.figure(figsize=(10,8))
plt.scatter(X, y)
plt.plot(X, y_pred)
print(r2_score(y, y_pred))
