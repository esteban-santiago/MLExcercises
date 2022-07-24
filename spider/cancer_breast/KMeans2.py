#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:16:36 2022

@author: esantiago
"""
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


df = load_iris()
X = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
X.drop(labels=['sepal width (cm)', 'sepal length (cm)'], axis=1, inplace=True)
y = pd.DataFrame(load_iris().target, columns=['target'])

model = KMeans(n_clusters=3)
y_pred = model.fit_predict(X,y)

print(y_pred)
#print(accuracy_score(y, y_pred))
#print('Inertia: ', model.inertia_)


plt.scatter(X[y_pred == 0]['petal length (cm)'],X[y_pred == 0]['petal width (cm)'],color='blue')
plt.scatter(X[y_pred == 1]['petal length (cm)'],X[y_pred == 1]['petal width (cm)'],color='green')
plt.scatter(X[y_pred == 2]['petal length (cm)'],X[y_pred == 2]['petal width (cm)'],color='yellow')

"""
sse = []
for i in range(1,10):
    model = KMeans(n_clusters=i)
    model.fit_predict(X,y)
    sse.append(model.inertia_)

plt.figure(figsize=(10,8))
plt.plot(range(1,10), sse)
"""