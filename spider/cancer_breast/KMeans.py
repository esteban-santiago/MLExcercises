#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 16:25:17 2022

@author: esantiago
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def numOfClusters(data, maxNumClusters):
    inertia = []
    for i in range(1, maxNumClusters):
        model = KMeans(n_clusters=i)
        model.fit_predict(data)
        inertia.append(model.inertia_)
    return inertia


np.random.seed(10)
n = 25
_clusters = 25


df = pd.DataFrame()
df['X'] = np.random.rand(n)
df['y'] = np.random.rand(n) ** 2
   
#km = KMeans(n_clusters=4)  

#y_pred = km.fit_predict(df[['X','y']])

sse = numOfClusters(df[['X','y']],_clusters)
#print('Inertia: ', km.inertia_)

plt.figure(figsize=(10,8))
plt.plot(range(1, _clusters), sse)



"""
plt.figure(figsize=(12,8))
plt.scatter(df[y_pred ==0]['X'], df[y_pred ==0]['y'], color='green')
plt.scatter(df[y_pred ==1]['X'], df[y_pred ==1]['y'], color='blue')
plt.scatter(df[y_pred ==2]['X'], df[y_pred ==2]['y'], color='black')
plt.scatter(df[y_pred ==3]['X'], df[y_pred ==3]['y'], color='orange')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='red', marker='*')
"""


