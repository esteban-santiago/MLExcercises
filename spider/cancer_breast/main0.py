#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 22:05:17 2022

@author: esantiago
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn 
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/data.csv")

print(df.info())
print(df.describe())

df = df.rename(columns={"diagnosis":"target"})



X = df.drop("target",axis=1)
y = df["target"]
y = y.replace({"M":0, "B":1}) 

print(y.value_counts())