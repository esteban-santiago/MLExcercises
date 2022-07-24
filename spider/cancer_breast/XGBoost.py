#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 22:12:30 2022

@author: esantiago
"""

import pandas as pd
import seaborn as sns

df = pd.read_csv('data/data.csv')
#print(df.describe().T)
#print(df.isnull().sum())

#df.drop(labels=['Unnamed: 32'],inplace=True)
df.rename(columns={'diagnosis':'label'}, inplace=True)

sns.countplot(x='label', data=df)

df['label'].value_counts()

y = df['label'].values


from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(y)

X = df.drop(labels=['label', 'id'], axis=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)


import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score


print("Accuracy: ", (accuracy_score(y_test, y_pred) *100),'%')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)

