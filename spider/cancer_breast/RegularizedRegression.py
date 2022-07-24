#Created on Fri Jul  1 17:47:12 2022
#
#@author: esantiago
#"""


#Ridge aka L2
#LASSO Regression aka L1 
#Elastic Net - mix between L1 and L2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

np.random.seed(42)

n_samples = 100

rng = np.random.randn(n_samples) * 10
y_gen = 0.5 * rng + 2 * np.random.randn(n_samples)

lr = LinearRegression()
lr.fit(rng.reshape(-1,1), y_gen)
model_pred = lr.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10,8))
plt.scatter(rng, y_gen)
plt.plot(rng, model_pred)
print("Coefficient Estimate: ", lr.coef_)


idx = rng.argmax()
y_gen[idx] = 200
idx = rng.argmin()
y_gen[idx] = -200

plt.figure(figsize=(10,8))
plt.scatter(rng, y_gen)

o_lr = LinearRegression(normalize=True)
o_lr.fit(rng.reshape(-1,1), y_gen)
o_model_pred = o_lr.predict(rng.reshape(-1,1))

plt.scatter(rng, y_gen)
plt.plot(rng, o_model_pred)
print("Coefficient Estimate =", o_lr.coef_)

#Ridge
ridge_mod = Ridge(alpha=0.5, normalize=True)
ridge_mod.fit(rng.reshape(-1, 1), y_gen)
ridge_mod_pred = ridge_mod.predict(rng.reshape(-1,1))

plt.figure(figsize=(10,8))
plt.scatter(rng, y_gen)
plt.plot(rng, ridge_mod_pred)
print("Coefficient Estimate =", ridge_mod.coef_)

#Lasso
lasso_mod = Lasso(alpha=0.4, normalize=True)
lasso_mod.fit(rng.reshape(-1,1), y_gen)
lasso_mod_pred = lasso_mod.predict(rng.reshape(-1,1))

plt.figure(figsize=(10,8))
plt.scatter(rng, y_gen)
plt.plot(rng, lasso_mod_pred)
print("Coefficient Estimate =", lasso_mod.coef_)

#Elastic Net
en_mod = ElasticNet(alpha=0.02, normalize=True)
en_mod.fit(rng.reshape(-1,1), y_gen)
en_mod_pred = en_mod.predict(rng.reshape(-1,1))

plt.figure(figsize=(10,8))
plt.scatter(rng, y_gen)
plt.plot(rng, en_mod_pred)
print("Coefficient Estimate =", en_mod.coef_)

