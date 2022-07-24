"""
Created on Fri Jul  1 15:05:26 2022

@author: esantiago
"""
import warnings;
warnings.filterwarnings("ignore");

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
%matplotlib inline
import seaborn as sns;
import sklearn;
import sys
import statsmodels.api as sm;
import statsmodels.formula.api as smf;
from sklearn.datasets import load_boston;



#data_url = "http://lib.stat.cmu.edu/datasets/boston"
#raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#target = raw_df.values[1::2, 2]



boston_data = load_boston();

df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names);

X = df.values;
y = boston_data.target;

X_constant = sm.add_constant(df);

model = sm.OLS(y, X_constant);

lr = model.fit();
print(lr.summary());

form_lr = smf.ols(formula="y~CRIM+ZN+CHAS+NOX+RM+DIS+RAD+TAX+PTRATIO+B+LSTAT", data=df)
mlr = form_lr.fit()
print(mlr.summary());

form_lr = smf.ols(formula="y~CRIM+ZN+INDUS+CHAS+NOX+RM+AGE+DIS+RAD+TAX+PTRATIO+B+LSTAT", data=df)
mlr = form_lr.fit()
print(mlr.summary());

form_lr = smf.ols(formula="y~CRIM+ZN+CHAS+NOX", data=df)
mlr = form_lr.fit()
print(mlr.summary());
