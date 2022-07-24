
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
#Keras import
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout


#df = pd.read_csv("data/data.csv") 

#print(df.describe().T) #Values to be normalizer before fitting

#print(df.isnull().sum())

#Rename dataset Diagnosis Column to label to easy to understand
#df = df.rename(columns={"diagnosis":"label"})
#print(df.describe().T)

np.random.seed(42)

n_samples = 100

rng = np.random.randn(n_samples) * 10

rng_ = rng.reshape(-1,1)

print(rng.argmax())
print(rng.argmin())



