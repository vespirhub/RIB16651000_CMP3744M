import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('CMP3744M_ADM_Assignment 2-dataset-nuclear_plants.csv')
data_iter = data.drop(data.columns[[0]], axis=1)
sns.set()

npdata = np.array(data.drop(data.columns[[0]], axis=1))
labels = np.array(data['Status'])
stats = ['Mean','Std','Min','Max']
statistics = {}
nullcount = 0

# Check for Categorical Data
category = data.select_dtypes(exclude=["number","bool"])
if not category.empty:
    print('Categorical Values Present in Columns: {}'.format(category.columns.values))

# Check for Missing Values
for i, col in enumerate(data_iter):
    for j in range(len(data_iter)):
        if npdata[j,i] is None:
            print('Null Value Found, Row:{}, Col:{}'.format(j,i))
            nullcount+=1
if nullcount == 0:
    print('No Null Values Found')

# Compute Statistics
for i, col in enumerate(data_iter):
    x = npdata[:,i]
    for j in range(4):
        mean = str.format('{0:.4f}',np.mean(x))
        std = str.format('{0:.4f}',np.std(x))
        min = str.format('{0:.4f}',np.min(x))
        max = str.format('{0:.4f}',np.max(x))
        tmp = [mean,std,min,max]
        for k in range(len(stats)):
            label = (data_iter.columns[i]+' '+stats[k])
            statistics[label] = tmp[k]

# Feature-Wise Normalisation
norm_data = np.array(data.drop(data.columns[[0]], axis=1))
for i, col in enumerate(data_iter):
    norm_data[:,i] = (norm_data[:,i] - np.mean(norm_data[:,i])) / (np.std(norm_data[:,i]) + 1e-7)

# Plots

sns.boxplot(x=norm_data[:,0],y=data['Status']);



# Size of data, 996,13 |||| 12 Features |||| Labels are categorical : i.e Normal Abnormal |||| Normalise so features have similar ranges for gradient descent to converge faster + no oscillation of weights.
