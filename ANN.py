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

normals = np.array(data[data['Status'] == 'Normal'])
abnormals = np.array(data[data['Status'] == 'Abnormal'])

fig = plt.figure();
sns.kdeplot(normals[:,5], color='red', shade=True);
sns.kdeplot(abnormals[:,5], color='blue', shade=True);

# Outliers found in the rightmost side of the plot, we can prove this speculation by calculating the outliers.

# Get first, second and third quartile of feature.
Q1_N, Q2_N, Q3_N = np.percentile(normals[:,5], [25, 50, 75])
Q1_A, Q2_A, Q3_A = np.percentile(abnormals[:,5], [25, 50, 75])

# Calculate the interquartile range.
IQR_N, IQR_A = (Q3_N - Q1_N, Q3_A - Q1_A)

# Establish inner and outer fences to gauge the type of outliers found.
inner_N1, inner_N2 = (Q1_N - IQR_N * 1.5, Q3_N + IQR_N * 1.5)
outer_N1, outer_N2 = (Q1_N - IQR_N * 3, Q3_N + IQR_N * 3)
inner_A1, inner_A2 = (Q1_A - IQR_A * 1.5, Q3_A + IQR_A * 1.5)
outer_A1, outer_A2 = (Q1_A - IQR_A * 3, Q3_A + IQR_A * 3)

# Allocate empty lists for outliers
minor_normal_outliers, major_normal_outliers, minor_abnormal_outliers, major_abnormal_outliers = [[] for i in range(4)]

# If values are less than the lower_inner fence or bigger than the bigger_inner fence assign as minor outlier, lands within 1.5 * the IQR
# If values are less than the lower_outer fence or bigger than the bigger_outer fence assign as major outlier, lands within 3 * the IQR
for i in range(len(normals)):
    if (normals[i,5] < inner_N1) or (normals[i,5] > inner_N2):
        minor_normal_outliers.append(normals[i,5])
    if (normals[i,5] < outer_N1) or (normals[i,5] > outer_N2):
        major_normal_outliers.append(normals[i,5])
print('Found {} minor outliers and {} major outliers in the Normal Class of {}!'.format(len(minor_normal_outliers),len(major_normal_outliers),data.columns.values[5]))
# Normals contain 15 minor outliers, 0 major outliers.

# If values are less than the lower_inner fence or bigger than the bigger_inner fence assign as minor outlier, lands within 1.5 * the IQR
# If values are less than the lower_outer fence or bigger than the bigger_outer fence assign as major outlier, lands within 3 * the IQR
for i in range(len(normals)):
    if (abnormals[i,5] < inner_A1) or (abnormals[i,5] > inner_A2):
        minor_abnormal_outliers.append(abnormals[i,5])
    if (abnormals[i,5] < outer_A1) or (abnormals[i,5] > outer_A2):
        major_abnormal_outliers.append(abnormals[i,5])

print('Found {} minor outliers and {} major outliers in the Abnormal Class of {}!'.format(len(minor_abnormal_outliers),len(major_abnormal_outliers),data.columns.values[5]))
# Abnormals contain 16 minor outliers, 3 major outliers.

# Size of data, 996,13 |||| 12 Features |||| Labels are categorical : i.e Normal Abnormal |||| Normalise so features have similar ranges for gradient descent to converge faster + no oscillation of weights.
