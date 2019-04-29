import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('CMP3744M_ADM_Assignment 2-dataset-nuclear_plants.csv')
data_iter = data.drop(data.columns[[0]], axis=1)
sns.set()

npdata = np.array(data.drop(data.columns[[0]], axis=1))
labels = np.array(data['Status'])
stats = ['Median','Mean','Std','Min','Max']
statistics = {}
nullcount = 0

# Check for Categorical Data
# Labels are categorical : i.e Normal Abnormal
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
    median = str.format('{0:.4f}', np.median(x))
    mean = str.format('{0:.4f}',np.mean(x))
    std = str.format('{0:.4f}',np.std(x))
    min = str.format('{0:.4f}',np.min(x))
    max = str.format('{0:.4f}',np.max(x))
    tmp = [mean,median,std,min,max]
    for j in range(len(stats)):
        label = (data_iter.columns[i]+' '+stats[j])
        statistics[label] = tmp[j]

# Feature-Wise Normalisation
# Normalise so features have similar ranges for gradient descent to converge faster + no oscillation of weights.
norm_data = np.array(data.drop(data.columns[[0]], axis=1))
for i, col in enumerate(data_iter):
    norm_data[:,i] = (norm_data[:,i] - np.mean(norm_data[:,i])) / (np.std(norm_data[:,i]) + 1e-7)

# Size of data, 996,13 |||| 12 Features ||||
# Plots
sns.boxplot(x=norm_data[:,0],y=data['Status']);

normals = np.array(data[data['Status'] == 'Normal'])
abnormals = np.array(data[data['Status'] == 'Abnormal'])

fig = plt.figure();
sns.kdeplot(normals[:,5], color='red', shade=True);
sns.kdeplot(abnormals[:,5], color='blue', shade=True);

# Outliers found in the rightmost side of the plot, we can prove this speculation by calculating the outliers.

def find_outliers(normals,abnormals, feature):
    # Get first, second and third quartile of feature.
    Q1_N, Q2_N, Q3_N = np.percentile(normals, [25, 50, 75])
    Q1_A, Q2_A, Q3_A = np.percentile(abnormals, [25, 50, 75])

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
        if (normals[i] < inner_N1) or (normals[i] > inner_N2):
            minor_normal_outliers.append(normals[i])
        if (normals[i] < outer_N1) or (normals[i] > outer_N2):
            major_normal_outliers.append(normals[i])
    print('Found {} minor outliers and {} major outliers in the Normal Class of {}!'.format(len(minor_normal_outliers),len(major_normal_outliers),feature))

    for i in range(len(normals)):
        if (abnormals[i] < inner_A1) or (abnormals[i] > inner_A2):
            minor_abnormal_outliers.append(abnormals[i])
        if (abnormals[i] < outer_A1) or (abnormals[i] > outer_A2):
            major_abnormal_outliers.append(abnormals[i])
    print('Found {} minor outliers and {} major outliers in the Abnormal Class of {}!'.format(len(minor_abnormal_outliers),len(major_abnormal_outliers),feature))

    return minor_normal_outliers, major_normal_outliers, minor_abnormal_outliers, major_abnormal_outliers

outliers = {}
# Using a simple script we can calculate the minor and major outliers for both classes for every feature in the feature set.
for i in range(1, 13):
    min_norm, maj_norm, min_abn, maj_abn = [[] for i in range(4)]
    min_norm, maj_norm, min_abn, maj_abn = find_outliers(normals[:,i], abnormals[:,i], data.columns.values[i])

    outliers['minor_normal'+' '+data.columns.values[i]] = min_norm
    outliers['major_normal'+' '+data.columns.values[i]] = maj_norm
    outliers['minor_abnormal'+' '+data.columns.values[i]] = min_abn
    outliers['major_abnormal'+' '+data.columns.values[i]] = maj_abn

Y = np.zeros(labels.shape, dtype=int)
for i in range(len(num_labels)):
    if labels[i] == 'Normal':
        Y[i] = 1
    else:
        Y[i] = 0
Y.reshape(-1, 1)

def sigmoid(z):
    z = 1 / 1(1 + np.exp(-z))
    return z

def init_weights(size):
    w = np.zeros(shape=(size, 1)) * 0.01
    b = 0
    return w, b

w, b = init_weights(25)



#def fortward_prop(w, b, X, Y):