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
n_nodes = [25,100,500]

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
Y = Y.reshape(-1, 1)

X = np.zeros(npdata.shape).T
for i in range(len(npdata)):
    X[:,i] = npdata[i,:]

def sigmoid(z):

    z = 1 / (1 + np.exp(-z))
    return z

def sigmoid_deriv(z):

    dz = sigmoid(z) * (1 - z)
    return dz

def init_weights(input, hidden, output):

    np.random.seed(2)
    w1 = np.random.randn(hidden, input) * 0.01
    b1 = np.zeros((hidden, 1))
    w2 = np.random.randn(output, hidden) * 0.01
    b2 = np.zeros((output, 1))

    params = {"w1" : w1,
              "b1" : b1,
              "w2" : w2,
              "b2" : b2}
    return params

params = init_weights(X.shape[0], n_nodes[1], 1)

#L(y_hat, y) = -(y * np.log(y_hat) + (1 - y) * log(1-y_hat))

#Forward Propagation Step Function
def forward_prop(X, params):

    # Retrieve Current Weights and Biases from dict
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    # Layer 1 output, dot of layer 1 weights and X + layer 1 biases
    l1 = w1 @ X + b1
    # Result of layer one, p1 = sigma of l1
    p1 = sigmoid(l1)
    # Layer 2 output, dot of layer 2 weights and p1 + layer 2 biases
    l2 = w2 @ p1 + b2
    # Result of network, then used for update step
    p2 = sigmoid(l2)

    # Restore current results in dict to calculate cost and update weights.
    results = {"l1" : l1,
               "p1" : p1,
               "l2" : l2,
               "p2" : p2}

    return p2, results
