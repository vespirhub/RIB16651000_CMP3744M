import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot


data = pd.read_csv('~/github/RIB16651000_CMP3744M/RIB16651000_CMP3744M-master/CMP3744M_ADM_Assignment 2-dataset-nuclear_plants.csv')
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
for i in range(len(Y)):
    if labels[i] == 'Normal':
        Y[i] = 1
    else:
        Y[i] = 0
Y = Y.reshape(1, -1)

Y.shape

X = np.zeros(npdata.shape).T
for i in range(len(npdata)):
    X[:,i] = npdata[i,:]
X.shape

X = norm_data.T

def sigmoid(z):

    z = 1 / (1 + np.exp(-z))
    return z

def sigmoid_deriv(z):

    dz = sigmoid(z) * (1 - sigmoid(z))
    return dz

def init_weights_he(input, hidden, output):

    np.random.seed(3)
    w1 = np.random.randn(hidden, input) * np.sqrt(2/(input-1))
    #print("w1",w1)
    b1 = np.zeros((hidden, 1))
    #b1.shape
    w2 = np.random.randn(output, hidden)* np.sqrt(2/(hidden-1))
    #print("w2",w2)
    b2 = np.zeros((output, 1))
    #b2.shape

    params = {"w1" : w1,
              "b1" : b1,
              "w2" : w2,
              "b2" : b2}
    return params

def init_weights_xavier(input,hidden,output):

    np.random.seed(3)
    w1 = np.random.randn(hidden, input) * np.sqrt(2/(hidden+input))
    b1 = np.zeros((hidden,1))
    w2 = np.random.randn(output, hidden) * np.sqrt(2/(output+hidden))
    b2 = np.zeros((output,1))

    params = {"w1" : w1,
              "b1" : b1,
              "w2" : w2,
              "b2" : b2}
    return params

# params = init_weights(X.shape[0], n_nodes[1], 1)

#Forward Propagation Step Function
def forward_prop(X, params):

    # Retrieve Current Weights and Biases from dict
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    # Layer 1 output, dot of layer 1 weights and X + layer 1 biases, "left side" of perceptron result
    z1 = w1 @ X + b1
    # Result of layer one, p1 = sigma of l1, "right side" of perceptron result
    a1 = sigmoid(z1)
    #print("a1",a1.shape)
    # Layer 2 output, dot of layer 2 weights and p1 + layer 2 biases, "left side" of perceptron result
    z2 = w2 @ a1 + b2
    #print("z2",z2.shape)
    # Result of network, then used for update step, "right side" of perceptron result our prediction, "y_hat"
    a2 = sigmoid(z2)
    #print("a2",a2.shape)

    # Store current results in dict to calculate cost and update weights.
    model = {"z1" : z1,
             "a1" : a1,
             "z2" : z2,
             "a2" : a2}

    return a2, model

def back_prop(X,Y,params,model):

    w1 = params['w1']
    w2 = params['w2']
    a1 = model['a1']
    z1 = model['z1']
    a2 = model['a2']
    m = X.shape[1]

    dz2 = a2 - Y
    #print("dz2",dz2.shape)
    dw2 = (1 / m) * dz2 @ a1.T
    #print("dw2",dw2.shape,"a1",a1.T.shape)
    db2 = (1 / m) * np.sum(dz2,axis=1,keepdims=True)
    dz1 = np.multiply(w2.T @ dz2, sigmoid_deriv(z1))
    #print("dz1",dz1.shape)
    dw1 = (1 / m) * dz1 @ X.T
    #print("dw1",dw1.shape)
    db1 = (1 / m) * np.sum(dz1,axis=1,keepdims=True)
    gradients = {"dw1" : dw1,
                 "db1" : db1,
                 "dw2" : dw2,
                 "db2" : db2}
    return gradients

def binary_crossentropy(y_hat, Y):

    return -(1 / Y.shape[1]) * np.sum((Y * np.log(y_hat) + (1 - Y) * np.log(1-y_hat)))

def find_acc(y_hat,y_true):
    tp = np.sum((y_true == 1) & (y_hat == 1))
    fn = np.sum((y_true == 1) & (y_hat == 0))
    fp = np.sum((y_true == 0) & (y_hat == 1))
    tn = np.sum((y_true == 0) & (y_hat == 0))

    accuracy = (tp + tn) / (tp+tn+fp+fn)
    #print('tp: {}' 'tn: {}' 'fp: {}' 'fn: {}'.format(tp,tn,fp,fn))

    return accuracy

def update_step(learning_rate, params, gradients):

    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    dw1 = gradients['dw1']
    db1 = gradients['db1']
    dw2 = gradients['dw2']
    db2 = gradients['db2']

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    params = {"w1": w1,
              "b1": b1,
              "w2": w2,
              "b2": b2}
    return params


def ANN(X, Y, hidden_nodes, batch_size, epochs, learning_rate):

    x_size = X.shape[0]
    x_size
    y_size = Y.shape[0]

    m = X.shape[1] # 996

    params = init_weights_xavier(x_size, hidden_nodes, y_size)

    for i in range(epochs):

        rp = np.random.permutation(X.shape[1])
        x = X[:,rp]
        y = Y[:,rp]

        if ((i % 500 == 0) & (i != 0)):
          learning_rate = learning_rate * .9

        counter = 0
        for j in range(int(m/batch_size)):

            x_train = x[:,counter:counter+batch_size]
            y_train = y[:,counter:counter+batch_size]

            a2, model = forward_prop(x_train, params)
            loss = binary_crossentropy(a2, y_train)
            gradients = back_prop(x_train,y_train,params,model)
            params = update_step(learning_rate,params,gradients)

            counter+=batch_size

        a2, model = forward_prop(x, params)
        loss = binary_crossentropy(a2, y)
        Yhat = np.round(a2)
        acc = find_acc(Yhat, y)

        print("epoch: {} - loss: {:.5f} - acc:{:.2f} - lr:{:.5f} - corrects: {}/{}".format(
            i, loss, acc, learning_rate, np.sum(Yhat == y), m))

    return params

params = ANN(X, Y, hidden_nodes=n_nodes[-1],
    batch_size=12, epochs=50000, learning_rate=0.01)
#
# y_tree = Y.T.flatten()
# X_tree = norm_data
#
# x_train, x_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.1)
#
# model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1);
#
# model.fit(x_train, y_train);
#
# predictions = model.predict(x_test)
#
# abs_errors = abs(predictions - y_test)
#
# acc = 100 * np.sum(predictions == y_test) / x_test.shape[0]
#
# print('Accuracy:', np.round(int(acc)), '%.')
#
# tree = model.estimators_[-1]
#
# export_graphviz(tree, out_file = 'tree.dot', feature_names = np.array(data.drop(data.columns[[0]], axis=1)).T, rounded = True, precision = 1)
#
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
#
# graph.write_png('tree.png')
