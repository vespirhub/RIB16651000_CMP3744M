import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot


data = pd.read_csv('CMP3744M_ADM_Assignment 2-dataset-nuclear_plants.csv')
data_iter = data.drop(data.columns[[0]], axis=1)
sns.set()

# pb

npdata = np.array(data.drop(data.columns[[0]], axis=1))
labels = np.array(data['Status'])
stats = ['Median','Mean','Std','Min','Max']
statistics = {}
nullcount = 0
n_nodes = [25,100,500]
n_trees = [10,50,100]

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
    median = str.format('{0:.4f}',np.median(x))
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
fig = plt.figure();
sns.boxplot(x=norm_data[:,0],y=data['Status']);
fig.savefig('boxplot.pdf')

normals = np.array(data[data['Status'] == 'Normal'])
abnormals = np.array(data[data['Status'] == 'Abnormal'])

fig = plt.figure();
sns.kdeplot(normals[:,5], color='red', shade=True, label="Normals");
sns.kdeplot(abnormals[:,5], color='blue', shade=True, label="Abnormals");
fig.savefig('density.pdf')
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
X = norm_data.T

def sigmoid(z):

    z = 1 / (1 + np.exp(-z))
    return z

def tanh(z):

    z = np.sinh(z) / np.cosh(z)
    return z

def relu(z):

    z = np.maximum(0, z)
    return z

def sigmoid_deriv(z):

    dz = sigmoid(z) * (1 - sigmoid(z))
    return dz

def tanh_deriv(z):

    dz = 1 - tanh(z)**2
    return dz

def relu_deriv(z):

    z[z < 0] = 0
    z[z > 0] = 1
    return z

def init_weights_he(input, hidden, output):

    np.random.seed(3)
    w1 = np.random.randn(hidden, input) * np.sqrt(2/(input-1))
    b1 = np.zeros((hidden, 1))
    w2 = np.random.randn(output, hidden)* np.sqrt(2/(hidden-1))
    b2 = np.zeros((output, 1))


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
def forward_prop(X, params, sigmoid_, tanh_, relu_):

    # Retrieve Current Weights and Biases from dict
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    # Layer 1 output, dot of layer 1 weights and X + layer 1 biases, "left side" of perceptron result
    z1 = w1 @ X + b1
    # Result of layer one, p1 = sigma of l1, "right side" of perceptron result
    if sigmoid_ == True:
        a1 = sigmoid(z1)
    elif tanh_ == True:
        a1 = tanh(z1)
    elif relu_ == True:
        a1 = relu(z1)
    # Layer 2 output, dot of layer 2 weights and p1 + layer 2 biases, "left side" of perceptron result
    z2 = w2 @ a1 + b2
    # Result of network, then used for update step, "right side" of perceptron result our prediction, "y_hat"
    a2 = sigmoid(z2)
    # Store current results in dict to calculate cost and update weights.
    model = {"z1" : z1,
             "a1" : a1,
             "z2" : z2,
             "a2" : a2}

    return a2, model

# Backstep Through network to find the gradients for optimising.
def back_prop(X,Y,params,model,sigmoid_,tanh_,relu_):

    w1 = params['w1']
    w2 = params['w2']
    a1 = model['a1']
    z1 = model['z1']
    a2 = model['a2']
    m = X.shape[1]

    dz2 = a2 - Y
    dw2 = (1 / m) * dz2 @ a1.T
    db2 = (1 / m) * np.sum(dz2,axis=1,keepdims=True)
    if sigmoid_ == True:
        dz1 = np.multiply(w2.T @ dz2, sigmoid_deriv(z1))
    elif tanh_ == True:
        dz1 = np.multiply(w2.T @ dz2, tanh_deriv(z1))
    elif relu_ == True:
        dz1 = np.multiply(w2.T @ dz2, relu_deriv(z1))
    dw1 = (1 / m) * dz1 @ X.T
    db1 = (1 / m) * np.sum(dz1,axis=1,keepdims=True)
    gradients = {"dw1" : dw1,
                 "db1" : db1,
                 "dw2" : dw2,
                 "db2" : db2}
    return gradients

# Optimizing the negative log(x) function
def binary_crossentropy(y_hat, Y):

    return -(1 / Y.shape[1]) * np.sum((Y * np.log(y_hat) + (1 - Y) * np.log(1-y_hat)))

# Accuracy metric using basic confusion matrix
def find_acc(y_hat,y_true):
    tp = np.sum((y_true == 1) & (y_hat == 1))
    fn = np.sum((y_true == 1) & (y_hat == 0))
    fp = np.sum((y_true == 0) & (y_hat == 1))
    tn = np.sum((y_true == 0) & (y_hat == 0))

    accuracy = (tp + tn) / (tp+tn+fp+fn)

    return accuracy
# Update step for the weights and bias using derivatives from backprop
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

# ANN where all functions are called
def ANN(X, Y, X_val, Y_val, hidden_nodes, batch_size, epochs, learning_rate, sigmoid_, tanh_, relu_):

    x_size = X.shape[0]
    y_size = Y.shape[0]

    m = X.shape[1] # 996
    n = X_val.shape[1] # 199

    # Different initialisation depending on which activation function is used.
    if ((sigmoid_ == True) or (tanh_ == True)):
        params = init_weights_xavier(x_size, hidden_nodes, y_size)
        print('x')
    elif relu_ == True:
        params = init_weights_he(x_size, hidden_nodes, y_size)
        print('y')

    # Random permutation of indices every loop, to shuffle the data
    for i in range(epochs):

        rp = np.random.permutation(X.shape[1])
        x = X[:,rp]
        y = Y[:,rp]

        # Decrease Learning Rate by 10% of its current self every 100 epochs
        if relu == True:
            if ((i % 100 == 0) & (i != 0)):
                learning_rate = learning_rate * .9
        else:
            if ((i % 100 == 0) & (i != 0)):
                learning_rate = learning_rate * .9

        # Counter for batching
        counter = 0
        for j in range(int(m/batch_size)):

            # Check if last batch is too large/small, create custom batch to end if so
            if counter+batch_size > m:
                x_train = x[:,counter:-1]
                y_train = y[:,counter:-1]
            else:
                x_train = x[:,counter:counter+batch_size]
                y_train = y[:,counter:counter+batch_size]

            # Retrieve yhat
            a2, model = forward_prop(x_train, params, sigmoid_, tanh_, relu_)
            # Calculate Cross entropy from the label
            loss = binary_crossentropy(a2, y_train)
            # Calculate derivatives through backprop
            gradients = back_prop(x_train,y_train,params,model, sigmoid_, tanh_, relu_)
            # Update step with backprop derivatives
            params = update_step(learning_rate,params,gradients)

            # Increase counter by batch size
            counter+=batch_size

        # Train whole training set with new weights
        a2_train, model = forward_prop(X, params, sigmoid_, tanh_, relu_)
        loss_train = binary_crossentropy(a2_train, Y)
        Yhat_train = np.round(a2_train)
        train_acc = find_acc(Yhat_train, Y)

        # Train validation set with new weights too
        a2, model = forward_prop(X_val, params, sigmoid_, tanh_, relu_)
        loss = binary_crossentropy(a2, Y_val)
        Yhat = np.round(a2)
        acc = find_acc(Yhat, Y_val)

        # Print stats every 100 epochs
        if i % 100 == 0:
            print("epoch: {} - train_loss: {:.5f} - train_acc:{:.2f} - train_corrects: {}/{} - lr:{:.5f}".format(
                i, loss_train, train_acc, np.sum(Yhat_train == Y), m, learning_rate))
            print("epoch: {} - valid_loss: {:.5f} - valid_acc:{:.2f}- val_corrects: {}/{}".format(
                i, loss, acc, np.sum(Yhat == Y_val), n))


    return params

def prediction(X,Y,params, sigmoid_, tanh_, relu_):

    # Prediction function for test set
    a2, model = forward_prop(X, params, sigmoid_, tanh_, relu_)
    Yhat = np.round(a2)
    acc = find_acc(Yhat, Y)

    return acc

# Lazy way of getting perfectly balanced data but it works.
for i in range(200):
    np.random.seed(i)
    rp = np.random.permutation(X.shape[1])

    train_split = int(X.shape[1] * .8)
    val_split = int(X.shape[1] * .1)
    test_split = int(X.shape[1] * .1)

    train_indices = rp[0:train_split]
    val_indices = rp[train_split:train_split+val_split]
    test_indices = rp[train_split+val_split:-1]

    x_train = X[:,train_indices]
    y_train = Y[:,train_indices]

    x_val = X[:,val_indices]
    y_val = Y[:,val_indices]

    x_test = X[:,test_indices]
    y_test = Y[:,test_indices]

    a = np.sum(y_test)
    b = np.sum(y_val)

    if ((a == 50) & (b == 50)):
        break

# Sigmoid activation in hidden layer through boolean passing
# sigmoid_weights = ANN(x_train, y_train, x_val, y_val, hidden_nodes=n_nodes[2],
#     batch_size=32, epochs=2500, learning_rate=0.18, sigmoid_=True, tanh_=False, relu_=False)
# # Tanh activation in hidden layer through boolean passing
# tanh_weights = ANN(x_train, y_train, x_val, y_val, hidden_nodes=n_nodes[2],
#     batch_size=32, epochs=1000, learning_rate=0.7, sigmoid_=False, tanh_=True, relu_=False)
# # ReLU activation in hidden layer through boolean passing
# relu_weights = ANN(x_train, y_train, x_val, y_val, hidden_nodes=n_nodes[2],
#     batch_size=32, epochs=1000, learning_rate=0.0475, sigmoid_=False, tanh_=False, relu_=True)
#
# # Test Results
# acc_sigmoid = prediction(x_test,y_test,sigmoid_weights, sigmoid_=True, tanh_=False, relu_=False)
# print("Sigmoid Test Accuracy: {:.2f}".format(acc_sigmoid * 100))
#
# acc_tanh = prediction(x_test,y_test,tanh_weights, sigmoid_=False, tanh_=True, relu_=False)
# print("Tanh Test Accuracy: {:.2f}".format(acc_tanh * 100))
#
# acc_relu = prediction(x_test,y_test,relu_weights, sigmoid_=False, tanh_=False, relu_=True)
# print("ReLU Test Accuracy: {:.2f}".format(acc_relu * 100))
#
# # Prepare Data for RCF
# y_tree = Y.T.flatten()
# X_tree = norm_data
# # SKLearn data split method
# x_train, x_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.1)
#
# # Create model with 100 Trees
# model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10);
# # Fitting the RFC classifier to the training set
# model.fit(x_train, y_train);
# # Predicting on the test set
# predictions = model.predict(x_test)
# # Calculating absolute errors
# abs_errors = abs(predictions - y_test)
# # Performing accuracy metric, correct predictions over total samples
# acc = 100 * np.sum(predictions == y_test) / x_test.shape[0]
#
# print('Accuracy:', np.round(int(acc)), '%.')

# # COde to print a tree
# tree = model.estimators_[-1]
# export_graphviz(tree, out_file = 'tree.dot', feature_names = np.array(data.drop(data.columns[[0]], axis=1)).T, rounded = True, precision = 1)
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
#
# a = 'tree.dot'
# graph.write_png('tree100.png')

# Integer division of total samples for splits
tensplit = int(X.shape[1] / 10)

# Seed for consistent pseudo-random
np.random.seed(8)
# Random index permutation for shuffle
rp = np.random.permutation(X.shape[1])

# Apply new indices
X = X[:,rp]
Y = Y[:,rp]

# Split to ten sets
x1 = X[:,0:tensplit]
y1 = Y[:,0:tensplit]

x2 = X[:,tensplit:tensplit*2]
y2 = Y[:,tensplit:tensplit*2]

x3 = X[:,tensplit*2:tensplit*3]
y3 = Y[:,tensplit*2:tensplit*3]

x4 = X[:,tensplit*3:tensplit*4]
y4 = Y[:,tensplit*3:tensplit*4]

x5 = X[:,tensplit*4:tensplit*5]
y5 = Y[:,tensplit*4:tensplit*5]

x6 = X[:,tensplit*5:tensplit*6]
y6 = Y[:,tensplit*5:tensplit*6]

x7 = X[:,tensplit*6:tensplit*7]
y7 = Y[:,tensplit*6:tensplit*7]

x8 = X[:,tensplit*7:tensplit*8]
y8 = Y[:,tensplit*7:tensplit*8]

x9 = X[:,tensplit*8:tensplit*9]
y9 = Y[:,tensplit*8:tensplit*9]

x10 = X[:,tensplit*9:-1]
y10 = Y[:,tensplit*9:-1]

# Create list of the mini- training sets to iterate over for the K-Fold CV
fold_x = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
fold_y = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]

# Initialise result lists
n1,n2,n3,tn1,tn2,tn3 = ([] for i in range(6))

# Loop over all k fold subsets
for i in range(10):
    # Loop over neuron configs
    for j in range(3):

        # Test is the current fold
        x_test = fold_x[i]
        y_test = fold_y[i]

        # Training is anything except current k'th fold
        x_train = fold_x[:i] + fold_x[i+1:]
        y_train = fold_y[:i] + fold_y[i+1:]

        x_train = np.hstack((x_train[0], x_train[1],
                   x_train[2], x_train[3],
                   x_train[4], x_train[5],
                   x_train[6], x_train[7],
                   x_train[8]))

        y_train = np.hstack((y_train[0], y_train[1],
                   y_train[2], y_train[3],
                   y_train[4], y_train[5],
                   y_train[6], y_train[7],
                   y_train[8]))

        # Train and predict with sigmoid activation, validate on test set
        sigmoid_weights = ANN(x_train, y_train, x_test, y_test, hidden_nodes=n_nodes[j],
            batch_size=32, epochs=1000, learning_rate=0.2, sigmoid_=True, tanh_=False, relu_=False)
        acc_sigmoid = prediction(x_test,y_test, sigmoid_weights, sigmoid_=True, tanh_=False, relu_=False)
        print("Sigmoid Test Accuracy: {:.2f}".format(acc_sigmoid * 100))

        # Train and predict with RFC, need to flatten and transpose some things to make dimensions match.
        model = RandomForestClassifier(n_estimators=n_trees[j], min_samples_leaf=1);
        model.fit((x_train.T), (y_train.T.flatten()));
        predictions = model.predict(x_test.T)
        abs_errors = abs(predictions - y_test.T.flatten())
        acc = np.sum(predictions == y_test.T.flatten()) / y_test.shape[0]

        # Append RESULTS
        if j == 0:
            n1.append(acc_sigmoid*100)
            tn1.append(acc)
        elif j == 1:
            n2.append(acc_sigmoid*100)
            tn2.append(acc)
        elif j == 2:
            n3.append(acc_sigmoid*100)
            tn3.append(acc)

# Result Percentages
ann_25 = np.round((np.sum(n1) / 10))
ann_50 = np.round((np.sum(n2) / 10))
ann_100 = np.round((np.sum(n3) / 10))

rfc_10 = np.round((np.sum(tn1) / 10))
rfc_50 = np.round((np.sum(tn2) / 10))
rfc_100 = np.round((np.sum(tn3) / 10))
