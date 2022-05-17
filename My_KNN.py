# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:19:52 2022

@author: anan
"""
# plot_single_pair
def plot_single_pair(ax, feature_ind1, feature_ind2, _X, _y, _features, colormap):
    """Plots single pair of features.

    Parameters
    ----------
    ax : Axes
        matplotlib axis to be plotted
    feature_ind1 : int
        index of first feature to be plotted
    feature_ind2 : int
        index of second feature to be plotted
    _X : numpy.ndarray
        Feature dataset of of shape m x n
    _y : numpy.ndarray
        Target list of shape 1 x n
    _features : list of str
        List of n feature titles
    colormap : dict
        Color map of classes existing in target

    Returns
    -------
    None
    """

    # Plot distribution histogram if the features are the same (diagonal of the pair-plot).
    if feature_ind1 == feature_ind2:
        tdf = pd.DataFrame(_X[:, [feature_ind1]], columns = [_features[feature_ind1]])
        tdf['target'] = _y
        for c in colormap.keys():
            tdf_filtered = tdf.loc[tdf['target']==c]
            ax[feature_ind1, feature_ind2].hist(tdf_filtered[_features[feature_ind1]], color = colormap[c], bins = 30)
    else:
        # other wise plot the pair-wise scatter plot
        tdf = pd.DataFrame(_X[:, [feature_ind1, feature_ind2]], columns = [_features[feature_ind1], _features[feature_ind2]])
        tdf['target'] = _y
        for c in colormap.keys():
            tdf_filtered = tdf.loc[tdf['target']==c]
            ax[feature_ind1, feature_ind2].scatter(x = tdf_filtered[_features[feature_ind2]], y = tdf_filtered[_features[feature_ind1]], color=colormap[c])

    # Print the feature labels only on the left side of the pair-plot figure
    # and bottom side of the pair-plot figure. 
    # Here avoiding printing the labels for inner axis plots.
    if feature_ind1 == len(_features) - 1:
        ax[feature_ind1, feature_ind2].set(xlabel=_features[feature_ind2], ylabel='')
    if feature_ind2 == 0:
        if feature_ind1 == len(_features) - 1:
            ax[feature_ind1, feature_ind2].set(xlabel=_features[feature_ind2], ylabel=_features[feature_ind1])
        else:
            ax[feature_ind1, feature_ind2].set(xlabel='', ylabel=_features[feature_ind1])

# myplotGrid
import matplotlib.pyplot as plt
def myplotGrid(X, y, features, colormap={0: "red", 1: "green", 2: "blue"}):
    """Plots a pair grid of the given features.

    Parameters
    ----------
    X : numpy.ndarray
        Dataset of shape m x n
    y : numpy.ndarray
        Target list of shape 1 x n
    features : list of str
        List of n feature titles

    Returns
    -------
    None
    """

    feature_count = len(features)
    # Create a matplot subplot area with the size of [feature count x feature count]
    fig, axis = plt.subplots(nrows=feature_count, ncols=feature_count)
    # Setting figure size helps to optimize the figure size according to the feature count.
    fig.set_size_inches(feature_count * 4, feature_count * 4)

    # Iterate through features to plot pairwise.
    for i in range(0, feature_count):
        for j in range(0, feature_count):
            plot_single_pair(axis, i, j, X, y, features, colormap)

    plt.show()
#%% Euclidean
"""Perhitungan jarak antar 2 titik dengan rumus Euclidean"""
import numpy as np

def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
#%% KNN
def KNN(Xtrain, Ytrain, Xtest, K=3):
    # Mengetahui panjang/jumlah dataset training (m)
    m = Xtrain.shape[0]
    data = []
    for i in range(m):
        xi = Xtrain[i]
        distance = euclidean(Xtest, xi)
        data.append((distance,Ytrain[i]))
        # mengurutkan data dari yang terkecil sejumlah K
    data = sorted(data, key=lambda x:x[0])[:K] 
    # memastikan data berbentuk array
    data = np.asarray(data)
    # menentukan label kelas yang terbanyak muncul di K data (data[:,1])
    data_baru = np.unique(data[:,1], return_counts= True)
    index = data_baru[1].argmax()
    output = data_baru[0][index]
    return output
#%% Akurasi
def akurasi(Ytest, Yprediksi):
        akurasi = np.sum(Ytest == Yprediksi) / len(Yprediksi)
        return akurasi
#%% Plot Dataset
from sklearn import datasets

iris = datasets.load_iris()
myplotGrid(iris.data, iris.target, iris.feature_names, colormap={0: "red", 1: "green", 2: "blue"})
#%% Dataset Iris
import pandas as pd

Dataset = pd.read_csv('iris.csv') # keterangan bisa lihat di Iris sklearn   
#%% Data Testing
Testing = Dataset[0:10]# 10 setosa (0)
Testing = Testing.append(Dataset[50:60])# 10 versicolor (1)
Testing = Testing.append(Dataset[100:110])# 10 virginica (2)
Xtest = Testing[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
Ytest = Testing[['target']]
#%% Data Training
Training = Dataset[10:50]# 40 setosa (0)
Training = Training.append(Dataset[60:100])# 40 versicolor (1)
Training = Training.append(Dataset[110:150])# 40 virginica (2)
Xtrain = Training[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
Ytrain = Training[['target']] 
#%% Call code
Ytrain = Training[['target_names']] 
Xtrain = np.asarray (Xtrain)
Ytrain = np.asarray (Ytrain)
Xtest = np.asarray (Xtest)
Ytest = np.asarray (Ytest)
F =[]
for i in range(len(Xtest)):
    F.append(KNN(Xtrain, Ytrain, Xtest[i,:]))
Yprediksi = pd.DataFrame(F)
Akurasi = akurasi(Ytest, F)  
#%%
from sklearn import datasets

iris_dataset = datasets.load_iris()
X = iris_dataset.data
Y = iris_dataset.target

iris_dataframe = pd.DataFrame(X, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y, figsize=(15, 15), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8) 
#%%
x = np.array([5,7,8, 2,17])
y1 = np.array([99,103,87,94,78])
y2 = np.array([26, 23, 18, 55, 16])

# Scatter Plot color array

plt.scatter(x, y1, color='green')
plt.scatter(x, y2, color='red')

# Display

plt.show()