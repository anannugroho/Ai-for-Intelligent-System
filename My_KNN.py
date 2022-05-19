# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:19:52 2022

@author: anan
"""
#%% Load Dataset Iris
from sklearn import datasets
iris = datasets.load_iris()

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
#%% Singleplot_Berpasangan
def Singleplot_Berpasangan(axis, urutan_feature_ke1, urutan_feature_ke2, Data_Features, Target_Classes, Nama_Features, Colormap):
    """
    Parameters
    ----------
    Data_Features  : numpy.ndarray
    Target_Classes : numpy.ndarray
    Nama_Features  : list of string

    """
    # Buat plot histogram untuk dua features yang sama
    if urutan_feature_ke1 == urutan_feature_ke2:
        data = pd.DataFrame(Data_Features[:, [urutan_feature_ke1]], columns = [Nama_Features[urutan_feature_ke1]])
        data['target'] = Target_Classes
        for c in Colormap.keys():
            data_filtered = data.loc[data['target']==c]
            axis[urutan_feature_ke1, urutan_feature_ke2].hist(data_filtered[Nama_Features[urutan_feature_ke1]], color = Colormap[c], bins = 30)
    else:
        # Buat plot scatter untuk dua features yang berbeda
        data = pd.DataFrame(Data_Features[:, [urutan_feature_ke1, urutan_feature_ke2]], columns = [Nama_Features[urutan_feature_ke1], Nama_Features[urutan_feature_ke2]])
        data['target'] = Target_Classes
        for c in Colormap.keys():
            data_filtered = data.loc[data['target']==c]
            axis[urutan_feature_ke1, urutan_feature_ke2].scatter(x = data_filtered[Nama_Features[urutan_feature_ke2]], y = data_filtered[Nama_Features[urutan_feature_ke1]], color=Colormap[c])

    # Menampilkan labels axis hanya di sisi kiri sumbu Y dan bawah sumbu X
    # mencegah label axis muncul di bagian dalam plot.
    if urutan_feature_ke1 == len(Nama_Features) - 1:
        axis[urutan_feature_ke1, urutan_feature_ke2].set(xlabel=Nama_Features[urutan_feature_ke2], ylabel='')
    if urutan_feature_ke2 == 0:
        if urutan_feature_ke1 == len(Nama_Features) - 1:
            axis[urutan_feature_ke1, urutan_feature_ke2].set(xlabel=Nama_Features[urutan_feature_ke2], ylabel=Nama_Features[urutan_feature_ke1])
        else:
            axis[urutan_feature_ke1, urutan_feature_ke2].set(xlabel='', ylabel=Nama_Features[urutan_feature_ke1])
#%% Multiplot_Berpasangan
import matplotlib.pyplot as plt
def Multiplot_Berpasangan(Data_Features, Target_Classes, Nama_Features, Colormap={0: "red", 1: "green", 2: "blue"}):
    """
    Parameters
    ----------
    Data_Features  : numpy.ndarray
    Target_Classes : numpy.ndarray
    Nama_Features  : list of string

    """
    Jumlah_Features = len(Nama_Features)
    # Buat area subplot dengan ukuran [Jumlah_Features x Jumlah_Features]
    fig, axis = plt.subplots(nrows=Jumlah_Features, ncols=Jumlah_Features)
    # Setting ukuran untuk mengoptimalkan size figure sesuai Jumlah_Features
    fig.set_size_inches(Jumlah_Features * 4, Jumlah_Features * 4)

    # Looping sejumlah features untuk plot secara berpasangan
    for i in range(0, Jumlah_Features):
        for j in range(0, Jumlah_Features):
            Singleplot_Berpasangan(axis, i, j, Data_Features, Target_Classes, Nama_Features, Colormap)
    plt.show()
#%% Plot Dataset Iris 1
import numpy as np
sepal_length = np.array(Dataset['petal length (cm)'])
sepal_width = np.array(Dataset['petal width (cm)'])
target = np.array(Dataset['target'])
colormap = np.array(['r', 'g', 'b'])
plt.scatter(sepal_length, sepal_width, c=colormap[target])

TesT_sepal_length = np.array(Xtest['petal length (cm)'])[1:2]
TesT_sepal_width = np.array(Xtest['petal width (cm)'])[1:2]
# TesT_sepal_length = np.array([2.5, 4.0])
# TesT_sepal_width = np.array([0.75, 2.0])
plt.scatter(TesT_sepal_length, TesT_sepal_width, c='violet', edgecolor='black')

plt.show()
#%% Plot Dataset Iris 1
Data_Features = np.array(Dataset[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']])
Target_Classes = np.array(Dataset[['target']])
Nama_Features = Dataset.columns # ini masih dalam format 'index'
Nama_Features = Nama_Features.tolist() # ubah ke format 'list'
Nama_Features = Nama_Features [:4] # hanya diambil 4 fitur dari 6 column dataframe
Multiplot_Berpasangan(Data_Features, Target_Classes, Nama_Features, Colormap={0: "red", 1: "green", 2: "blue"})
# Multiplot_Berpasangan(np.array(Xtest), np.array(Ytest), Nama_Features, Colormap={0: "red", 1: "green", 2: "blue"})
# Multiplot_Berpasangan(np.array(Xtrain), np.array(Ytrain), Nama_Features, Colormap={0: "red", 1: "green", 2: "blue"})
#%% Plot Tunggah

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

#%% Call code
Ytrain = Training[['target']] 
Xtrain = np.asarray (Xtrain)
Ytrain = np.asarray (Ytrain)
Xtest = np.asarray (Xtest)
Ytest = np.asarray (Ytest)
F =[]
for i in range(len(Xtest)):
    F.append(KNN(Xtrain, Ytrain, Xtest[i,:]))
Yprediksi = pd.DataFrame(F)
Akurasi = akurasi(Ytest, Yprediksi)  