# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:57:21 2022

@author: anan
"""
#%% DATASET: Iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% INISIALISASI: Nilai K & Centroid Awal
# Memilih sejumlah K data secara random sebagai centroid awal  
def centroid_inisial(Data, K):
    Jumlah_Data  = Data.shape[0]
    Jumlah_Fitur = Data.shape[1]
    centroids = np.zeros((K, Jumlah_Fitur))
    for i in range(K):
        centroid = Data[np.random.choice(range(Jumlah_Data))]
        centroids[i] = centroid
    return centroids    
#%% CLUSTERING: berdasar kalkulasi euclidean distance
def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2, axis=1))

def clustering(Data, centroids):
    # Buat container kosong untuk list hasil hitung euclidean & K cluster
    distances = []
    clusters = [[] for _ in range(K)]

    # Hitung jarak tiap data & ditempatkan pada cluster berdasar euclidean terpendek
    for urutan_data, nilai_data in enumerate(Data):
        distance = euclidean(nilai_data,centroids)
        centroid_terdekat = np.argmin(distance)
        distances.append(distance) # Simpan data jarak euclidean
        clusters[centroid_terdekat].append(urutan_data) # Simpan cluster sesuai urutan data
    return distances, clusters
#%% PREDIKSI: kelas cluster
def prediksi(clusters, Data):
    Jumlah_Data  = Data.shape[0]
    Prediksi_kelas = np.zeros(Jumlah_Data)
    for Index, Value in enumerate(clusters):
        for urutan_data in Value:
            Prediksi_kelas[urutan_data] = Index #Array float64
    return Prediksi_kelas.astype(np.int64) #Ubah ke array int64 spt data target
#%% PLOT: Sebaran data cluster & centroid
def plotting(Data,centroids,Prediksi_kelas,i):
    colormap = np.array(['r', 'g', 'b'])
    plt.scatter(Data[:, 0], Data[:, 1], c=colormap[Prediksi_kelas], alpha = 0.3)
    plt.scatter(centroids[:, 0], centroids[:, 1], s= 70, c=colormap, edgecolor='k')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Iterasi ke i=%i' %i)
    plt.show()
#%% CENTROID: update centroid
def centroid_update(clusters, Data):
    Jumlah_Fitur = Data.shape[1]
    centroids = np.zeros((K, Jumlah_Fitur))
    for Index, Value in enumerate(clusters):
        centroid_baru = np.mean(Data[Value], axis=0)
        centroids[Index] = centroid_baru
    return centroids
#%% KMEANS
Dataset = pd.read_csv('iris.csv')
Data = Dataset[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
Data = np.asarray (Data)
# # Pemilihan centroid
# centroids = []
# centroids.append(Data[7])
# centroids.append(Data[67])
# centroids.append(Data[140])
# centroids = np.array(centroids)
# INISIALISASI
K = 3 # Jumlah cluster
centroids = centroid_inisial(Data, K)
plt.scatter(Data[:,0], Data[:,1], c='w', edgecolors='k')
colormap = np.array(['r', 'g', 'b'])
plt.scatter(centroids[:, 0], centroids[:, 1], s= 70, c=colormap, edgecolor='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Inisialisasi Centroid')
plt.show()
for i in range(100):
    # CLUSTERING
    _,clusters = clustering(Data, centroids)
    # PREDIKSI
    Prediksi_kelas = prediksi(clusters, Data)
    plotting(Data,centroids,Prediksi_kelas,i)
    # CENTROID
    centroid_lama = centroids
    centroids = centroid_update(clusters, Data)
    cek_centroid = centroids - centroid_lama
    if not cek_centroid.any():
        print("Stopping criterion terpenuhi")
        break
#%% Akurasi
def akurasi(Ytest, Yprediksi):
    akurasi = np.sum(Ytest == Yprediksi) / len(Yprediksi)
    return akurasi
#%% Visualisasi Prediksi Kmeans
target = np.array(Dataset['target'])
acc = akurasi(target,Prediksi_kelas)
acc = np.around(acc, 2)
print(acc)
colormap = np.array(['r', 'g', 'b'])
plt.scatter(Data[:, 0], Data[:, 1], c=colormap[Prediksi_kelas])
plt.xlabel('Sepa1 Length')
plt.ylabel('Sepal Width')
plt.title('KMeans Akurasi: %acc' %acc)
plt.show()
#%% Visualisasi Sebaran Data Acuan
target = np.array(Dataset['target'])
colormap = np.array(['r', 'g', 'b'])
plt.scatter(Data[:, 0], Data[:, 1], c=colormap[target])
plt.xlabel('Sepa1 Length')
plt.ylabel('Sepal Width')
plt.title('Ground Truth')
plt.show()