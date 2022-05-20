# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:26:27 2022

@author: anan
"""
#%% Header Libraries
import matplotlib.pyplot as plt
import pandas as pd
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