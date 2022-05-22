import pandas as pd
Dataset = pd.read_csv('iris.csv')
#%% Data Testing
Testing = Dataset[0:10] # Setosa
Testing = Testing.append(Dataset[50:60]) # Versicolor
Testing = Testing.append(Dataset[100:110]) # Virginica
Xtest = Testing[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
Ytest = Testing[['target']]
#%% Data Training
Training = Dataset[10:50] # Setosa
Training = Training.append(Dataset[60:100]) # Versicolor
Training = Training.append(Dataset[110:150]) # Virginica
Xtrain = Training[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
Ytrain = Training[['target']]
#%% Euclidean
import numpy as np

def euclidean(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
#%% KNN
def KNN(Xtrain,Ytrain,Xtest, K = 4):
    m= Xtrain.shape[0]
    data = []
    for i in range(m):
        xi =Xtrain[i]
        distance = euclidean(Xtest, xi)
        data.append((distance,Ytrain[i]))
    data = sorted(data, key=lambda x:x[0])[:K]   
    data = np.asarray (data)
    data_baru = np.unique(data[:,1], return_counts=True)
    index = data_baru[1].argmax()
    output = data_baru[0][index]
    return output  
#%% Akurasi
def akurasi(Ytest,prediksi):
  akurasi = np.sum(Ytest == prediksi)/len(prediksi)*100
  return akurasi
#%% Panggil KNN
# Xtrain = np.asarray (Xtrain)
# Ytrain = np.asarray (Ytrain)
# Xtest = np.asarray (Xtest)
# Ytest = np.asarray (Ytest)
prediksi = []
for i in range(len(Xtest)):
    prediksi.append(KNN(Xtrain, Ytrain, Xtest[i,:]))
# %% Hitung Akurasi
Yprediksi = pd.DataFrame(prediksi)
Akurasi = akurasi(Ytest,Yprediksi)