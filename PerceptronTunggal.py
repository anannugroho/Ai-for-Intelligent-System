import numpy as np

class Perceptron:

     def __init__(self, learning_rate = 0.01, iterasi = 1000):
        self.alfa = learning_rate
        self.iter = iterasi
        self.weights = None
        self.bias = None
        self.fungsi_aktivasi = self.fungsi_aktivasi_step

     def prediksi(self, X):
        z = np.dot(X, self.weights) + self.bias
        Yp = self.fungsi_aktivasi(z)
        return Yp

     def fungsi_aktivasi_step (self, z):
       return  np.where(z >= 0, 1, 0)

     def train(self, X, target):
        n_sample, n_features = X.shape

        self.bias = 0
        self.weights = np.zeros(n_features)

        Yt = target

        for _ in range(self.iter):
            for idx, x_i in enumerate (X):
                z = np.dot(x_i, self.weights) + self.bias
                Yp = self.fungsi_aktivasi(z)

                delta_w = self.alfa * (Yt[idx] - Yp)
                self.weights += delta_w * x_i 
                self.bias += delta_w