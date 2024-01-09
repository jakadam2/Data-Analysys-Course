import random 
import numpy as np
import math
import matplotlib.pyplot as plot


class sgd(): 

    def __init__(self):
        self._mp = [0, 0]
        self._mn = [1, 1]
        self._N = 200  # numero di campioni 
        self._sigma = 0.5  # varianza
        self._u = 0.2   # step size
        # inizializzazione di beta 
        self._b0 = random.uniform(-3, 3)
        self._b1 = random.uniform(-3, 3)
        # preparazione del dataset 
        data_positive = self._data_generation(self._mp[0], self._mp[1], self._sigma, int(self._N/2), 1)
        data_negative = self._data_generation(self._mn[0], self._mn[1], self._sigma, int(self._N/2), -1)
        self._dataset = np.concatenate((data_positive, data_negative), axis=0) # tutti i dati in una sola matrice
        indices = np.arange(self._dataset.shape[0])  # shuffle del dataset 
        np.random.shuffle(indices)
        self._dataset = self._dataset[indices]

        # test dataset 
        # print(self._dataset[103])
        # print(self._dataset[5])
        # print(self._dataset[77])
        # print(self._dataset[167])

    def sgd(self, decreasing=False): 
        if decreasing == True: 
            tau = 500
            self._u = tau
        b0s = np.ones(self._dataset.shape[0])
        b1s = np.ones(self._dataset.shape[0])
        Js = np.ones(self._dataset.shape[0])
        for i in range(self._dataset.shape[0]): 
            sample = self._dataset[i] # estrazione di un campione 
            # aggiornamento del pesi 
            grad = self._loss_grad(sample[0], sample[1], sample[2], self._b0, self._b1)
            self._b0 = self._b0 - self._u*grad[0]
            self._b1 = self._b1 - self._u*grad[1]
            b0s[i] = self._b0
            b1s[i] = self._b1
            Js[i] = self._cost()
            if decreasing == True: 
                self._u = tau/(1+i)
        return b0s, b1s, Js

    def plot_results(self, b0s, b1s, Js):
        num_points = self._dataset.shape[0]
        plot.figure()
        plot.scatter(b0s, b1s, c=np.arange(num_points), cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
        plot.savefig('scatter_weights.png')

        plot.figure()
        plot.plot(range(self._dataset.shape[0]), Js)
        plot.savefig('cost.png')



    def _data_generation(self, m0, m1, sigma, N, label): 
        data = np.zeros((N, 3)) # inzializzazione del vettore 
        data[:, 0] = np.random.normal(m0, math.sqrt(sigma), (1, N))
        data[:, 1] = np.random.normal(m1, math.sqrt(sigma), (1, N))
        data[:, 2] = np.ones(N)*label
        # data[:, 2] = np.round(data[:, 2]).astype(int)
        # print(data)
        return data

    def _loss_grad(self, x0, x1, label, b0, b1): 
        '''Fa il gradiente della loss su un campione'''
        exponent = label*x0*b0 + label*x1*b1
        factor = -label/(np.exp(exponent)+1)
        grad0 = factor*x0
        grad1 = factor*x1
        return [grad0, grad1]
    
    def _loss(self, x0, x1, label, b0, b1): 
        exponent = label*x0*b0 + label*x1*b1  # y*x^T*b
        loss = np.log(1+np.exp(-1*exponent))
        return loss


    def _cost(self): 
        '''calcola il costo coni valori attuali'''
        J = 0
        for i in range(self._dataset.shape[0]):
            sample = self._dataset[i]
            J += self._loss(sample[0], sample[1], sample[2], self._b0, self._b1)
        
        J = J/(self._dataset.shape[0])  # media della loss (valore atteso)
        return J

obj = sgd()
b0s, b1s, Js = obj.sgd(True)
obj.plot_results(b0s, b1s, Js)
print("I plot sono salvati come immagini nella cartella di lavoro")
