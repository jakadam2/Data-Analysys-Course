import random 
import numpy as np
import math
import matplotlib.pyplot as plot


class sgd(): 

    def __init__(self):
        self._mp_good = 5
        self._mp_bad = 0.5
        self._mn = 0
        self._labels = [-1, 1]
        self._N = 160000  # numero di campioni 
        self._sigma = 1  # varianza
        self._u = 0.09 # step size
        # inizializzazione di beta 
        self._b0 = random.uniform(-3, 3)
        self._b1 = random.uniform(-3, 3)
        # preparazione del dataset 
        # data_positive = self._data_generation(self._mp, self._sigma, int(self._N/2), 1)
        # data_negative = self._data_generation(self._mn, self._sigma, int(self._N/2), -1)
        # self._dataset = np.concatenate((data_positive, data_negative), axis=0) # tutti i dati in una sola matrice
        # indices = np.arange(self._dataset.shape[0])  # shuffle del dataset 
        # np.random.shuffle(indices)
        self._dataset_good = self._data_generation(0.5, self._mp_good, self._mn, self._sigma, self._N, self._labels)
        self._dataset_bad = self._data_generation(0.5, self._mp_bad, self._mn, self._sigma, self._N, self._labels)
        self._dataset = np.concatenate((self._dataset_good, self._dataset_bad), axis=0)

    

        # test dataset 
        # print(self._dataset[103])
        # print(self._dataset[5])
        # print(self._dataset[77])
        # print(self._dataset[167])

    def sgd(self, decreasing=False): 
        if decreasing == True: 
            tau = 200
            self._u = tau
        b0s = np.ones(self._dataset.shape[0])  # inizializzazione dei vettori
        b1s = np.ones(self._dataset.shape[0])
        Js = np.ones(self._dataset.shape[0])
        for i in range(self._dataset.shape[0]): 
            sample = self._dataset[i] # estrazione di un campione 
            # aggiornamento del pesi 
            grad = self._loss_grad2(sample[0], sample[1], self._b0, self._b1)
            self._b0 = self._b0 - self._u*grad[0]    # aggiornamento 
            self._b1 = self._b1 - self._u*grad[1]
            b0s[i] = self._b0
            b1s[i] = self._b1
            # Js[i] = self._cost()
            Js[i] = 1
            if decreasing == True: 
                self._u = tau/(1+i)
        return b0s, b1s, Js

    def plot_results(self, b0s, b1s, Js):
        num_points = self._dataset.shape[0]
        plot.figure()
        plot.xlabel('b0')
        plot.ylabel('b1')
        plot.scatter(b0s, b1s, c=np.arange(num_points), cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
        plot.savefig('scatter_weights.png')

        plot.figure()
        plot.plot(range(self._dataset.shape[0]), Js)
        plot.savefig('cost.png')

        plot.figure()
        plot.plot(range(len(b0s)), b0s)
        plot.plot(range(len(b0s)), np.ones((len(b0s)))*(-self._mp_good**2/2), color='green')
        plot.plot(range(len(b0s)), np.ones((len(b0s)))*(-self._mp_bad**2/2), color='red')
        plot.xlabel('steps')
        plot.ylabel('beta_0')
        plot.legend(['beta_0', '-mp_good^2/2', '-mp_bad^2/2'])
        plot.savefig('b0.png')

        plot.figure()
        plot.plot(range(len(b1s)), b1s)
        plot.plot(range(len(b1s)), np.ones((len(b1s)))*(self._mp_good), color='green')
        plot.plot(range(len(b1s)), np.ones((len(b1s)))*(self._mp_bad), color='red')
        plot.xlabel('steps')
        plot.ylabel('beta_1')
        plot.legend(['beta_1', 'mp_good', 'mp_bad'])
        plot.savefig('b1.png')

        print('b0 = '+str(self._b0))
        print('b1 = '+str(self._b1))

    def _data_generation(slef, prior, mp, mn, sigma, N, labels):
        data = np.ones((N, 2))
        for i in range(N):
            label = random.choice([labels[0], labels[1]])
            # random_bool = random.choices([labels[0], labels[0]], weights=[prior, 1-prior], k=1)[0]
            if label == labels[0]:   # campione negativo 
                value = np.random.normal(mn, math.sqrt(sigma))
            else:   # campione positivo 
                value = np.random.normal(mp, math.sqrt(sigma))
            data[i] = [value, label]
        return data



    def _data_generation_old(self, m0, sigma, N, label): 
        data = np.zeros((N, 2)) # inzializzazione del vettore 
        data[:, 0] = np.random.normal(m0, math.sqrt(sigma), (1, N))
        # data[:, 1] = np.random.normal(m1, math.sqrt(sigma), (1, N))
        data[:, 1] = np.ones(N)*label
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

    def _loss_grad2(self, x, label, b0, b1): 
        exp = np.exp(label*(b0+x*b1))
        grad_b1 = -(label*x)/(exp+1)
        grad_b0 = -label/(exp+1)
        return [grad_b0, grad_b1]

    
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
b0s, b1s, Js = obj.sgd(False)
obj.plot_results(b0s, b1s, Js)
print("I plot sono salvati come immagini nella cartella di lavoro")
