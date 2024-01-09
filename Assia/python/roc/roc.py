import random 
import numpy as np
import math
import matplotlib.pyplot as plot
from scipy.stats import norm



class roc(): 
    def __init__(self, N, nMC):
        self._mp = [0, 0]
        self._mn = [1, 1]
        self._N = N  # numero di campioni 
        self._sigma = 0.5  # varianza
        self._nMC = nMC
        # preparazione del dataset
        # self._dataset = self._dataset_generation()                                                                                                                                                                                             


    def _data_generation(self, m0, m1, sigma, N, label): 
        data = np.zeros((N, 3)) # inzializzazione del vettore 
        data[:, 0] = np.random.normal(m0, math.sqrt(sigma), (1, N))
        data[:, 1] = np.random.normal(m1, math.sqrt(sigma), (1, N))
        data[:, 2] = np.ones(N)*label
        return data

    def _dataset_generation(self):
        data_positive = self._data_generation(self._mp[0], self._mp[1], self._sigma, int(self._N/2), 1)
        data_negative = self._data_generation(self._mn[0], self._mn[1], self._sigma, int(self._N/2), -1)
        dataset = np.concatenate((data_positive, data_negative), axis=0) # tutti i dati in una sola matrice
        indices = np.arange(dataset.shape[0])  # shuffle del dataset 
        np.random.shuffle(indices)
        dataset = dataset[indices]  
        # print(dataset.shape)
        return dataset
    
    def _likelihood_ratio(self, x0, x1): 
        lpx0 = norm.pdf(x0, self._mp[0], math.sqrt(self._sigma))  # likelihood of being positive
        lpx1 = norm.pdf(x1, self._mp[1],math.sqrt(self._sigma))
        lp = lpx0 * lpx1  # per distrubuzioni indipendenti la congiunta è il prodotto delle marignali 

        lnx0 = norm.pdf(x0, self._mn[0], math.sqrt(self._sigma))  # likelihood of being negative
        lnx1 = norm.pdf(x1, self._mn[1],math.sqrt(self._sigma))
        ln = lnx0 * lnx1

        ratio = lp/ln
        return ratio

    def _get_gammas(self):
        first_part = np.linspace(0, 1, 10)
        second_part = np.linspace(1, 100, 99)
        gammas = np.concatenate((first_part, second_part))
        return gammas
    
    def _np_test(self, x0, x1, gamma): 
        res = 0
        ratio = self._likelihood_ratio(x0, x1)
        if ratio >= gamma:
            res = 1
        else:
            res = -1
        return res

    def _roc_points(self, dataset, gammas):
        # gammas = self._get_gammas()
        points = np.ones((len(gammas), 2))
        for j in range(len(gammas)):
            gamma = gammas[j]
            fp = 0  # number of false positives
            tp = 0  # number of true positives
            for i in range(dataset.shape[0]):
                sample = dataset[i]
                res = self._np_test(sample[0], sample[1], gamma)   # CLASSIFICAZIONE
                if sample[2]==1 and res==1:
                    tp = tp + 1
                if sample[2]==-1 and res==1:
                    fp = fp + 1
            points[j] = [fp, tp]

        # plot.figure()
        # plot.plot(points[:, 0], points[:, 1])
        # plot.savefig('roc.png')
        return points
    
    def montecarlo(self):
        gammas = self._get_gammas()
        tp = np.ones((self._nMC, len(gammas)))
        fp = np.ones((self._nMC, len(gammas)))
        for i in range(self._nMC):
            dataset = self._dataset_generation()
            points = self._roc_points(dataset, gammas)
            fp[i] = points[:, 0]
            tp[i] = points[:, 1]
            print('Montecarlo run '+str(i+1))
        len_dataset = dataset.shape[0]
        mean_fp = np.sum(fp, axis=0) / len_dataset
        mean_tp = np.sum(tp, axis=0) / len_dataset
        print('Complete')
        return [mean_fp, mean_tp]
    
    def plot_result(self, fp, tp):
        plot.figure()
        plot.plot(fp, tp)
        plot.xlabel('False positives')
        plot.ylabel('True positives')
        plot.title('ROC curve')
        plot.savefig('roc.png')
        print('Graifco salvato nella directory di lavoro')


N = 150
nMC = 5
obj = roc(N, nMC)
# dataset = obj._dataset_generation()
# gammas = obj._get_gammas()
fp, tp = obj.montecarlo()
obj.plot_result(fp, tp)








