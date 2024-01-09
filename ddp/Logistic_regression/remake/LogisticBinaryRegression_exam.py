import random
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.special


class LogisticBinaryRegression():

    def __init__(self, nMC=10, m=[1,1], step=0.001, N=100):
        self._nMC = nMC
        self._N = N
        self._m = m
        self._n_dimension = len(m)
        self._labels = [-1, 1]
        self._alphas = np.arange(0, 1+step, step).tolist()
        self._gammas = self._gamma_calculate(self._alphas)
        self._TPR = list(range(len(self._alphas)))
        self._FPR = list(range(len(self._alphas)))
        self._one_minus_betas = self._one_minus_beta_calculate()
        

    def run(self):
        TP = [[0 for j in range(len(self._alphas))] for i in range(self._nMC)]
        TN = [[0 for j in range(len(self._alphas))] for i in range(self._nMC)]
        FP = [[0 for j in range(len(self._alphas))] for i in range(self._nMC)]
        FN = [[0 for j in range(len(self._alphas))] for i in range(self._nMC)]

        TPR = [[0 for j in range(len(self._alphas))] for i in range(self._nMC)]
        FPR = [[0 for j in range(len(self._alphas))] for i in range(self._nMC)]
        for run in range(self._nMC):
            y, x = self._generate_data(self._N)
            for k in range(self._N):
                for i in range(len(self._alphas)):
                    if y[k] == -1:
                        if self._dot_product(x[k], self._m) < self._gammas[i]:
                            TN[run][i] = TN[run][i] + 1
                        else:
                            FP[run][i] = FP[run][i] + 1
                    else:
                        if self._dot_product(x[k], self._m) > self._gammas[i]:
                            TP[run][i] = TP[run][i] + 1
                        else:
                            FN[run][i] = FN[run][i] + 1
            for i in range(len(self._alphas)):
                if FP[run][i] + TN[run][i] != 0:
                    FPR[run][i] =  FP[run][i] / (FP[run][i] + TN[run][i])
                else:
                    print("Divisione per zero nella FPR")
                    FPR[run][i] = 0
                
                if TP[run][i] + FN[run][i] != 0:
                    TPR[run][i] = TP[run][i] / (TP[run][i] + FN[run][i])
                else:
                    print("Divisione per zero nella TPR")
                    TPR[run][i] = 0
        
        for i in range(len(self._alphas)):
            fpr = 0
            tpr = 0
            for run in range(self._nMC):
                fpr = fpr + FPR[run][i]
                tpr = tpr + TPR[run][i]
            self._FPR[i] = fpr/(run+1)
            self._TPR[i] = tpr/(run+1)




    # def run(self):
    #     TP = [0] * len(self._alphas)
    #     TN = [0] * len(self._alphas)
    #     FP = [0] * len(self._alphas)
    #     FN = [0] * len(self._alphas)
    #     for run in range(self._nMC):
    #         y, x = self._generate_data()
    #         for i in range(len(self._alphas)):
                
    #             if y == -1:
    #                 if self._dot_product(x, self._m) < self._gammas[i]:
    #                     TN[i] = TN[i] + 1
    #                 else:
    #                     FP[i] = FP[i] + 1
    #             else:
    #                 if self._dot_product(x, self._m) > self._gammas[i]:
    #                     TP[i] = TP[i] + 1
    #                 else:
    #                     FN[i] = FN[i] + 1

    #     for i in range(len(self._alphas)):
    #         self._FPR[i] =  FP[i] / (FP[i] + TN[i])

    #         self._TPR[i] = TP[i] / (TP[i] + FN[i])

    def plot_results(self):
        plt.plot(self._FPR, self._TPR, label="estimated ROC")
        plt.plot(self._alphas, self._one_minus_betas, label="true ROC")
        plt.title("Curva ROC")
        plt.show()
    
    def _dot_product(self, x1, x2):
        value = 0
        for i in range(len(x1)):
            value = value + x1[i]*x2[i]
        return value


    def _generate_data(self, N):
        x = [[0 for j in range(self._n_dimension)] for i in range(N)]
        label = [0]*N
        # m_0 è m quando la label è uguale a -1
        m_0 = [0] * self._n_dimension
        for n in range(N):
            label[n] = random.choice(self._labels)
            if label[n] == 1:
                m = self._m
            else:
                m = m_0
            for i in range(self._n_dimension):
                x[n][i] = np.random.normal(m[i], 1)
        return label, x


    # def _generate_data(self):
    #     label = random.choice(self._labels)
    #     x = list(range(self._n_dimension))
    #     if label == 1:
    #         m = self._m
    #     else:
    #         m = [0] * self._n_dimension
    #     for i in range(self._n_dimension):
    #         x[i] = np.random.normal(m[i], 1)
    #     return label, x
    

    def _gamma_calculate(self, alphas):
        gammas = list(range(len(alphas)))
        norm_m = self._norm(self._m)
        const = (norm_m**2)/2
        for i in range(len(alphas)):
            gammas[i] = self._inverse_Q_function(alphas[i])*norm_m + const
        return gammas
    
    def _one_minus_beta_calculate(self):
        betas = list(range(len(self._alphas)))
        for i in range(len(self._alphas)):
            betas[i] = self._Q_function((self._inverse_Q_function(self._alphas[i]) - self._norm(self._m)))
        return betas

    def _inverse_Q_function(self, probability):
        return norm.ppf(1 - probability)
    
    def _Q_function(self, x):
        return 0.5 * scipy.special.erfc(x / (2**0.5))

    def _norm(self, values):
        return math.sqrt(sum(x**2 for x in values))
    

l = LogisticBinaryRegression(nMC=100, m=[1, 1, 1], N=100)
l.run()
l.plot_results()
