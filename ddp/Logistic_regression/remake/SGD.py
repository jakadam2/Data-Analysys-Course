import math
import random
import numpy as np
import matplotlib.pyplot as plt

class SGD():
    
    def __init__(self, m=[1,1], labels=[-1, 1], prior=[1/2, 1/2], nMC=1000, learning_rate = 0.01, decay=False, step_stop=10):
        self._m = m
        self._labels = labels
        self._nMC = nMC
        self._prior = prior
        self._best_beta = [(-self._norm(m)**2)/2 + math.log(prior[1]/prior[0])] + m
        self._learning_rate = learning_rate
        self._decay = decay
        self._step_stop = step_stop
        self._beta = None
        self._J = [None] * nMC
        self._Q = [None] * nMC
        self._MSE = [None] * nMC
        

    def run(self):
        stop = 0
        # beta = list(range(len(self._m) + 1))
        x_J = list(range(self._nMC))
        y_J = list(range(self._nMC))
        # for i in range(len(beta)):
        #     beta[i] = random.random()
        beta =[random.random()] + [0] * len(self._m)
        learning_rate = self._learning_rate
        for i in range(self._nMC):
            y, x = self._new_sample()
            x_J[i] = x
            y_J[i] = y
            grad = self._calculate_gradient(y, x, beta)
            
            beta = self._update_beta(beta, grad, learning_rate)
            #self._J[i] = self._calculate_J(i, y, x, beta)
            self._J[i] = self._calculate_J(i, y_J, x_J, beta)
            self._Q[i] = math.log(1 + math.exp(-y*self._dot_product(x, beta)))
            #self._MSE[i] = self._calculate_MSE(beta)
            if i!=0 and (abs(self._J[i]-self._J[i-1])<=0.0000001):
                stop = stop+1
                if stop >= self._step_stop:
                    print("Stop at iteretion number", i)
                    break
            else:
                stop = 0
            if self._decay:
                learning_rate = self._learning_rate/(i+1)
        self._beta = beta
        

    def plot_costs(self):
        plt.plot(list(range(self._nMC)), self._J, label="J")
        # plt.plot(list(range(self._nMC)), self._Q, label="Q")
        #plt.plot(list(range(self._nMC)), self._MSE, label="MSE")

        plt.legend()
        plt.show()

    def test_beta(self, beta=None, n=100):
        if beta == None:
            beta = self._beta
        
        x0 = np.linspace(-5, 5, 100)
        x1 = (-beta[0] - beta[1] * x0) / beta[2]
        plt.plot(x0, x1, label="estimated beta")
        plt.legend()

        x1 = (-self._best_beta[0] - self._best_beta[1] * x0) / self._best_beta[2]
        plt.plot(x0, x1, label="best beta")
        plt.legend()

        TP = []
        TN = []
        FN = []
        FP = []
        for i in range(n):
            y, x = self._new_sample()
            f_x = 0
            for k in range(len(x)):
                f_x = f_x + beta[k]*x[k]
            if y == 1:
                if f_x > 0:
                    TP.append([x[1], x[2]])
                else:
                    FN.append([x[1], x[2]])
            else:
                if f_x > 0:
                    FP.append([x[1], x[2]])
                else:
                    TN.append([x[1], x[2]])

        self._scatter_points(TN, "black", "TN")
        self._scatter_points(TP, "green", "TP")
        self._scatter_points(FP, "red", "FP")
        self._scatter_points(FN, "blue", "FN")

        plt.show()

    def _scatter_points(self, points, color, label):
        x0 = [point[0] for point in points]
        x1 = [point[1] for point in points]
        plt.scatter(x0, x1, color=color, label=label)
        plt.legend()


    def _calculate_J(self, run, y_J, x_J, beta):
        values = 0
        
        for i in range(run+1):
            values = values + math.log(1 + math.exp(-y_J[i]*self._dot_product(x_J[i], beta)))
        return values/(run+1)


    # def _calculate_J(self, run, y, x, beta):
    #     actual_J = math.log(1 + math.exp(-y*self._dot_product(x, beta)))
    #     if run != 0:
    #         # In self._J[run-1] ho la media aritmetica dei costi fino a run-1.
    #         # Se moltiplico la media per run, ottengo la somma dei logaritmi delle iterazioni passate
    #         past_J = self._J[run-1]*run
    #         # Il costo attuali è pari alla media aritmetica della somma dei logaritmi,
    #         # dove la somma dei logaritmi è pari alla somma dei logaritmi delle iterazioni passate
    #         # più il logaritmo attuale.
    #         # Faccio questo solo per un motivo computazionale, altrimenti avrei dovuto salvarmi i costi a ogni iterazione
    #         # e calcolare la media ogni volta sui costi -> è inefficente fra
    #         actual_J = (actual_J + past_J)/(run+1)
            
    #         return actual_J
    #     else:
    #         return actual_J

    def _calculate_gradient(self, y, x, beta):
        const = 1/(1+math.exp(y*self._dot_product(x, beta)))
        grad = list(range(len(x)))
        for i in range(len(grad)):
            grad[i] = -y*x[i]*const
        return grad

    def _update_beta(self, beta, grad, learning_rate):
        for i in range(len(beta)):
            beta[i] = beta[i] - learning_rate*grad[i]
        return beta

    def _new_sample(self):
        label = random.choice(self._labels)
        x = list(range(len(self._m)))
        if label == 1:
            m = self._m
        else:
            m = [0] * len(self._m)
        for i in range(len(self._m)):
            x[i] = np.random.normal(m[i], 1)
        x = [1] + x
        return label, x
    
    def _dot_product(self, x1, x2):
        value = 0
        for i in range(len(x1)):
            value = value + x1[i]*x2[i]
        return value

    def _norm(self, values):
        return math.sqrt(sum(x**2 for x in values))
    
    # def _calculate_MSE(self, beta):
    #     err = list(range(len(beta)))
    #     for i in range(len(beta)):
    #         err[i] = beta[i] - self._best_beta[i]
    #     print(err)
    #     return self._norm(err)**2


s = SGD(nMC=100000, m=[5, 5], step_stop=10, decay=True, learning_rate=10)
s.run()
s.plot_costs()
s.test_beta()