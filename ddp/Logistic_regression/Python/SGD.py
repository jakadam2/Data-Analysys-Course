import random
import numpy as np
import math
import matplotlib.pyplot as plt

class Logistic_regression():

    def __init__(self, nMC = 100000 , m = [1, 1], sigma = 1, learning_rate = 0.01, labels = [-1, 1], decay=False):
        self._nMC = nMC
        self._m = m
        self._sigma = sigma
        self._learning_rate = learning_rate
        self._beta = [random.uniform(0, 1), 0, 0]
        self._labels = labels
        self._best_beta = [-(self._norm(m)**2)/2] + m
        self._decay = decay
        self._errors = list(range(nMC))
        self._cost = list(range(nMC))
        self._real_cost = list(range(nMC))
        print("Best beta: ", self._best_beta)

    def _norm(self, values):
        return math.sqrt(sum(x**2 for x in values))

    def SGD(self):
        cost = 0
        real_cost = 0
        err_beta = list(range(len(self._beta)))
        for i in range(0,self._nMC):


            # generate data
            label = random.choice(self._labels)
            x = self._generate_data(label)

            # calculate gradient
            grad = self._gradient(x, label, self._beta)

            # update beta
            if self._decay:
                for j in range(0,len(self._beta)):
                    self._beta[j] = self._beta[j] - (self._learning_rate/(i+1))*grad[j]
                    err_beta[j] = self._beta[j]-self._best_beta[j]
            else:
                for j in range(0,len(self._beta)):
                    self._beta[j] = self._beta[j] - self._learning_rate*grad[j]
                    err_beta[j] = self._beta[j]-self._best_beta[j]
            
            # calculate error
            self._errors[i] = self._loss_function(x, label, self._beta)

            # calculate cost
            cost = cost + self._errors[i]
            self._cost[i] = cost/(i+1)
            real_cost = real_cost + (self._norm(err_beta)**2)
            self._real_cost[i] = real_cost/(i+1)

        return self._beta


    def _generate_data(self, label):
        m = []
        if label == -1:
            m = [0, 0]
        else:
            m = self._m

        x0 = np.random.normal(m[0], self._sigma, 1)[0]
        x1 = np.random.normal(m[1], self._sigma, 1)[0]

        return [1, x0, x1]
    
    def _loss_function(self, x, y, beta):
        x_by_beta= 0
        for i in range(0, len(x)):
            x_by_beta = x_by_beta + x[i]*beta[i]
        return math.log(1 + math.exp(-y*x_by_beta))

    def _gradient(self, x, y, beta):
        x_by_beta= 0
        for i in range(0, len(x)):
            
            x_by_beta = x_by_beta + x[i]*beta[i]
        const = -y*math.exp(-y*x_by_beta)/(1 + math.exp(-y*x_by_beta))
        grad = [0, 0, 0]
        for j in range(0, len(x)):
            grad[j] = x[j]*const
        return grad
        
    def LR(self, N, beta=None):
        if beta == None:
            beta = self._beta
        dataset = self._generate_dataset(N)
        TP = []
        TN = []
        FN = []
        FP = []
        for i in range(0, N):
            x0 = dataset[i]["x"][0]
            x1 = dataset[i]["x"][1]
            f_x = beta[0] + beta[1]*x0 + beta[2]*x1
            if dataset[i]["y"] == -1:
                if f_x < 0:
                    TN.append([x0, x1])
                else:
                    FP.append([x0, x1])
            else:
                if f_x > 0:
                    TP.append([x0, x1])
                else:
                    FN.append([x0, x1])

        self._scatter_points(TN, "black", "TN")
        self._scatter_points(TP, "green", "TP")
        self._scatter_points(FP, "red", "FP")
        self._scatter_points(FN, "blue", "FN")

        x0 = np.linspace(-5, 5, 100)
        x1 = (-beta[0] - beta[1] * x0) / beta[2]
        plt.plot(x0, x1, label="estimated beta")
        plt.legend()

        x1 = (-self._best_beta[0] - self._best_beta[1] * x0) / self._best_beta[2]
        plt.plot(x0, x1, label="best beta")
        plt.legend()

        plt.xlabel(r'$\mathit{x}_0$')
        plt.ylabel(r'$\mathit{x}_1$')

        title = r'LBR con $N={}$, $m_0=(0,0)$ e $m_1=({}, {}),\ nMC={}$'.format(N, self._m[0], self._m[1], self._nMC)
        plt.title(title)
        
        plt.show()

    def _scatter_points(self, points, color, label):
        x0 = [point[0] for point in points]
        x1 = [point[1] for point in points]
        plt.scatter(x0, x1, color=color, label=label)
        plt.legend()
        
    def _generate_dataset(self, N):
        dataset = list(range(N))
        for i in range(0, N):
            label = random.choice(self._labels)
            _, x0, x1 = self._generate_data(label)
            dataset[i] = {"y":label, "x":[x0, x1]}
        return dataset
            
    def plot_error(self):
        plt.plot(range(0, self._nMC), self._errors, linestyle='-', label="Errors")
        plt.legend()
        plt.show()
    
    def plot_cost(self):
        plt.plot(range(0, self._nMC), self._cost, linestyle='-', label="Estimated cost")
        plt.plot(range(0, self._nMC), self._real_cost, linestyle='-', label="MSE")
        plt.legend()
        plt.show()
