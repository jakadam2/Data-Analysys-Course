import numpy as np
import matplotlib.pyplot as plt
import math

class ApproximationRegression():

    def __init__(self, nMC=100, N=10000, a=3):
        self._nMC = nMC
        self._N = N
        self._a = a
        law = math.sqrt(N)
        self._kn = math.floor(law)
        self._h = 1/law
        self._MSE_KNN = None
        self._MSE_NK = None

    def set_N(self, N):
        self._N = N
        law = math.sqrt(N)
        self._kn = math.floor(law)
        self._h = 1/law

    def run(self):
        self.run_KNN()
        self.run_NK()

    def results(self):
        return {"MSE_KNN":self._MSE_KNN, "MSE_NK":self._MSE_NK}


    #aggiusta i bordi
    # peova a fare x_KNN[k] = x[i+kn/2]
    def run_KNN(self, y=None, x=None):
        MSE_run = list(range(self._nMC))
        for run in range(self._nMC):
            if y == None:
                y, x = self._generate_data()
            x_KNN = list(range(math.floor(self._N/self._kn)))
            y_KNN = list(range(math.floor(self._N/self._kn)))
            i = 0
            k = 0
            MSE = 0
            while i < self._N:
                x_KNN[k] = x[i]
                values = y[i:i+self._kn]
                i = i+self._kn + 1
                y_KNN[k] = self._mean(values)
                MSE = MSE + (y_KNN[k] - np.sin(2 * np.pi * x_KNN[k]))**2
                k = k + 1
            MSE_run[run] = MSE/k

        self._MSE_KNN = self._mean(MSE_run)


    def run_NK(self, y=None, x=None):
        MSE_run = list(range(self._nMC))
        for run in range(self._nMC):
            if y == None:
                y, x = self._generate_data()
            n_intervals = math.floor(self._a/self._h)
            x_NK = list(range(n_intervals))
            y_NK = list(range(n_intervals))
            k = 0
            MSE = 0
            for i in range(n_intervals):
                x_NK[i] = self._h*(1/2 + i)
                j = 0
                values = 0
                while x[k] <= self._h*(i+1):
                    values = values + y[k]
                    j = j + 1
                    k = k + 1
                    if k >= self._N:
                        break
                if j != 0:
                    y_NK[i] = values/j
                else:
                    y_NK[i] = 0
                MSE = MSE + (y_NK[i] - np.sin(2 * np.pi * x_NK[i]))**2
                if k>=self._N:
                    break
            MSE_run[run] = MSE/n_intervals
        
        self._MSE_NK = self._mean(MSE_run)
            
    def print_results(self):
        print("*** RESULTS ***")
        print()
        print("Parameters")
        print("runs MonteCarlo:", self._nMC)
        print("N:", self._N, " a:", self._a)
        print()
        print("MSE KNN:", self._MSE_KNN)
        print("MSE NK:", self._MSE_NK)
        print()
        print("***************")
        
    def _generate_data(self):
        x = np.random.uniform(0, self._a, self._N).tolist()
        x.sort()
        errors = np.random.randn(self._N)
        y = list(range(self._N))
        for i in range(0, len(x)):
            y[i] = np.sin(2 * np.pi * x[i]) + errors[i]
        return y, x

    def _mean(self, data):
        if not isinstance(data, list):
            return data
        values = 0
        for i in range(0, len(data)):
            values = values + data[i]
        return values/len(data)
    
    def plot_results(self, run, MSE_KNN, MSE_NK):
        plt.plot(run, MSE_KNN, label="KNN")
        plt.plot(run, MSE_NK, label="NK")
        plt.legend()
        plt.show()
