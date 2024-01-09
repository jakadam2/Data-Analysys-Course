import random
import numpy as np
import matplotlib.pyplot as plt
import math

class TwoTranches():
    def __init__(self, nMC=100, N=100, s1=1, s2=1, theta=0):
        self._nMC = nMC
        self._N = N
        self._s1 = s1
        self._s2 = s2
        self._theta = theta
        self._t_x1 = 0
        self._t_x2 = 0
        self._t_ave = 0
        self._t_ml = 0

    def set_s1(self, s1):
        self._s1 = s1

    def set_s2(self, s2):
        self._s2 = s2

    def set_theta(self, theta):
        self._theta = theta
    
    def set_N(self, N):
        self._N = N

    def run(self):
        """Metodo che esegue il programma"""
        t_x1 = [0]*self._nMC
        t_x2 = [0]*self._nMC
        t_ave = [0]*self._nMC
        t_ml = [0]*self._nMC

        p = self._s2/(self._s2+self._s1)
        for run in range(self._nMC):
            x1, x2 = self._generate_data()
            # stimatore media aritmetica sulle x1
            t_x1[run] = self._mean(x1)
            # stimatore media aritmetica sulle x2
            t_x2[run] = self._mean(x2)
            # stimatore media aritmetica su tutto il dataset
            t_ave[run] = (t_x1[run] + t_x2[run])/2
            # stimatore ml (best)
            t_ml[run] = p*t_x1[run] + (1-p)*t_x2[run]
        
        # Media delle stime
        self._t_x1 = self._mean(t_x1)
        self._t_x2 = self._mean(t_x2)
        self._t_ave = self._mean(t_ave)
        self._t_ml = self._mean(t_ml)
        # Calcolo dell'MSE sulle stime
        self._MSE_t_x1 = self._MSE(t_x1, self._theta)
        self._MSE_t_x2 = self._MSE(t_x2, self._theta)
        self._MSE_t_ave = self._MSE(t_ave, self._theta)
        self._MSE_t_ml = self._MSE(t_ml, self._theta)

    def results(self):
        return {"t_x1" : self._t_x1, "t_x2" : self._t_x2, "t_ave" : self._t_ave, "t_ml" : self._t_ml, "MSE_t_x1" : self._MSE_t_x1, "MSE_t_x2" : self._MSE_t_x2, "MSE_t_ave" : self._MSE_t_ave, "MSE_t_ml" : self._MSE_t_ml}


    def _generate_data(self):
        """Generazione dei dati"""
        thranche_1 = np.random.normal(self._theta, math.sqrt(self._s1), int(self._N/2)).tolist()
        thranche_2 = np.random.normal(self._theta, math.sqrt(self._s2), int(self._N/2)).tolist()
        return thranche_1, thranche_2
    
    def _mean(self, x):
        """
        Input:
            - x: List       # vettore su cui fare la media
        Output:
            - mean: float   # media del vettore
        """
        return sum(x)/len(x)
    
    def _MSE(self, x, m):
        """
        Input:
            - x: List       # vettore su cui calcolare l'MSE. Ogni elemento del vettore è una stima di m
            - m: float      # valore vero
        Output:
        - MSE: float        # valore dell'MSE
        """
        mse = 0
        for i in range(len(x)):
            mse = mse + (x[i]-m)**2
        return mse/len(x)
    
    def print_results(self):
        """Mostra i risultati dopo aver eseguito il metodo run"""
        print("*** RESULTS ***")
        print()
        print("Parameters")
        print("runs MonteCarlo:", self._nMC)
        print("N:", self._N, " mean:", self._theta, " sigma1:", self._s1, " sigma2:", self._s2)
        print()
        print("Est1:", self._t_x1, " MSE:", self._MSE_t_x1)
        print("Est2:", self._t_x2, " MSE:", self._MSE_t_x2)
        print("Estav:", self._t_ave, " MSE:", self._MSE_t_ave)
        print("Estml:", self._t_ml, " MSE:", self._MSE_t_ml)
        print()
        print("***************")


def plot_results(s1, t_x1, t_x2, t_ave, t_ml, MSE_t_x1, MSE_t_x2, MSE_t_ave, MSE_t_ml):
    # # plot per le stime di theta
    # plt.title("Estimates")
    # plt.plot(s1, t_x1, label=r'$\bar{x1}$')
    # plt.plot(s1, t_x2, label=r'$\bar{x2}$')
    # plt.plot(s1, t_ave, label=r'$\bar{x}$')
    # plt.plot(s1, t_x1, label="ml")
    # plt.xlabel("s1")
    # plt.ylabel(r'$\hat{\theta}$')
    # plt.legend()
    # plt.savefig("utils/TwoTranches/Estimate.png")
    plt.figure()

    # plt.legend(labels=None)
    # plot per gli MSE
    plt.title("Estimates")
    plt.plot(s1, MSE_t_x1, label=r'$\bar{x1}$')
    plt.plot(s1, MSE_t_x2, label=r'$\bar{x2}$')
    plt.plot(s1, MSE_t_ave, label=r'$\bar{x}$')
    plt.plot(s1, MSE_t_ml, label="ml")
    plt.xlabel("s1")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("utils/TwoTranches/MSE.png")



t = TwoTranches(N=100, nMC=100)

step = 0.05
init_value = 0
end_value = 10
s1 = np.arange(init_value, end_value, step)
n = len(s1)
t_x1 = [0]*n
t_x2 = [0]*n
t_ave = [0]*n
t_ml = [0]*n
MSE_t_x1 = [0]*n
MSE_t_x2 = [0]*n
MSE_t_ave = [0]*n
MSE_t_ml = [0]*n

t = TwoTranches(N=100, s2=3)

for i in range(0, n):
    t.set_s1(s1[i])
    t.run()
    results = t.results()
    t_x1[i] = results["t_x1"]
    t_x2[i] = results["t_x2"]
    t_ave[i] = results["t_ave"]
    t_ml[i] = results["t_ml"]
    MSE_t_x1[i] = results["MSE_t_x1"]
    MSE_t_x2[i] = results["MSE_t_x2"]
    MSE_t_ave[i] = results["MSE_t_ave"]
    MSE_t_ml[i] = results["MSE_t_ml"]

plot_results(s1, t_x1, t_x2, t_ave, t_ml, MSE_t_x1, MSE_t_x2, MSE_t_ave, MSE_t_ml)