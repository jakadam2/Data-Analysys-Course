import random
import numpy as np
import matplotlib.pyplot as plt
import math

class TwoTranches():
    
    def __init__(self, nMC=1000, N=20, s1=1, s2=1, theta=0):
        self._nMC = nMC
        if N % 2 == 1:
            N = N + 1
        self._N = N
        self._s1 = s1
        self._s2 = s2
        self._theta = theta
        self._av_t1 = 0
        self._av_t2 = 0
        self._av_tav = 0
        self._av_tml = 0
        self._MSE_t1 = 0
        self._MSE_t2 = 0
        self._MSE_tav = 0
        self._MSE_tml = 0


    def set_s1(self, s1):
        self._s1 = s1

    def set_s2(self, s2):
        self._s2 = s2

    def set_theta(self, theta):
        self._theta = theta
    
    def set_N(self, N):
        self._N = N

        
    def run(self):
        p = self._s2/(self._s2+self._s1)
        est_1 = list(range(self._nMC))
        est_2 = list(range(self._nMC))
        est_av = list(range(self._nMC))
        est_ml = list(range(self._nMC))

        for i in range(0, self._nMC):
            data = self.generate_two_tranches()
            first = data["first"]
            second = data["second"]

            est_1[i] = self._mean(first)
            est_2[i] = self._mean(second)
            est_av[i] = (est_1[i] + est_2[i])/2
            est_ml[i] = est_1[i]*p + est_2[i]*(1-p)
        
        self._av_t1 = self._mean(est_1)
        self._av_t2 = self._mean(est_2)
        self._av_tav = self._mean(est_av)
        self._av_tml = self._mean(est_ml)

        self._MSE_t1 = self._MSE(est_1, self._theta)
        self._MSE_t2 = self._MSE(est_2, self._theta)
        self._MSE_tav = self._MSE(est_av, self._theta)
        self._MSE_tml = self._MSE(est_ml, self._theta)

    def print_results(self):
        print("*** RESULTS ***")
        print()
        print("Parameters")
        print("runs MonteCarlo:", self._nMC)
        print("N:", self._N, " mean:", self._theta, " sigma1:", self._s1, " sigma2:", self._s2)
        print()
        print("Est1:", self._av_t1, " MSE:", self._MSE_t1)
        print("Est2:", self._av_t2, " MSE:", self._MSE_t2)
        print("Estav:", self._av_tav, " MSE:", self._MSE_tav)
        print("Estml:", self._av_tml, " MSE:", self._MSE_tml)
        print()
        print("***************")
        

    def results(self):
        return {"t1" : self._av_t1, "t2" : self._av_t2, "tav" : self._av_tav, "tml" : self._av_tml, "MSE1" : self._MSE_t1, "MSE2" : self._MSE_t2, "MSEav" : self._MSE_tav, "MSEml" : self._MSE_tml}

    def generate_two_tranches(self):
        data = {"first": [], "second": []}
        data["first"] = self._generate_normal_data(self._theta, math.sqrt(self._s1), int(self._N/2))
        data["second"] = self._generate_normal_data(self._theta, math.sqrt(self._s2), int(self._N/2))
        return data

    def _generate_normal_data(self, mean, std, N):
        return np.random.normal(mean, std, N)
    
    def _mean(self, data):
        return sum(data)/len(data)
    
    def _MSE(self, data, mean):
        values = 0
        for i in range(0,len(data)):
            values = values + ((data[i] - mean)**2)
        return values/len(data)
    

    def plot_results(self, s1, av_t1, av_t2, av_tav, av_tml, MSE_t1, MSE_t2, MSE_tav, MSE_tml):
        ### Estimates ###
            
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].plot(s1, av_t1, label="t1", color="violet")
        axs[0, 0].set_title('t1')
        axs[0, 0].set_xlabel("s1")
        axs[0, 0].set_ylabel("t1")
        axs[0, 0].legend()

        axs[0, 1].plot(s1, av_t2, label="t2", color="red")
        axs[0, 1].set_title('t2')
        axs[0, 1].set_xlabel("s1")
        axs[0, 1].set_ylabel("t2")
        axs[0, 1].legend()

        axs[1, 0].plot(s1, av_tav, label="tav", color="green")
        axs[1, 0].set_title('tav')
        axs[1, 0].set_xlabel("s1")
        axs[1, 0].set_ylabel("tav")
        axs[1, 0].legend()

        axs[1, 1].plot(s1, av_tml, label="tml", color="blue")
        axs[1, 1].set_title('tml')
        axs[1, 1].set_xlabel("s1")
        axs[1, 1].set_ylabel("tml")
        axs[1, 1].legend()

        plt.tight_layout()

        plt.show()


        ### MSE ###


        plt.plot(s1, MSE_t1, label="MSE_t1", color="violet")
        plt.title('MSE_t1')
        plt.xlabel("s1")
        plt.ylabel("MSE_t1")
        plt.legend()

        plt.plot(s1, MSE_t2, label="MSE_t2", color="red")
        plt.title('MSE_t2')
        plt.xlabel("s1")
        plt.ylabel("MSE_t2")
        plt.legend()

        plt.plot(s1, MSE_tav, label="MSE_tav", color="green")
        plt.title('MSE_tav')
        plt.xlabel("s1")
        plt.ylabel("MSE_tav")
        plt.legend()

        plt.plot(s1, MSE_tml, label="MSE_tml", color="blue")
        plt.title('MSE_tml')
        plt.xlabel("s1")
        plt.ylabel("MSE_tml")
        plt.legend()

        plt.tight_layout()

        plt.show()



### TWO TRANCHES ###
step = 0.05
init_value = 0
end_value = 5
s1 = np.arange(init_value, end_value, step)
n = len(s1)
av_t1 = list(range(n))
av_t2 = list(range(n))
av_tav = list(range(n))
av_tml = list(range(n))
MSE_t1 = list(range(n))
MSE_t2 = list(range(n))
MSE_tav = list(range(n))
MSE_tml = list(range(n))

t = TwoTranches(N=5, s2=1, theta=3)

for i in range(0, n):
    t.set_s1(s1[i])
    t.run()
    results = t.results()
    av_t1[i] = results["t1"]
    av_t2[i] = results["t2"]
    av_tav[i] = results["tav"]
    av_tml[i] = results["tml"]
    MSE_t1[i] = results["MSE1"]
    MSE_t2[i] = results["MSE2"]
    MSE_tav[i] = results["MSEav"]
    MSE_tml[i] = results["MSEml"]


t.plot_results(s1, av_t1, av_t2, av_tav, av_tml, MSE_t1, MSE_t2, MSE_tav, MSE_tml)
