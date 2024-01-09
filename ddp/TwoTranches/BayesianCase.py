import random
import numpy as np
import matplotlib.pyplot as plt
import math

class BayesianCase():

    def __init__(self, nMC=100, N=10, sy=1, sw=1):
        self._nMC = nMC
        self._N = N
        self._sy = sy
        self._sw = sw

        self._MSE_tml = 0
        self._MSE_tbayes = 0

        self._th_err = (sy*sw/N)/(sy+sw/N)


    def run(self):
        est_bayes = list(range(self._nMC))
        est_ml = list(range(self._nMC))
        y = list(range(self._nMC))
        a = self._sy/(self._N*self._sy + self._sw)
        for i in range(0, self._nMC):
            y[i] = np.random.normal(0, math.sqrt(self._sy), 1)[0]
            x = np.random.normal(y[i], math.sqrt(self._sw), self._N)
            sum_x = sum(x)
            est_ml[i] = sum_x/self._N
            est_bayes[i] = sum_x*a

        self._MSE_tml = self._MSE(est_ml, y)
        self._MSE_tbayes = self._MSE(est_bayes, y)
        
    def get_th_err(self):
        return self._th_err
    
    def get_sy(self):
        return self._sy
    
    def get_sw(self):
        return self._sw
    
    def get_N(self):
        return self._N

    def set_sy(self, sy):
        self._sy = sy
        self._th_err = (self._sy*self._sw/self._N)/(self._sy+self._sw/self._N)


    def set_sw(self, sw):
        self._sw = sw
        self._th_err = (self._sy*self._sw/self._N)/(self._sy+self._sw/self._N)

    def set_N(self, N):
        self._N = N
        self._th_err = (self._sy*self._sw/self._N)/(self._sy+self._sw/self._N)

    def results(self):
        return {"MSE_ml" : self._MSE_tml, "MSE_bayes" : self._MSE_tbayes}


    def print_results(self):
        print("*** RESULTS ***")
        print()
        print("Parameters")
        print("runs MonteCarlo:", self._nMC)
        print("N:", self._N, " mean:", 0, " sigmay:", self._sy, " sigmaw:", self._sw)
        print()
        print("MSE ml:", self._MSE_tml)
        print("MSE bayes:", self._MSE_tbayes)
        print()
        print("***************")

    
    def _MSE(self, data, means):
        values = 0
        for i in range(0,len(data)):
            values = values + ((data[i] - means[i])**2)
        return values/len(data)
    

    def plot_results(self, xlable_name, xlable, MSE_tml ,MSE_tbayes, th_err, others=None, others_label=None, others2=None, others_label2=None):
        
        plt.plot(xlable, MSE_tml, label="ML")
        plt.plot(xlable, MSE_tbayes, label="Bayes")
        plt.plot(xlable, th_err, label="th err")
        if others != None:
            plt.plot(xlable, others, label=others_label)
        if others2 != None:
            plt.plot(xlable, others2, label=others_label2)

        plt.xscale('log')

        plt.xlabel(xlable_name)
        plt.ylabel("MSE")

        plt.legend()
        plt.show()




b = BayesianCase(nMC=100, N=10, sw=1000)

# ## Vary sy
# step = 0.1
# init_value = 0
# end_value = 1000
# sy = np.arange(init_value, end_value, step)
# n = len(sy)

# MSE_tml = list(range(n))
# MSE_tbayes = list(range(n))
# th_err = list(range(n))

# for i in range(0, n):
#     b.set_sy(sy[i])
#     b.run()
#     results = b.results()
#     MSE_tml[i] = results["MSE_ml"]
#     MSE_tbayes[i] = results["MSE_bayes"]
#     th_err[i] = b.get_th_err()
#     if(i%50==0):
#         print(i)

# max_mse = [b.get_sw()/b.get_N()] * n
# b.plot_results("sy", sy, MSE_tml ,MSE_tbayes, th_err, max_mse, "sw/N", sy.tolist(), "no data")
# b.set_sy(1)

# ## Vary sw
# step = 10
# init_value = 0
# end_value = 1000
# sw = np.arange(init_value, end_value, step)
# n = len(sw)

# MSE_tml = list(range(n))
# MSE_tbayes = list(range(n))
# th_err = list(range(n))

# for i in range(0, n):
#     b.set_sw(sw[i])
#     b.run()
#     results = b.results()
#     MSE_tml[i] = results["MSE_ml"]
#     MSE_tbayes[i] = results["MSE_bayes"]
#     th_err[i] = b.get_th_err()

# max_mse = [b.get_sy()] * n
# b.plot_results("sw", sw, MSE_tml ,MSE_tbayes, th_err, max_mse, "sy")
# b.set_sw(1)

## Vary N
step = 1
init_value = 1
end_value = 100
N = np.arange(init_value, end_value, step)
n = len(N)

MSE_tml = list(range(n))
MSE_tbayes = list(range(n))
th_err = list(range(n))

for i in range(0, n):
    b.set_N(N[i])
    b.run()
    results = b.results()
    MSE_tml[i] = results["MSE_ml"]
    MSE_tbayes[i] = results["MSE_bayes"]
    th_err[i] = b.get_th_err()

max_mse = [0] * n
b.plot_results("N", N, MSE_tml ,MSE_tbayes, th_err, max_mse, "0")