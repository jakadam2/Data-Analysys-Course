from TwoTranches import TwoTranches
from BayesianCase import BayesianCase
import numpy as np
import matplotlib.pyplot as plt


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


### BAYESIAN CASE ###

# b = BayesianCase(nMC=100, N=10, sw=1000)

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

# ## Vary N
# step = 1
# init_value = 1
# end_value = 100
# N = np.arange(init_value, end_value, step)
# n = len(N)

# MSE_tml = list(range(n))
# MSE_tbayes = list(range(n))
# th_err = list(range(n))

# for i in range(0, n):
#     b.set_N(N[i])
#     b.run()
#     results = b.results()
#     MSE_tml[i] = results["MSE_ml"]
#     MSE_tbayes[i] = results["MSE_bayes"]
#     th_err[i] = b.get_th_err()

# max_mse = [0] * n
# b.plot_results("N", N, MSE_tml ,MSE_tbayes, th_err, max_mse, "0")


