import numpy as np
from ApproximationRegression import ApproximationRegression

a = ApproximationRegression()

step = 10
init_value = 10
end_value = 2000
run = np.arange(init_value, end_value, step).tolist()
n = len(run)

MSE_KNN = list(range(n))
MSE_NK = list(range(n))


for i in range(n):
    if i%100==0:
        print(i)
    a.set_N(run[i])
    a.run()
    results = a.results()
    MSE_KNN[i] = results["MSE_KNN"]
    MSE_NK[i] = results["MSE_NK"]

a.plot_results(run, MSE_KNN, MSE_NK)