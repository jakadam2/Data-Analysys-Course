from SGD import Logistic_regression

lr = Logistic_regression(nMC = 100000)

beta = lr.SGD()

print(beta)

## lr.LR(100)

## lr.plot_error()

lr.plot_cost()