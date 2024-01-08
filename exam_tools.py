from typing import Any
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def pca(data:np.array,pve:float = None,k:int  = None) -> np.array:
    '''PCA from data, stop conditions:'
    1) part of variance explained (0-1): stop when percent of explained variance is at least pve
    2) stops on k's principal component 
    '''
    assert(pve is None or pve <= 1)
    assert(k is None or k <= data.shape[1])

    scaler = StandardScaler()
    scale_data = scaler.fit_transform(data)
    u,s,_ = np.linalg.svd(scale_data)

    if pve is None and k is None:
        return u[:,0:data.shape[1]] @ np.diag(s)
    
    elif k is not None:
        return u[:,0:k] @ np.diag(s[0:k])
    
    else:
        evalues = s * s
        total_pve = np.sum(evalues)
        current_pve = 0
        current_pca = 0
        while (current_pve/total_pve) < pve:
            current_pve += evalues[current_pca]
            current_pca += 1
        return u[:,0:current_pca] @ np.diag(s[0:current_pca])


def mmse_tranch(tranch:np.array,prior_var:float,data_var:float) -> list:
    '''we have one tranch generate with var = data_varm and mean = 0 and prior with var = prior_var and mean = 0'''
    mean = ((tranch.shape[0]/data_var)/((1/prior_var) + (tranch.shape[0]/data_var))) * np.mean(tranch)
    var = 1/((1/prior_var) + (tranch.shape[0]/data_var))
    return [mean,var]


def knn(data:np.array,new_sample:np.array,k:int) -> float:
    X = data[:,0:data.shape[1] - 1]
    y = data[:,data.shape[1] - 1]
    order = np.argsort(np.linalg.norm(X - new_sample,axis = 1))
    X,y = X[order],y[order]
    to_mean = []
    for i in range(k):
        to_mean.append(y[i])
    to_mean = np.array(to_mean)
    return np.mean(to_mean)


def naive_kernel(data:np.array,new_sample:np.array,limit:float) -> float:
    X = data[:,0:data.shape[1] - 1]
    y = data[:,data.shape[1] - 1]
    order = np.argsort(np.linalg.norm(X - new_sample,axis = 1))
    X,y = X[order],y[order]
    dist = np.linalg.norm(X - new_sample,axis = 1)
    to_mean = []
    i = 0 
    while dist[i] <= limit:
        to_mean.append(y[i])
        i += 1
        if i == len(y):
            break
    to_mean = np.array(to_mean)
    return np.mean(to_mean)


def two_tranches(tranch1,tranch2,var1,var2):
    mean1 = np.mean(tranch1)
    mean2 = np.mean(tranch2)
    p = var1/(var1 + var2)
    plain_mean = np.mean(np.concatenate([tranch1,tranch2]))
    mle_mean = p * mean1 + (1 - p) * mean2
    return (mean1,mean2,plain_mean,mle_mean)


def plot_ROC(data,labels,L0,L1,step = 0.1):
    ratio_first = L1.pdf(data)
    ratio_zero = L0.pdf(data)
    ratios = ratio_first/ratio_zero
    threashold = 0
    x = []
    y = []
    while True:
        pred = ratios > threashold
        true_positive = np.sum(np.logical_and(pred == labels,pred == True))
        true_negative = np.sum(np.logical_and(pred == labels,pred == False))
        false_positive = np.sum(np.logical_and(pred != labels,pred == True))
        false_negative = np.sum(np.logical_and(pred != labels,pred == False))
        alfa = false_positive/(len(labels)/2)
        beta = false_negative/(len(labels)/2)
        x.append(alfa)
        y.append(1 - beta)
        threashold += step

        if true_positive < len(labels)* 10 ** -2:
            print(true_positive)
            break
    plt.plot(x,y)
    plt.plot([0,1],[0,1])
    plt.xlabel('alfa')
    plt.ylabel('1 - beta')
    plt.title('ROC curve')


class SGD:

    def __init__(self,LR = 10**-3) -> None:
        self.fitted = False
        self.LR = LR

    def __call__(self,x) -> Any:
        return self._predict(x)

    def _log_loss(self,x,y):
        return np.log(1 + (np.exp(-(y * (self.B@x.T)))))
    
    def _grad(self,x,y):
        pred = 1/(1 + np.exp(-(self.B@x.T)))
        diff = pred - ((1 + y)/2)
        return diff * x
    
    @staticmethod
    def _unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def _predict(self,x:np.array):
        X = np.hstack((x, np.ones((x.shape[0], 1), dtype=x.dtype)))
        return 1/(1 + np.exp(self.B@X.T))


    def fit(self,data,labels,LR = 10**-4,epochs = 500,epsilon = 10 ** -2,verbose = False):        
        X = np.hstack((data, np.ones((data.shape[0], 1), dtype=data.dtype)))
        X,Y = self._unison_shuffled_copies(X,labels)
        j = 0
        prev_loss = 0

        if self.fitted and data.shape[1] + 1 != self.B.shape[0]:
            raise ArithmeticError(f'data should have {self.B.shape[0] - 1} features insted of {data.shape[1]}')
        elif not self.fitted:
            self.B = np.zeros(X.shape[1])

        while True:
            cumm_loss = 0
            for i in range(X.shape[0]):
                x = X[i,:]
                y = Y[i,:]
                cumm_loss += self._log_loss(x,y)
                self.B = self.B - (LR *self._grad(x,y))    
            j += 1   

            if verbose:
                print(f'EPOCH: {j} LOSS: {cumm_loss} WEIGHTS:{self.B}')

            if abs(cumm_loss - prev_loss) < epsilon or j == epochs:
                if verbose:
                    print('TRAINING FINISHED')
                    print(f'FINAL WEIGHTS: {self.B}')
                self.fitted = True
                return
    
            prev_loss = cumm_loss