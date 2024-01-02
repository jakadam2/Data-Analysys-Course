from sklearn.preprocessing import StandardScaler
import numpy as np


def pca(data,pve = None,k = None):
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
