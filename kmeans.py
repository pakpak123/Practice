import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def kmean(k):
    # initial centroids
    ic = np.random.permutation(len(Xtrain))[:k]
    C = Xtrain[ic]
    while True:
        # Distances
        D = []
        for c in C:
            D.append(np.sum((Xtrain - c) ** 2, axis=1))
        D = np.array(D)
        j = D.argmin(axis=0)
        # update centroids
        Cold = C.copy()
        for i in range(len(C)):
            C[i] = np.mean(Xtrain[j == i], axis=0)
        if np.sum(np.abs(Cold - C)) == 0:
            return C


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
itrain = np.r_[:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]

Xtrain = df.iloc[itrain, :-1].values
Xtest = df.iloc[itest, :-1].values
Ytrain = df.iloc[itrain, -1].values
Ytest = df.iloc[itest, -1].values
C = kmean(3)
print(C)
