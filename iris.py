import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def knn(x):
    return Ytrain[np.sum((Xtrain - x) ** 2, axis=1).argmin()]

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
itrain = np.r_[:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]

Xtrain = df.iloc[itrain, :-1].values
Xtest = df.iloc[itest, :-1].values
Ytrain = df.iloc[itrain, -1].values
Ytest = df.iloc[itest, -1].values

Ztest = []
for x in Xtest:
    Ztest.append(knn(x))

Ztest = np.array(Ztest)
acc = np.sum(Ztest == Ytest) / len(Ytest) * 100
print(acc)