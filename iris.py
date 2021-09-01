import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
def knn(x):
    return Ytrain[np.sum((Xtrain - x) ** 2, axis=1).argmin()]
    # ** 2 คือยกกำลัง 2



url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
itrain = np.r_[:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]

Xtrain = df.iloc[itrain, :-1].values
Xtest = df.iloc[itest, :-1].values
Ytrain = df.iloc[itrain, -1].values
# -1 เอาทุกคอลัมยกเว้นตัวสุดท้าย
Ytest = df.iloc[itest, -1].values

Ztest = []
for x in Xtest:
    Ztest.append(knn(x))
Ztest = np.array(Ztest)
acc = np.sum(Ztest==Ytest)/ len(Ytest) * 100
print(acc)
# fig = plt.figure().gca(projection='3d')
# fig.scatter(df[0][:50], df[1][:50], df[2][:50], c='r')
# fig.scatter(df[0][50:100], df[1][50:100], df[2][50:100], c='g')
# fig.scatter(df[0][100:], df[1][100:], df[2][100:], c='b')
# plt.show()

# plt.plot(df[0][:50], df[1][:50], '.r')
# plt.plot(df[0][50:100], df[1][50:100], '.g')
# plt.plot(df[0][100:], df[1][100:], '.b')
# plt.show()

