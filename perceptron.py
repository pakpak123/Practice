import numpy as np

# And problem
X = np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
Y = np.array([1, -1, -1, -1])

X_ = X.copy()
for i in range(len(X_)):
    X_[i] = Y[i] * X[i]

# initial W
W = X_[np.random.choice(len(X))]
c = 0
while c != len(X):
    c = 0
    for x in X_:
        if np.dot(x, W) < 0:
            W = W + x
            print(W)
        else:
            c += 1
print(W)

def f(x):
    return 1 if x >= 0 else -1

for x in X:
    print(f(np.dot(x, W)))
