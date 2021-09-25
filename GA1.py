import numpy as np
import matplotlib.pyplot as plt

solution = [1, 3, -4, 2, -1, 1]
n = 50
x = np.random.rand(n)
y = np.polyval(solution, x)

def fitness(pop):
    E = []
    for p in pop:
        z = np.polyval(p, x)
        E.append(np.mean((z - y)**2))
    return np.array(E)

# Init pop
n_pop = 100
degree = 5
pop = np.random.rand(n_pop, degree+1)
X = np.arange(0, 1, 0.001)
while True:
    f = fitness(pop)
    idx = f.argsort()
    print(f'fitness = {f[idx[0]]}')
    plt.clf()
    plt.plot(x, y, '.r')
    Y = np.polyval(pop[idx[0]], X)
    plt.plot(X, Y, 'b')
    plt.draw()
    plt.pause(0.0001)
    pop = pop[idx]
    for i in range(50, 100):
        parents = np.random.permutation(50)[:2]
        xoverpoint = np.random.randint(0, degree+1)
        pop[i] = np.concatenate((pop[parents[0]][:xoverpoint], pop[parents[1]][xoverpoint:]))
        if np.random.rand() > 0.9:
            mutatepoint = np.random.randint(0, degree+1)
            pop[i][mutatepoint] += np.random.rand() * (-1)**np.random.randint(2)
