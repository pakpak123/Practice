import numpy as np
import matplotlib.pyplot as plt

n = 100000
stone = np.random.rand(n, 2)
plt.plot(stone[:, 0], stone[:, 1], '.r')
c = 0
for s in stone:
    if s[0] ** 2 + s[1] ** 2 <= 1:
        plt.plot(s[0], s[1], '.g')
        c += 1
monte = c / n
print(f'Simu Pi = {4*monte}')
print(f'Real Pi = {np.pi}')
print(f'Error = {abs(4*monte-np.pi) / np.pi * 100} %')
plt.show()
