import numpy as np
import matplotlib.pyplot as plt

A_shape = 200, 300, 3
A = np.zeros(A_shape, dtype=np.uint8)
A[:, :, :] = 255
r = 3/5 * 100
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if (i-A.shape[0]/2)**2 + (j-A.shape[1]/2)**2 <= r**2:
            A[i, j, 0] = 255
            A[i, j, 1] = 204
            A[i, j, 2] = 0

plt.imshow(A)
plt.show()