import np as np
import numpy as np
x = np.array([[197,2,2,30],
              [35,1,1,19],
              [52,2,1,25],
              [88,2,2,11],
              [65,2,2,8],
              [55,1,1,25],
              [46,1,1,42],
              [76,2,2,42],
              [196,3,3,66],
              [90,2,2,66],
              [96,2,2,23],
              [54,1,1,8]])
c = np.array([62.02231928,35.51894631,39.08033684,65.95423881])

#15470 4100
#y = np.array([15470])
#c = np.linalg.lstsq(x, y)[0]
print(c);
print(x @ c)
'''
x = np.array([197,2,2,30])
y = np.array(15470)

print(c); print(x @ c)'''
