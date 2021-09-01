import numpy as np
import matplotlib.pyplot as plt

A_shape = 200, 300, 3
A = np.zeros(A_shape,dtype=np.uint8)
# A+=254 สีขาว
# A[:,:,0]=255 แดง
# A[:,:,1]=255 เขียว
# A[:,:,2]=255 น้ำเงิน
# A[:,:,[0,2]]=255 ชมพู
'''
ธงชาติไทย ขนาด 6, 9
A[[0,5],:,0] =255
A[[1,4],:,:] =255
A[[2,3],:,2] =255
'''
'''ธงฝรั่งเศส ขนาด 2, 3
A[:,2,0] =255
A[:,1,:] =255
A[:,0,2] =255
'''
'''ธงเยอรมณี ขนาด 3, 5
A[0,:,:]
A[1,:,0] =255
A[2,:,[0,1]] =255
'''
r=3/5 *100
A[:,:,:] =255
for i in range(A_shape[0]):
    for j in range(A_shape[1]):
        if (i-A_shape[0]/2)** 2 + (j-A_shape[1]/2)** 2 <= r** 2:
            A[i,j,0]=255
            A[i,j,1:]=0

plt.imshow(A)
plt.show()