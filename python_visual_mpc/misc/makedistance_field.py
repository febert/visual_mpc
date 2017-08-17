import numpy as np
from matplotlib import pyplot as plt

f = np.zeros((64,64))

g = np.array([25, 60])
for i in range(64):
    for j in range(64):
        f[i,j] = np.linalg.norm(np.array([i,j])- g)


plt.imshow(f, zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
plt.show()