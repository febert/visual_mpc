import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import cPickle




# fig, ax = plt.subplots(1)

[value] = cPickle.load(open('values.pkl', "rb"))
fig, ax = plt.subplots(1)


plt.imshow(value, zorder=0, interpolation='none')
ax.set_xlim([0, 63])
ax.set_ylim([0, 63])
plt.colorbar()
plt.show()

### calculate maximum derivattices in the valuefunction
xdiff = value[:-1, :] - value[1:, :]
xdiff = xdiff[:, :-1]
ydiff = value[:, :-1] - value[:, 1:]
ydiff = ydiff[:-1, :]

plt.imshow(xdiff)
plt.show()

i, j = np.unravel_index(xdiff.argmax(), xdiff.shape)

# tmpax = fig.add_axes()
# tmpax.set_axis_off()


ax.scatter(i, j, s=80, facecolors='none', edgecolors='r')
ax.scatter(1, 1, s=80, facecolors='none', edgecolors='g')
ax.scatter(63, 63, s=80, facecolors='none', edgecolors='b')
plt.show()
