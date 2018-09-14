import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []
for i in range(18):
    images.append(cv2.resize(cv2.imread('im_{}.jpg'.format(i)), (64, 48)).astype(float) / 255)


goal = cv2.resize(cv2.imread('goal.jpg'), (64, 48)).astype(float) / 255.
mse = []
for img in images:
    err = np.mean(np.square(np.square(goal) - np.square(img)))
    mse.append(err)

plt.plot(mse)
plt.show()


