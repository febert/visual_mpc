import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np
import cv2


class Getdesig(object):
    def __init__(self,img, maxclick):
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        self.coords = None

        self.maxclick = maxclick
        self.i_click = 0
        self.coord_list = []

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        self.i_click += 1

        if self.i_click == self.maxclick:
            plt.close()

        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        self.coords = np.around(np.array([event.ydata, event.xdata])).astype(np.int32)
        self.ax.scatter(self.coords[1], self.coords[0], marker= "o", s=30, facecolors='b', edgecolors='b')
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.draw()

        self.coord_list.append(self.coords)


def draw_poly(img):
    c = Getdesig(img, 8)
    coords = [np.flip(c, 0) for c in c.coord_list]
    coords = np.stack(coords)

    mask = np.zeros(img.shape[:2])
    mask = cv2.fillPoly(mask, [coords], 1.)
    plt.imshow(mask)
    plt.show()

    return mask


if __name__ == '__main__':
    draw_poly(cv2.imread('im0.png'))


