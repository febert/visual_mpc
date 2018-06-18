from matplotlib import pyplot as plt
import pdb
import numpy as np

class Getdesig(object):
    def __init__(self,img,conf=None, img_namesuffix=None):
        self.suf = img_namesuffix
        if conf is not None:
            self.outputdir = conf['output_dir']
        else: self.outputdir = None
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)

        plt.imshow(img)
        self.coords = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        self.coords = np.array([event.ydata, event.xdata])
        self.ax.scatter(self.coords[1], self.coords[0], marker= "o", s=70, facecolors='b', edgecolors='b')
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.draw()
        if self.outputdir is not None:
            plt.savefig(self.outputdir +'/img_desigpix'+self.suf)