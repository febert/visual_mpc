import argparse
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import os


class Getdesig(object):
    def __init__(self, img):
        self.im_shape = [img.shape[0], img.shape[1]]

        self.n_desig = 1
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)
        plt.imshow(img)

        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_click = 0

        self.desig = np.zeros((1, 2))  #idesig, (r,c)

        self.i_click_max = 1
        self.clicks_per_desig = 1

        self.i_desig = 0
        self.i_goal = 0
        self.marker_list = ['o',"D","v","^"]

        plt.show()

    def onclick(self, event):
        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        import matplotlib.pyplot as plt
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)

        i_task = self.i_click//self.clicks_per_desig

        if self.i_click == self.i_click_max:
            plt.close()
            return

        rc_coord = np.array([event.ydata, event.xdata])

        if self.i_click % self.clicks_per_desig == 0:
            self.desig[i_task, :] = rc_coord
            color = "r"
        else:
            self.goal[i_task, :] = rc_coord
            color = "g"
        marker = self.marker_list[i_task]
        self.ax.scatter(rc_coord[1], rc_coord[0], s=100, marker=marker, facecolors=color)

        plt.draw()

        self.i_click += 1


def annotate_files(val_files, save_path, target_width, ncam, T, offset):
    f_num = offset
    for f in val_files[offset:]:
        points = np.zeros((ncam, T, 2))

        paths = ['{}/val{}'.format(save_path, f_num)] + \
                ['{}/val{}/images{}'.format(save_path, f_num, n) for n in range(ncam)]
        for s_path in paths:
            if not os.path.exists(s_path):
                os.makedirs(s_path)
                print('created {}'.format(s_path))
        for n in range(ncam):
            imgs = [cv2.imread('{}/images{}/im_{}.jpg'.format(f, n, t))[:, :, ::-1] for t in range(T)]
            if 'mirror' in f and n == 0:
                imgs = [i[:, ::-1] for i in imgs]
            assert target_width <= imgs[0].shape[1], "Should not resize to larger than source"
            target_height = int(float(target_width) / imgs[0].shape[1] * imgs[0].shape[0])
            imgs = [cv2.resize(i, (target_width, target_height), interpolation=cv2.INTER_AREA) for i in imgs]

            print('Annotating Camera: {}'.format(n))
            if n == 0:
                n_rows = math.ceil(T / 5)
                for i, img in enumerate(imgs):
                    plt.subplot(n_rows, 5, i + 1)
                    plt.imshow(img)
                plt.show()

            for i, img in enumerate(imgs):
                d = Getdesig(img)
                points[n, i] = d.desig.reshape(-1)
                print('got desig: {}'.format(points[n, i]))
                cv2.imwrite('{}/val{}/images{}/im_{}.jpg'.format(save_path, f_num, n, i), img[:, :, ::-1])

        np.save('{}/val{}/points'.format(save_path, f_num), points, allow_pickle=False)
        f_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_files', type=str,
                        help='path to text file containing validation folder paths (one per traj per line)')
    parser.add_argument('--save_path', type=str, default='./annotations', help="Folder to save annotations to")
    parser.add_argument('--offset', type=int, default=0, help="Number of validation folders already saved")
    parser.add_argument('--target_width', type=int, default=128, help="width to resize images to")
    parser.add_argument('--ncam', type=int, default=2, help="Number of Cameras")
    parser.add_argument('--T', type=int, default=30, help="Number of Timesteps")
    args = parser.parse_args()

    val_files = []
    with open(args.validation_files, 'r') as f:
        for line in f:
            if line == '\n':
                continue
            val_files.append(line.strip())
    annotate_files(val_files, args.save_path, args.target_width, args.ncam, args.T, args.offset)