import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
import pickle



class Getdesig(object):
    def __init__(self,img, num_pts):
        self.img = img
        self.num_pts = num_pts

        self.im_height =img.shape[0]
        self.im_width = img.shape[1]

        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, self.im_width)
        self.ax.set_ylim(self.im_height, 0)
        plt.imshow(img)

        self.coords = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.coord_list = []
        plt.show()

    def onclick(self, event):
        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        self.coords = np.round(np.array([event.ydata, event.xdata])).astype(np.int)
        self.ax.scatter(self.coords[1], self.coords[0], marker= "o", s=70, facecolors='b', edgecolors='b')
        self.coord_list.append(self.coords)
        self.ax.set_xlim(0, self.im_width)
        self.ax.set_ylim(self.im_height, 0)
        plt.draw()

        if self.num_pts == (len(self.coord_list)):
            plt.close()


def visualize(file):
    pkldata = pickle.load(open(file, 'rb'))

    mpc_t = 1
    warped_images = pkldata['warped_im_t{}'.format(mpc_t)]
    gen_images = pkldata['gen_images_t{}'.format(mpc_t)]
    warp_pts = pkldata['warp_pts_t{}'.format(mpc_t)]
    flow_field = pkldata['flow_fields{}'.format(mpc_t)]
    goal_image = pkldata['goal_image']

    b_ind = 0

    warped_im = warped_images[0][b_ind]

    num_samples = 10
    c = Getdesig(warped_im, num_samples)
    pts_output = np.stack(c.coord_list, axis=0)

    for t in range(14):
        warped_im = warped_images[t][b_ind]
        gen_im = gen_images[t][b_ind]

        plt.figure(figsize=(10, 5), dpi=200)
        ax1 = plt.subplot(141)
        ax2 = plt.subplot(142)
        ax3 = plt.subplot(143)
        ax4 = plt.subplot(144)


        ax1.imshow(gen_im)
        ax2.imshow(warped_im)

        flow_mag = np.linalg.norm(flow_field[t][b_ind], axis=2)
        ax3.imshow(flow_mag)

        ax4.imshow(goal_image[0][0])
        # plt.show()

        coordsA = "data"
        coordsB = "data"
        # random pts

        imheight = gen_images[0].shape[1]
        imwidth  = gen_images[0].shape[2]

        # row_inds = np.random.randint(0, imheight, size=(num_samples)).reshape((num_samples, 1))
        # col_inds = np.random.randint(0, imwidth, size=(num_samples)).reshape((num_samples, 1))
        # pts_output = np.concatenate([row_inds, col_inds], axis=1)



        for p in range(num_samples):
            pt_output = pts_output[p]
            sampled_location = warp_pts[t][b_ind,pt_output[0],pt_output[1]].astype('uint32')
            sampled_location = np.flip(sampled_location, 0)
            print("point in warped img", pt_output, "sampled location", sampled_location)

            con = ConnectionPatch(xyA=np.flip(pt_output,0), xyB=np.flip(sampled_location,0), coordsA=coordsA, coordsB=coordsB,
                         axesA=ax2, axesB=ax1,
                         arrowstyle="<->",shrinkB=5, linewidth=1., color=np.random.uniform(0,1.,3))
            ax2.add_artist(con)
        # ax1.set_xlim(0, 128)
        # ax1.set_ylim(0, 128)
        # ax2.set_xlim(0, 128)
        # ax2.set_ylim(0, 128)
        # plt.draw()
        plt.show()


if __name__ == '__main__':
    file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks/goal_img_goalevalspot/verbose/traj0_conf0/plan/data1.pkl'
    visualize(file)