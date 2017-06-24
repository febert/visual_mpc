import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec

import pdb
import cPickle


class Visualizer(object):
    # def __init__(self, conf, file_path, name= "", examples = 4, show_parts=False):
    def __init__(self):

        dict_ = cPickle.load(open(file_path + '/dict_.pkl', "rb"))
        gen_images = dict_['gen_images']

        self.num_ex = 4
        self.row_list = []

        if 'ground_truth' in dict_:
            ground_truth = dict_['ground_truth']
            if not isinstance(ground_truth, list):
                ground_truth = np.split(ground_truth, ground_truth.shape[1], axis=1)
                ground_truth = [np.squeeze(g) for g in ground_truth]
            ground_truth = ground_truth[1:]

            self.row_list.append((ground_truth, 'Ground Truth'))

        self.row_list.append((gen_images, 'Gen Images'))

        self.build_figure()


    def build_figure(self):

        # plot each markevery case for linear x and y scales
        figsize = (10, 8)
        fig = plt.figure(num=1, figsize=figsize)
        axes_list = []

        num_rows = len(self.row_list)
        outer_grid = gridspec.GridSpec(num_rows, 1)
        for row in range(num_rows):
            # outer_ax = fig.add_subplot(outer_grid[row])
            # if self.row_list[row][1] != '':
            #     outer_ax.set_title(self.row_list[1])

            inner_grid = gridspec.GridSpecFromSubplotSpec(1, self.num_ex,
                              subplot_spec=outer_grid[row], wspace=0.0, hspace=0.0)

            image_row = self.row_list[row][0]

            for col in range(self.num_ex):
                ax = plt.Subplot(fig, inner_grid[col])
                ax.set_xticks([])
                ax.set_yticks([])
                axes_list.append(fig.add_subplot(ax))
                if row==0:
                    axes_list[-1].set_title('ex{}'.format(col))
                axes_list[-1].imshow(image_row[0][col], interpolation='none')

        plt.axis('off')
        fig.tight_layout()
        plt.show()

        # initialization function: plot the background of each frame

        # Set up formatting for the movie files
        Writer = animation.writers['imagemagick_file']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init,
                                       frames=13, interval=50, blit=True)
        anim.save('basic_animation.gif', writer='imagemagick')
        plt.show()

    def init(self):
        self.im.set_data(self.gen_images[0][0])
        return [self.im]

    def animate(self, i):
        self.im.set_array(self.gen_images[i][0])
        return [self.im]

if __name__ == '__main__':
    file_path = '/home/frederik/Documents/catkin_ws/src/lsdc/tensorflow_data/occulsionmodel/CDNA_quad_sawyer_refeed/modeldata'
    v  = Visualizer()