import pickle
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import npy_to_gif
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import assemble_gif
from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs_single, add_crosshairs

def visualize_video_track(dict=None, dir = None, exmps=None, points_file=None):
    if exmps == None:
        exmps = range(10)

    num_ex = len(exmps)

    if dict == None:
        dict = pickle.load(open(dir + '/data.pkl', 'rb'))
    print('loaded')

    videos = dict['videos']
    I0_ts = videos['I0_ts']
    warp_pts_bwd = videos['warp_pts_bwd']

    goal_images = I0_ts[-1]
    seqlen = len(I0_ts)
    annotated_goal_images = []
    goal_pix = []

    if not os.path.exists(points_file):
        for ex in exmps:
            print(ex)
            c = Getdesig(goal_images[ex])
            goal_pix.append(np.round(c.coords).astype(np.int))
            # goal_pix.append(np.array([0,0]))
            annotated_goal_images.append(add_crosshairs_single(goal_images[ex], goal_pix[-1]))

        annotated_goal_images = np.stack(annotated_goal_images, 0)
        annotated_goal_images = [annotated_goal_images for _ in range(seqlen)]
        pkl.dump({'goal_pix':goal_pix, 'annotated_goal_images':annotated_goal_images}, open(points_file, 'wb'))
    else:
        print('loading goalpix')
        dict = pkl.load(open(points_file, 'rb'))
        goal_pix = dict['goal_pix']
        annotated_goal_images = dict['annotated_goal_images']
        num_ex = len(goal_pix)

    desig_pix = np.zeros((num_ex, seqlen, 2))
    for t in range(seqlen):
        for b, ex in enumerate(exmps):
            desig_pix[b, t] = np.flip(warp_pts_bwd[t][ex, goal_pix[b][0], goal_pix[b][1]], 0)

    I0_ts = [i0[exmps] for i0 in I0_ts]
    I0_ts = add_crosshairs(I0_ts, desig_pix)
    goal_pix = np.stack(goal_pix, 0)
    goal_pix = np.tile(goal_pix[:,None, :], [1, seqlen, 1])
    gen_images_I1 = [i0[exmps] for i0 in videos['gen_images_I1']]
    gen_images_I1 = add_crosshairs(gen_images_I1, goal_pix)

    imlist = assemble_gif([annotated_goal_images, I0_ts, gen_images_I1], num_ex)
    npy_to_gif(imlist, dir + '/gdn_tracking')


def make_plots(conf, dict=None, dir = None):
    if dict == None:
        dict = pickle.load(open(dir + '/data.pkl'))

    print('loaded')
    videos = dict['videos']

    I0_ts = videos['I0_ts']

    # num_exp = I0_t_reals[0].shape[0]
    num_ex = 4
    start_ex = 0
    num_rows = num_ex*len(list(videos.keys()))
    num_cols = len(I0_ts) + 1

    print('num_rows', num_rows)
    print('num_cols', num_cols)

    width_per_ex = 2.5

    standard_size = np.array([width_per_ex * num_cols, num_rows * 1.5])  ### 1.5
    figsize = (standard_size).astype(np.int)

    f, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    print('start')
    for col in range(num_cols -1):
        row = 0
        for ex in range(start_ex, start_ex + num_ex, 1):
            for tag in list(videos.keys()):
                print('doing tag {}'.format(tag))
                if isinstance(videos[tag], tuple):
                    im = videos[tag][0][col]
                    score = videos[tag][1]
                    axarr[row, col].set_title('{:10.3f}'.format(score[col][ex]), fontsize=5)
                else:
                    im = videos[tag][col]

                h = axarr[row, col].imshow(np.squeeze(im[ex]), interpolation='none')

                if len(im.shape) == 3:
                    plt.colorbar(h, ax=axarr[row, col])
                axarr[row, col].axis('off')
                row += 1

    row = 0
    col = num_cols-1

    if 'I1' in dict:
        for ex in range(start_ex, start_ex + num_ex, 1):
            im = dict['I1'][ex]
            h = axarr[row, col].imshow(np.squeeze(im), interpolation='none')
            plt.colorbar(h, ax=axarr[row, col])
            axarr[row, col].axis('off')
            row += len(list(videos.keys()))

    # plt.axis('off')
    f.subplots_adjust(wspace=0, hspace=0.3)

    # f.subplots_adjust(vspace=0.1)
    # plt.show()
    plt.savefig(conf['output_dir']+'/warp_costs_{}.png'.format(dict['name']))


if __name__ == '__main__':
    # filedir = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/weiss_thresh0.5_56x64/modeldata'
    filedir = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/tdac_weiss_cons0_56x64/modeldata'
    view = 1
    # filedir = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/multiview/view{}/modeldata'.format(view)
    # filedir = '/mnt/sda1/visual_mpc/tensorflow_data/gdn/weiss/multiview/view1'

    points_file = '/mnt/sda1/pushing_data/weiss_gripper_20k/points.pkl'
    # points_file = '/mnt/sda1/pushing_data/sawyer_grasping/sawyer_data/points_view{}.pkl'.format(view)

    # exmps = [2, 4, 13, 14, 16, 20, 21, 24, 32, 33]
    exmps = None

    visualize_video_track(exmps=exmps, dir=filedir, points_file=points_file)

