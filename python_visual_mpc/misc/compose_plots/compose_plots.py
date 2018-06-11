import pickle as pkl
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs_single
from python_visual_mpc.utils.txt_in_image import draw_text_onimage
import numpy as np
from PIL import Image
import copy


def compose(dir):
    nstep = 30
    dict = pkl.load(open(dir + '/verbose/warperrs_tradeoff.pkl', 'rb'), encoding='latin1')
    warpperrs = dict['warperrs']
    reg_tradeoff = dict['tradeoff']

    columns = []

    for t in range(1,nstep, 3):

        column = []
        tdict = pkl.load(open(dir + '/verbose/plan/pred_t{}iter2.pkl'.format(t), 'rb'), encoding='latin1')

        curr_image = tdict['curr_img_cam0']
        desig = tdict['desig_pix']
        curr_image = add_crosshairs_single(curr_image[0][0], desig[0, 0], color=np.array([1, 0, 0]))
        curr_image = add_crosshairs_single(curr_image, desig[0, 1], color=np.array([0, 0, 1]))
        column.append(curr_image)

        warped_img_start = copy.deepcopy(tdict['warp_start_cam0'][0][0])
        column.append(warped_img_start)

        warped_img_goal = copy.deepcopy(tdict['warp_goal_cam0'][0][0])
        column.append(warped_img_goal)

        column = np.concatenate(column, 0)
        columns.append(column)


    image = np.concatenate(columns, 1)

    im = Image.fromarray((image*255).astype(np.uint8))
    im.save(dir + '/composed.png')


if __name__ == '__main__':
    dir = '/mnt/sda1/visual_mpc/experiments/cem_exp/benchmarks_sawyer/weissgripper_regstartgoal_tradeoff/exp/succ3_pkl'
    compose(dir)
