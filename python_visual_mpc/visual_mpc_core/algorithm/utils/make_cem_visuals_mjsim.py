import numpy as np
import os
import collections
import pickle
from python_visual_mpc.video_prediction.basecls.utils.visualize import add_crosshairs
import pdb

from python_visual_mpc.utils.txt_in_image import draw_text_onimage

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import resize_image
from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import get_score_images

from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import make_direct_vid

def image_addgoalpix(bsize, seqlen, image_l, goal_pix):
    goal_pix_ob = np.tile(goal_pix[None, None, :], [bsize, seqlen, 1])
    return add_crosshairs(image_l, goal_pix_ob)

def images_addwarppix(gen_images, warp_pts_l, pix, num_objects):
    warp_pts_arr = np.stack(warp_pts_l, axis=1)
    for ob in range(num_objects):
        warp_pts_ob = warp_pts_arr[:, :, pix[ob, 0], pix[ob, 1]]
        gen_images = add_crosshairs(gen_images, np.flip(warp_pts_ob, 2))
    return gen_images


class CEM_Visual_Preparation(object):
    def __init__(self):
        pass

    def visualize(self, vd):
        """
        :param vd:  visualization data
        :return:
        """

        bestindices = vd.scores.argsort()[:vd.K]

        print('in make_cem_visuals')
        plt.switch_backend('agg')

        gen_images = vd.gen_images[bestindices]

        self.num_ex = bestindices.shape[0]
        self._t_dict = collections.OrderedDict()
        self._t_dict['mj_rollouts'] = gen_images

        if hasattr(vd,'demo'):
            self._t_dict['goal_images'] = vd.goal_images
            self._t_dict['diff'] = gen_images - vd.goal_images
        pdb.set_trace()

        make_direct_vid(self._t_dict, self.num_ex, vd.save_dir,
                                suf='t{}iter{}'.format(vd.t, vd.cem_itr))


