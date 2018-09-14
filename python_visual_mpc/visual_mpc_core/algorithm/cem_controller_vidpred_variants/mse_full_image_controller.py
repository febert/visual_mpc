from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.utils.make_cem_visuals import CEM_Visual_Preparation_FullImage
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.full_image_reg_controller import Full_Image_Reg_Controller
from python_visual_mpc.visual_mpc_core.algorithm.utils.cem_cost_functions import mse_based_cost
import os
import matplotlib.pyplot as plt


class MSE_Full_Image_Controller(Full_Image_Reg_Controller, CEM_Controller_Vidpred):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        CEM_Controller_Vidpred.__init__(self, ag_params, policyparams, gpu_id, ngpu)
        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.pred_len = self.seqlen - self.ncontxt
        self.visualizer = CEM_Visual_Preparation_FullImage()
        self.step = 0

    def _default_hparams(self):
        default_dict = {
            'follow_traj': False,
            'goal_image_seq': False
        }
        parent_params = super()._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def save_cost_plots(self, scores, per_timestep_scores, cem_itr):
        dir = self.agentparams['record'] + 'plan/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        indices = np.argsort(scores)[:10]
        plt.figure()
        for n, i in enumerate(indices):
            costs = per_timestep_scores[i]
            plt.plot(costs)
        plt.savefig(dir + 'costs_step{}_{}'.format(self.t, cem_itr))

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):
        #goal_images = np.tile(self.goal_image[None], [self.bsize, 1, 1, 1,1,1])   # shape b,t,n, r, c, 3
        scores, per_timestep_scores = mse_based_cost(gen_images, self.goal_image, self._hp, normalize=False, squaring=True)
        self.vd.goal_image = self.goal_image
        best_cost_perstep = per_timestep_scores[np.argmin(scores)]
        self.save_cost_plots(scores, per_timestep_scores, cem_itr)
        self.step += 1
        return scores

    def act(self, t, i_tr, images, goal_image, state):
        return super(MSE_Full_Image_Controller, self).act(t, i_tr, images, goal_image, state, None, None)


