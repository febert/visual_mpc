from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.utils.make_cem_visuals import CEM_Visual_Preparation_FullImage
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.full_image_reg_controller import Full_Image_Reg_Controller
from python_visual_mpc.visual_mpc_core.algorithm.utils.cem_cost_functions import mse_based_cost


class MSE_Full_Image_Controller(Full_Image_Reg_Controller, CEM_Controller_Vidpred):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(MSE_Full_Image_Controller, self).__init__(ag_params, policyparams, gpu_id, ngpu)
        self.pred_len = self.seqlen - self.ncontxt
        self.visualizer = CEM_Visual_Preparation_FullImage()

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):
        goal_images = np.tile(self.goal_image[None], [self.bsize, 1, 1, 1,1,1])   # shape b,t,n, r, c, 3
        scores = mse_based_cost(gen_images, goal_images, self._hp)
        self.vd.goal_image = self.goal_image
        return scores

    def act(self, t, i_tr, images, goal_image, state, desig_pix, goal_pix):
        self.images = images
        self.state = state

        if self.policyparams['new_goal_freq'] == 'follow_traj':
            self.goal_image = goal_image[t:t+self.pred_len]
        else:
            new_goal_freq = self.policyparams['new_goal_freq']
            demo_image_interval = self.policyparams['demo_image_interval']
            assert demo_image_interval <= new_goal_freq
            igoal = t//new_goal_freq + 1
            use_demo_step = np.clip(igoal*demo_image_interval, 0, self.agentparams['num_load_steps']-1)
            self.goal_image = goal_image[use_demo_step][None]
            print('using goal image at of step {}'.format(igoal*demo_image_interval))

        return super(MSE_Full_Image_Controller, self).act(t, i_tr)

    def make_input_distrib(self, itr):
        return np.zeros((1, self.netconf['context_frames'], self.ncam, self.img_height, self.img_width, self.ndesig), dtype=np.float32)
