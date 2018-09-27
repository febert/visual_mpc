from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred
import copy
import numpy as np
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import resize_image
from python_visual_mpc.visual_mpc_core.algorithm.utils.make_cem_visuals import CEM_Visual_Preparation_FullImageReg
import imp
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_base import CEM_Controller_Base

from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn

class Full_Image_Reg_Controller(CEM_Controller_Vidpred):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(Full_Image_Reg_Controller, self).__init__(ag_params, policyparams, gpu_id, ngpu)

        params = imp.load_source('params', ag_params['current_dir'] + '/gdnconf.py')
        self.gdnconf = params.configuration
        self.gdnconf['batch_size'] = self.bsize
        self.pred_len = self.seqlen - self.ncontxt
        self.goal_image_warper = setup_gdn(self.gdnconf, gpu_id)
        self.visualizer = CEM_Visual_Preparation_FullImageReg()

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):
        flow_fields = np.zeros([self.M, self.pred_len, self.ncam, self.img_height, self.img_width, 2])
        warped_images = np.zeros([self.M, self.pred_len, self.ncam, self.img_height, self.img_width, 3])
        warp_pts_l = []
        goal_images = np.tile(self.goal_image[None], [self.bsize, 1, 1, 1,1,1])   # shape b,t,n, r, c, 3

        for tstep in range(self.pred_len):
            if self.policyparams['new_goal_freq'] == 'follow_traj':
                goal_im = goal_images[:, tstep]
            else:
                goal_im = goal_images[:, -1]

            if 'warp_goal_to_pred' in self.policyparams:
                warped_image, flow_field, warp_pts = self.goal_image_warper(goal_im, gen_images[:,tstep])
            else:
                warped_image, flow_field, warp_pts = self.goal_image_warper(gen_images[:,tstep], goal_im)

            flow_fields[:, tstep] = flow_field
            warped_images[:, tstep] = warped_image
            warp_pts_l.append(warp_pts)

        scores_percam = []
        for icam in range(self.ncam):
            scores_percam.append(self.compute_warp_cost(flow_fields[:,:,icam], warped_images[:,:,icam]))
        scores = np.stack(scores_percam, axis=1)
        scores = np.mean(scores, axis=1)
        self.vd.flow_mags = np.linalg.norm(flow_fields, axis=-1)
        self.vd.warped_images = warped_images
        self.vd.goal_image = self.goal_image
        return scores


    def compute_warp_cost(self, flow_field, warped_images):
        """
        :param flow_field:  shape: batch, time, r, c, 2
        :param goal_pix: if not None evaluate flowvec only at position of goal pix
        :return:
        """
        flow_mags = np.linalg.norm(flow_field, axis=4)
        flow_scores = np.mean(np.mean(flow_mags, axis=2), axis=2)

        per_time_multiplier = np.ones([1, flow_scores.shape[1]])
        per_time_multiplier[:, -1] = self.policyparams['finalweight']

        if 'warp_success_cost' in self.policyparams:
            ws_costs = np.mean(np.mean(np.mean(np.square(warped_images - self.goal_image[None,:,0]), axis=2), axis=2), axis=2)
            ws_costs = np.sum(ws_costs * per_time_multiplier, axis=1)
            stand_ws_costs = (ws_costs - np.mean(ws_costs)) / (np.std(ws_costs) + 1e-7)

            flow_scores = np.sum(flow_scores * per_time_multiplier, axis=1)
            stand_flow_scores = (flow_scores - np.mean(flow_scores)) / (np.std(flow_scores) + 1e-7)

            w = self.policyparams['warp_success_cost']
            scores = stand_flow_scores * (1 - w) + stand_ws_costs * w
        else:
            scores = np.sum(flow_scores * per_time_multiplier, axis=1)
        return scores

    def act(self, t, i_tr, images, goal_image, state, desig_pix, goal_pix):
        self.images = images
        self.state = state

        if self._hp.follow_traj:
            self.goal_images = goal_image[t:t + self.len_pred, 0]  #take first cam
        elif self._hp.goal_image_seq:
            new_goal_freq = self._hp.new_goal_freq
            demo_image_interval = self.hp.demo_image_interval
            assert demo_image_interval <= new_goal_freq
            igoal = t//new_goal_freq + 1
            use_demo_step = np.clip(igoal*demo_image_interval, 0, self.agentparams['num_load_steps']-1)
            self.goal_image = goal_image[use_demo_step][None]
        else:
            self.goal_image = goal_image[-1][None]   # take the last loaded image as goalimage for all steps

        return super(CEM_Controller_Vidpred, self).act(t, i_tr)

    def make_input_distrib(self, itr):
        return np.zeros((1, self.netconf['context_frames'], self.ncam, self.img_height, self.img_width, self.ndesig), dtype=np.float32)
