""" This file defines the linear Gaussian policy class. """
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_sim import CEM_Controller_Sim
from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import make_direct_vid
from python_visual_mpc.visual_mpc_core.infrastructure.assemble_cem_visuals import get_score_images

import collections
from python_visual_mpc.visual_mpc_core.algorithm.utils.make_cem_visuals_mjsim import CEM_Visual_Preparation

class VisualzationData():
    def __init__(self):
        """
        container for visualization data
        """
        pass

def MSE_based_cost(gen_images, goal_image, hp):

    sq_diff = np.square(gen_images - goal_image[None])
    mean_sq_diff = np.mean(sq_diff.reshape([sq_diff.shape[0],sq_diff.shape[1],-1]), -1)

    per_time_multiplier = np.ones([1, gen_images.shape[1]])
    per_time_multiplier[:, -1] = hp.finalweight
    return np.sum(mean_sq_diff * per_time_multiplier, axis=1)


class CEM_Controller_Sim_GoalImage(CEM_Controller_Sim):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super().__init__(ag_params, policyparams, gpu_id, ngpu)

    def _default_hparams(self):
        default_dict = {
                        'cost_func':MSE_based_cost,
                        'follow_traj':False,   # follow the demonstration frame by frame
                        'new_goal_freq':3,   # -1: only take a new goal once per trajectory
        }
        parent_params = super()._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def get_scores(self, images, scores_mjc):
        return self._hp.cost_func(images, self.goal_images, self._hp)

    def save_gif(self, images, all_scores, cem_itr):
        bestindices = all_scores.argsort()[:self.K]
        images = images[bestindices]
        self._t_dict = collections.OrderedDict()
        self._t_dict['mj_rollouts'] = images
        self._t_dict['goal_images'] = np.repeat(self.goal_images[None], self.K, 0)
        self._t_dict['diff'] = images - self.goal_images
        file_path = self.agentparams['record']

        self._t_dict['scores'] = get_score_images(all_scores[bestindices], 48, 64, self.len_pred, self.K)
        make_direct_vid(self._t_dict, self.K, file_path,
                        suf='t{}iter{}'.format(self.t, cem_itr))

    def act(self, t, i_tr, qpos_full, qvel_full, state, object_qpos, goal_pos, reset_state, goal_image):
        # def act(self, t, i_tr, qpos_full, goal_pos, reset_state):
        self.curr_sim_state = reset_state
        self.qpos_full = qpos_full[t]
        self.qvel_full = qvel_full[t]
        self.goal_pos = goal_pos
        self.goal_images = goal_image

        if t == 0:
            self.create_sim()

        if self._hp.follow_traj:
            self.goal_images = goal_image[t:t + self.len_pred, 0]  #take first cam
        else:
            new_goal_freq = self._hp.new_goal_freq
            demo_image_interval = self.hp.demo_image_interval
            assert demo_image_interval <= new_goal_freq
            igoal = t//new_goal_freq + 1
            use_demo_step = np.clip(igoal*demo_image_interval, 0, self.agentparams['num_load_steps']-1)
            self.goal_image = goal_image[use_demo_step][None]
            print('using goal image at of step {}'.format(igoal*demo_image_interval))

        return super(CEM_Controller_Sim, self).act(t, i_tr)
