""" This file defines the linear Gaussian policy class. """
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
from .cem_controller_base import CEM_Controller_Base
import ray
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_sim import SimWorker
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_sim import CEM_Controller_Sim


def MSE_based_cost(gen_images, goal_image, hp):
    sq_diff = np.square(gen_images - goal_image[:,None])
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
        # self.parallel = True
        self.parallel = False
        if self.parallel:
            ray.init()

    def _default_hparams(self):
        default_dict = {
                          'cost_func':MSE_based_cost
        }
        parent_params = super()._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def get_scores(self, images, scores_mjc):
        return self._hp.cost_func(images, self.goal_image, self._hp)

    def act(self, t, i_tr, qpos_full, qvel_full, state, object_qpos, goal_pos, reset_state, goal_image):
        # def act(self, t, i_tr, qpos_full, goal_pos, reset_state):
        self.curr_sim_state = reset_state
        self.qpos_full = qpos_full[t]
        self.qvel_full = qvel_full[t]
        self.goal_pos = goal_pos
        self.goal_image = goal_image

        if t == 0:
            self.create_sim()
        return super(CEM_Controller_Sim, self).act(t, i_tr)
