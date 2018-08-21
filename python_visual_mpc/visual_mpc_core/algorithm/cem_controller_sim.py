""" This file defines the linear Gaussian policy class. """
import cv2
import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import reuse_cov, reuse_mean
from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
import time
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
import copy
from datetime import datetime
import os

from PIL import Image
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import construct_initial_sigma

from pyquaternion import Quaternion
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
from python_visual_mpc.visual_mpc_core.agent.utils.get_masks import get_obj_masks
from python_visual_mpc.visual_mpc_core.agent.utils.gen_gtruth_desig import gen_gtruthdesig
from python_visual_mpc.visual_mpc_core.agent.utils.convert_world_imspace_mj1_5 import project_point, get_3D

from python_visual_mpc.visual_mpc_core.agent.agent_mjc import get_target_qpos

from .cem_controller_base import CEM_Controller_Base


class CEM_Controller_Sim(CEM_Controller_Base):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, imiation_conf, ag_params, policyparams):
        super(CEM_Controller_Sim, self).__init__(ag_params, policyparams)

    def create_sim(self):
        self.sim = MjSim(load_model_from_path(self.agentparams['gen_xml_fname']))

    def finish(self):
        self.viewer.finish()

    def setup_mujoco(self):
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = self.init_model.data.qpos
        sim_state.qvel[:] = self.init_model.data.qvel
        self.sim.set_state(sim_state)
        self.sim.forward()

    def eval_action(self):
        abs_distances = []
        abs_angle_dist = []
        qpos_dim = self.sdim // 2  # the states contains pos and vel

        if 'gtruth_desig_goal_dist' in self.policyparams:
            width = self.agentparams['viewer_image_width']
            height = self.agentparams['viewer_image_height']
            dlarge_img = self.sim.render(width, height, camera_name="maincam", depth=True)[1][::-1, :]
            large_img = self.sim.render(width, height, camera_name="maincam")[::-1, :, :]
            curr_img = cv2.resize(large_img, dsize=(self.agentparams['image_width'], self.agentparams['image_height']), interpolation = cv2.INTER_AREA)
            _, curr_large_mask = get_obj_masks(self.sim, self.agentparams)
            qpos_dim = self.sdim//2
            i_ob = 0
            obj_pose = self.sim.data.qpos[i_ob * 7 + qpos_dim:(i_ob + 1) * 7 + qpos_dim].squeeze()
            desig_pix, goal_pix = gen_gtruthdesig(obj_pose, self.goal_obj_pose, curr_large_mask, dlarge_img, 10, self.agentparams, curr_img, self.goal_image)
            total_d = []
            for i in range(desig_pix.shape[0]):
                total_d.append(np.linalg.norm(desig_pix[i] - goal_pix[i]))
            return np.mean(total_d)
        else:

            for i_ob in range(self.agentparams['num_objects']):
                goal_pos = self.goal_obj_pose[i_ob, :3]
                curr_pose = self.sim.data.qpos[i_ob * 7 + qpos_dim:(i_ob+1) * 7 + qpos_dim].squeeze()
                curr_pos = curr_pose[:3]

                abs_distances.append(np.linalg.norm(goal_pos - curr_pos))

                goal_quat = Quaternion(self.goal_obj_pose[i_ob, 3:])
                curr_quat = Quaternion(curr_pose[3:])
                diff_quat = curr_quat.conjugate*goal_quat
                abs_angle_dist.append(np.abs(diff_quat.radians))

        # return np.sum(np.array(abs_distances)), np.sum(np.array(abs_angle_dist))
        return np.sum(np.array(abs_distances))

    def calc_action_cost(self, actions):
        actions_costs = np.zeros(self.M)
        for smp in range(self.M):
            force_magnitudes = np.array([np.linalg.norm(actions[smp, t]) for
                                         t in range(self.naction_steps * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self.action_cost_mult
        return actions_costs


    def get_rollouts(self, actions, cem_itr, itr_times):
        all_scores = np.empty(self.M, dtype=np.float64)
        image_list = []
        for smp in range(self.M):
            self.setup_mujoco()
            score, images = self.sim_rollout(actions[smp])
            image_list.append(images)
            # print('score', score)
            per_time_multiplier = np.ones([len(score)])
            per_time_multiplier[-1] = self.policyparams['finalweight']
            all_scores[smp] = np.sum(per_time_multiplier*score)

            # if smp % 10 == 0 and self.verbose:
            #     self.save_gif(images, '_{}'.format(smp))

        if self.verbose:
            bestindices = all_scores.argsort()[:self.K]
            best_vids = [image_list[ind] for ind in bestindices]
            vid = [np.concatenate([b[t_] for b in best_vids], 1) for t_ in range(self.naction_steps * self.repeat)]
            self.save_gif(vid, 't{}_iter{}'.format(self.t, cem_itr))

        return all_scores

    def save_gif(self, images, name):
        file_path = self.agentparams['record']
        npy_to_gif(images, file_path +'/video'+name)

    def sim_rollout(self, actions):
        costs = []
        self.hf_qpos_l = []
        self.hf_target_qpos_l = []

        images = []
        self.gripper_closed = False
        self.gripper_up = False
        self.t_down = 0

        for t in range(self.naction_steps * self.repeat):
            mj_U = actions[t]
            # print 'time ',t, ' target pos rollout: ', roll_target_pos

            if 'posmode' in self.agentparams:  #if the output of act is a positions
                if t == 0:
                    self.prev_target_qpos = copy.deepcopy(self.sim.data.qpos[:self.adim].squeeze())
                    self.target_qpos = copy.deepcopy(self.sim.data.qpos[:self.adim].squeeze())
                else:
                    self.prev_target_qpos = copy.deepcopy(self.target_qpos)

                zpos = self.sim.data.qpos[2]
                self.target_qpos, self.t_down, self.gripper_up, self.gripper_closed = get_target_qpos(
                    self.target_qpos, self.agentparams, mj_U, t, self.gripper_up, self.gripper_closed, self.t_down, zpos)
                # print('target_qpos', self.target_qpos)
            else:
                ctrl = mj_U.copy()

            for st in range(self.agentparams['substeps']):
                if 'posmode' in self.agentparams:
                    ctrl = self.get_int_targetpos(st, self.prev_target_qpos, self.target_qpos)
                self.sim.data.ctrl[:] = ctrl
                self.sim.step()
                self.hf_qpos_l.append(copy.deepcopy(self.sim.data.qpos))
                self.hf_target_qpos_l.append(copy.deepcopy(ctrl))

            costs.append(self.eval_action())

            if self.verbose:
                width = self.agentparams['viewer_image_width']
                height = self.agentparams['viewer_image_height']
                images.append(self.sim.render(width, height, camera_name="maincam")[::-1, :, :])

            # print(t)
        # self.plot_ctrls()
        return np.stack(costs, axis=0), images

    def get_int_targetpos(self, substep, prev, next):
        assert substep >= 0 and substep < self.agentparams['substeps']
        return substep/float(self.agentparams['substeps'])*(next - prev) + prev

    def plot_ctrls(self):
        plt.figure()
        # a = plt.gca()
        self.hf_qpos_l = np.stack(self.hf_qpos_l, axis=0)
        self.hf_target_qpos_l = np.stack(self.hf_target_qpos_l, axis=0)
        tmax = self.hf_target_qpos_l.shape[0]
        for i in range(self.adim):
            plt.plot(list(range(tmax)) , self.hf_qpos_l[:,i], label='q_{}'.format(i))
            plt.plot(list(range(tmax)) , self.hf_target_qpos_l[:, i], label='q_target{}'.format(i))
            plt.legend()
            plt.show()

    def act(self, traj, t, init_model=None, goal_obj_pose=None):
        self.goal_obj_pose = copy.deepcopy(goal_obj_pose)
        self.init_model = init_model
        if t == 0:
            self.create_sim()
        return super(CEM_Controller_Sim, self).act(traj, t)
