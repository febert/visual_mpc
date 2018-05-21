""" This file defines the linear Gaussian policy class. """
import cv2
import numpy as np

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


class CEM_controller(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams):
        Policy.__init__(self)

        self.agentparams = ag_params
        self.policyparams = policyparams

        self.t = None

        if 'low_level_ctrl' in self.policyparams:
            self.low_level_ctrl = policyparams['low_level_ctrl']['type'](None, policyparams['low_level_ctrl'])


        if 'verbose' in self.policyparams:
            self.verbose = True
        else: self.verbose = False


        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

        self.action_list = []

        self.nactions = self.policyparams['nactions']
        self.repeat = self.policyparams['repeat']
        self.M = self.policyparams['num_samples']
        self.K = 10  # only consider K best samples for refitting

        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]

        # the full horizon is actions*repeat
        if 'action_cost_factor' in self.policyparams:
            self.action_cost_mult = self.policyparams['action_cost_factor']
        else: self.action_cost_mult = 0.00005

        self.adim = self.agentparams['adim'] # action dimension
        self.sdim = self.agentparams['sdim'] # action dimension
        self.initial_std = policyparams['initial_std']
        if 'exp_factor' in policyparams:
            self.exp_factor = policyparams['exp_factor']

        self.naction_steps = self.policyparams['nactions']

        # define which indices of the action vector shall be discretized:
        if 'discrete_adim' in self.agentparams:
            self.discrete_ind = self.agentparams['discrete_adim']
        else:
            self.discrete_ind = None


        self.init_model = None
        #history of designated pixels
        self.desig_pix = []

        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices =[]

        self.target = np.zeros(2)

        self.rec_input_distrib =[]  # record the input distributions
        self.corr_gen_images = []
        self.corrector = None

        self.mean =None
        self.sigma =None

    def create_sim(self):
        self.sim = MjSim(load_model_from_path(self.agentparams['gen_xml_fname']))

    def reinitialize(self):
        self.use_net = self.policyparams['usenet']
        self.action_list = []
        self.gtruth_images = [np.zeros((self.M, 64, 64, 3)) for _ in range(self.nactions * self.repeat)]
        self.initial_std = self.policyparams['initial_std']
        # history of designated pixels
        self.desig_pix = []
        # predicted positions
        self.pred_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.rec_target_pos = np.zeros((self.M, self.niter, self.repeat * self.nactions, 2))
        self.bestindices_of_iter = np.zeros((self.niter, self.K))

        self.indices = []

        self.target = np.zeros(2)

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
                                         t in range(self.nactions * self.repeat)])
            actions_costs[smp]=np.sum(np.square(force_magnitudes)) * self.action_cost_mult
        return actions_costs

    def discretize(self, actions):
        for b in range(self.M):
            for a in range(self.naction_steps):
                for ind in self.discrete_ind:
                    actions[b, a, ind] = np.clip(np.floor(actions[b, a, ind]), 0, 4)
        return actions

    def perform_CEM(self, t):
        # initialize mean and variance

        # initialize mean and variance
        self.mean = np.zeros(self.adim * self.naction_steps)
        # initialize mean and variance of the discrete actions to their mean and variance used during data collection
        self.sigma = construct_initial_sigma(self.policyparams)

        print('------------------------------------------------')
        print('starting CEM cylce')

        for itr in range(self.niter):
            print('------------')
            print('iteration: ', itr)
            t_startiter = time.time()
            actions = np.random.multivariate_normal(self.mean, self.sigma, self.M)
            actions = actions.reshape(self.M, self.naction_steps, self.adim)
            if self.discrete_ind != None:
                actions = self.discretize(actions)
            actions = np.repeat(actions, self.repeat, axis=1)

            if 'random_policy' in self.policyparams:
                print('sampling random actions')
                self.bestaction_withrepeat = actions[0]
                return

            t_start = time.time()

            scores = self.take_mujoco_smp(actions, itr, self.t)

            print('overall time for evaluating actions {}'.format(time.time() - t_start))

            actioncosts = self.calc_action_cost(actions)
            scores += actioncosts

            self.indices = scores.argsort()[:self.K]
            self.bestindices_of_iter[itr] = self.indices

            self.bestaction_withrepeat = actions[self.indices[0]]

            actions = actions.reshape(self.M, self.naction_steps, self.repeat, self.adim)
            actions = actions[:, :, -1, :]  # taking only one of the repeated actions
            actions_flat = actions.reshape(self.M, self.naction_steps * self.adim)

            self.bestaction = actions[self.indices[0]]
            # print 'bestaction:', self.bestaction

            arr_best_actions = actions_flat[self.indices]  # only take the K best actions
            self.sigma = np.cov(arr_best_actions, rowvar=False, bias=False)
            self.mean = np.mean(arr_best_actions, axis=0)

            print('iter {0}, bestscore {1}'.format(itr, scores[self.indices[0]]))
            print('action cost of best action: ', actioncosts[self.indices[0]])

            print('overall time for iteration {}'.format(time.time() - t_startiter))

    def take_mujoco_smp(self, actions, itr, tstep):
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
            vid = [np.concatenate([b[t_] for b in best_vids], 1) for t_ in range(self.nactions*self.repeat)]
            self.save_gif(vid, 't{}_iter{}'.format(tstep, itr))

        return all_scores

    def save_gif(self, images, name):
        file_path = self.agentparams['record']
        npy_to_gif(images, file_path +'/video'+name)

    def clip_targetpos(self, pos):
        pos_clip = self.agentparams['targetpos_clip']
        return np.clip(pos, pos_clip[0], pos_clip[1])

    def sim_rollout(self, actions):
        costs = []
        self.hf_qpos_l = []
        self.hf_target_qpos_l = []

        images = []
        self.gripper_closed = False
        self.gripper_up = False
        # print('start episdoe')

        for t in range(self.nactions*self.repeat):
            mj_U = actions[t]
            # print 'time ',t, ' target pos rollout: ', roll_target_pos

            if 'posmode' in self.agentparams:  #if the output of act is a positions
                if t == 0:
                    self.prev_target_qpos = copy.deepcopy(self.sim.data.qpos[:self.adim].squeeze())
                    self.target_qpos = copy.deepcopy(self.sim.data.qpos[:self.adim].squeeze())
                else:
                    self.prev_target_qpos = copy.deepcopy(self.target_qpos)

                if 'discrete_adim' in self.agentparams:
                    up_cmd = mj_U[2]
                    assert np.floor(up_cmd) == up_cmd
                    if up_cmd != 0:
                        self.t_down = t + up_cmd
                        self.target_qpos[2] = self.agentparams['targetpos_clip'][1][2]
                        self.gripper_up = True
                    if self.gripper_up:
                        if t == self.t_down:
                            self.target_qpos[2] = self.agentparams['targetpos_clip'][0][2]
                            self.gripper_up = False
                    self.target_qpos[:2] += mj_U[:2]
                    if self.adim == 4:
                        self.target_qpos[3] += mj_U[3]
                elif 'close_once_actions' in self.agentparams:
                    assert self.adim == 5
                    self.target_qpos[:4] = mj_U[:4] + self.target_qpos[:4]
                    grasp_thresh = 0.5
                    if mj_U[4] > grasp_thresh:
                        self.gripper_closed = True
                    if self.gripper_closed:
                        self.target_qpos[4] = 0.1
                    else:
                        self.target_qpos[4] = 0.0
                else:
                    self.target_qpos = mj_U + self.target_qpos
                self.target_qpos = self.clip_targetpos(self.target_qpos)
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

    def act(self, traj, t, init_model, goal_obj_pose, agent_params, goal_image):

        self.agentparams = agent_params
        self.goal_obj_pose = copy.deepcopy(goal_obj_pose)

        if 'task_switch' in self.policyparams:
            if t < self.agentparams['T']//2:
                i_ob = 0
            else:
                i_ob = 1
            qpos_dim = self.sdim//2
            curr_pose = init_model.data.qpos[i_ob * 7 + qpos_dim:(i_ob+1) * 7 + qpos_dim].squeeze()
            self.goal_obj_pose[i_ob] = curr_pose

        self.goal_image = goal_image
        self.t = t
        self.init_model = init_model

        if t == 0:
            action = np.zeros(self.adim)
            self.create_sim()
        else:
            if 'use_first_plan' in self.policyparams:
                print('using actions of first plan, no replanning!!')
                if t == 1:
                    self.perform_CEM(t)
                action = self.bestaction_withrepeat[t - 1]
            elif 'replan_interval' in self.policyparams:
                print('using actions of first plan, no replanning!!')
                if (t-1) % self.policyparams['replan_interval'] == 0:
                    self.last_replan = t
                    self.perform_CEM(t)
                print('last replan', self.last_replan)
                action = self.bestaction_withrepeat[t - self.last_replan]
            else:
                self.perform_CEM(t)
                action = self.bestaction[0]

        self.action_list.append(action)
        print('timestep: ', t, ' taking action: ', action)

        return action

