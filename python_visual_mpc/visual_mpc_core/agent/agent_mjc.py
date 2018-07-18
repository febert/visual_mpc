""" This file defines an agent for the MuJoCo simulator environment. """
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import copy
import numpy as np
import pickle
from PIL import Image
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
from pyquaternion import Quaternion
from mujoco_py import load_model_from_path, MjSim
from python_visual_mpc.visual_mpc_core.agent.utils.get_masks import get_obj_masks
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from inspect import signature, Parameter
from python_visual_mpc.visual_mpc_core.agent.utils.target_qpos_utils import get_target_qpos

def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class Image_dark_except(Exception):
    def __init__(self):
        pass


class AgentMuJoCo(object):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._setup_world()

        self.T = self._hyperparams['T']
        self.sdim = self._hyperparams['sdim']
        self.adim = self._hyperparams['adim']
        self.goal_obj_pose = None
        self.goal_image = None
        self.goal_mask = None
        self.goal_pix = None
        self.curr_mask = None
        self.curr_mask_large = None
        self.desig_pix = None
        if 'cameras' in self._hyperparams:
            self.ncam = len(self._hyperparams['cameras'])
        else: self.ncam = 1
        self.start_conf = None
        self.load_obj_statprop = None  #loaded static object properties

    def _setup_world(self):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        env_type, env_params = self._hyperparams['env']
        self.env = env_type(env_params)

        self._hyperparams['adim'] = self.env.adim
        self._hyperparams['sdim'] = self.env.sdim
        self.num_objects = self.env.num_objects

    def apply_start_conf(self, dict):
        if 'reverse_action' in self._hyperparams:
            init_index = -1
            goal_index = 0
        else:
            init_index = 0
            goal_index = -1

        self.load_obj_statprop = dict['obj_statprop']
        self._hyperparams['xpos0'] = dict['qpos'][init_index]
        self._hyperparams['object_pos0'] = dict['object_full_pose'][init_index]
        self.object_full_pose_t = dict['object_full_pose']
        self.goal_obj_pose = dict['object_full_pose'][goal_index]   #needed for calculating the score
        if 'lift_object' in self._hyperparams:
            self.goal_obj_pose[:,2] = self._hyperparams['targetpos_clip'][1][2]

        if self.ncam != 1:
            self.goal_image = np.stack([dict['images0'][goal_index], dict['images1'][goal_index]], 0) # assign last image of trajectory as goalimage
        else:
            self.goal_image = dict['images'][goal_index]  # assign last image of trajectory as goalimage

        if len(self.goal_image.shape) == 3:
            self.goal_image = self.goal_image[None]
        if 'goal_mask' in self._hyperparams:
            self.goal_mask = dict['goal_mask'][goal_index]  # assign last image of trajectory as goalimage
        if 'compare_mj_planner_actions' in self._hyperparams:
            self.mj_planner_actions = dict['actions']

    def sample(self, policy, i_tr):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        if self.start_conf is not None:
            self.apply_start_conf(self.start_conf)

        if "gen_xml" in self._hyperparams:
            if i_tr % self._hyperparams['gen_xml'] == 0 and i_tr > 0:
                self._setup_world()

        traj_ok, obs_dict, policy_outs = False, None, None
        i_trial = 0
        imax = 100
        while not traj_ok and i_trial < imax:
            i_trial += 1
            try:
                traj_ok, obs_dict, policy_outs = self.rollout(policy, i_trial)
            except Image_dark_except:
                traj_ok = False

        print('needed {} trials'.format(i_trial))

        if 'make_final_gif' in self._hyperparams or 'make_final_gif_pointoverlay' in self._hyperparams:
            self.save_gif(i_tr, 'make_final_gif_pointoverlay' in self._hyperparams)

        if 'verbose' in self._hyperparams:
            self.plot_ctrls(i_tr)

        return obs_dict, policy_outs

    def hide_arm_store_image(self):
        highres_image = self.env.snapshot_noarm()
        target_dim = (self._hyperparams['image_width'], self._hyperparams['image_height'])


    def get_int_targetpos(self, substep, prev, next):
        assert substep >= 0 and substep < self._hyperparams['substeps']
        return substep/float(self._hyperparams['substeps'])*(next - prev) + prev

    def _post_process_obs(self, env_obs, initial_obs = False):
        """
        Handles conversion from the environment observations, to agent observation
        space. Observations are accumulated over time, and images are resized to match
        the given image_heightximage_width dimensions.

        Original images from cam index 0 are added to buffer for saving gifs (if needed)

        :param env_obs: observations dictionary returned from the environment
        :param initial_obs: Whether or not this is the first observation in rollout
        :return: obs: dictionary of observations up until (and including) current timestep
        """

        agent_img_height = self._hyperparams['image_height']
        agent_img_width = self._hyperparams['image_width']

        if initial_obs:
            T = self._hyperparams['T'] + 1
            self._agent_cache = {}
            for k in env_obs:
                if k == 'images':
                    if 'obj_image_locations' in env_obs:
                        self.traj_points = []
                    n_cams = env_obs['images'].shape[0]
                    obs_shape = (T, n_cams, agent_img_height, agent_img_width, 3)
                    self._agent_cache['images'] = np.zeros(obs_shape, dtype = np.uint8)
                elif isinstance(env_obs[k], np.ndarray):
                    obs_shape = [T] + list(env_obs[k].shape)
                    self._agent_cache[k] = np.zeros(tuple(obs_shape), dtype=env_obs[k].dtype)
                else:
                    self._agent_cache[k] = []
            self._cache_cntr = 0

        t = self._cache_cntr
        self._cache_cntr += 1

        obs = {}
        for k in env_obs:
            if k == 'images':
                self.large_images_traj.append(env_obs['images'][0])
                new_dims = (agent_img_width, agent_img_height)
                for i in range(env_obs['images'].shape[0]):
                    self._agent_cache['images'][t, i] = cv2.resize(env_obs['images'][i], new_dims,
                                                                    interpolation=cv2.INTER_AREA)
            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))
                env_obs['obj_image_locations'][:, :, 0] *= agent_img_height / env_obs['images'].shape[1]
                env_obs['obj_image_locations'][:, :, 1] *= agent_img_width / env_obs['images'].shape[2]
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        return obs

    def rollout(self, policy, i_tr):
        self._init()
        agent_img_height, agent_img_width = self._hyperparams['image_height'], self._hyperparams['image_width']
        if self.goal_obj_pose is not None:
            self.goal_pix = self.env.get_goal_pix(self.ncam, agent_img_width, self.goal_obj_pose)

        if 'first_last_noarm' in self._hyperparams:
            start_img = self.hide_arm_store_image()

        # Take the sample.
        t = 0
        done = False
        self.large_images_traj, self.traj_points= [], None
        obs = self._post_process_obs(self.env.reset(), True)
        policy_outputs = []

        while not done:
            """
            Currently refactoring the agent loop.
            Many features are being moved from agent into environment
            As a result many designated pixel related functions do not work
            This has implications for running MPC in sim
            """
            if 'get_curr_mask' in self._hyperparams:
                self.curr_mask, self.curr_mask_large = get_obj_masks(self.env.sim, self._hyperparams, include_arm=False) #get target object mask
            else:
                self.desig_pix = self.env.get_desig_pix(self.ncam, agent_img_width)

            policy_args = {}
            policy_signature = signature(policy.act)              #Gets arguments required by policy
            for arg in policy_signature.parameters:               #Fills out arguments according to their keyword
                value = policy_signature.parameters[arg].default
                if arg in obs:
                    value = obs[arg]
                elif arg == 't':
                    value = t

                if value is Parameter.empty:
                    #required parameters MUST be set by agent
                    raise ValueError("Required Policy Param {} not set in agent".format(arg))
                policy_args[arg] = value

            pi_t = policy.act(**policy_args)
            policy_outputs.append(pi_t)

            try:
                print('step {}'.format(t))
                obs = self._post_process_obs(self.env.step(copy.deepcopy(pi_t['actions'])))
            except ValueError:
                return False, None, None

            if (self._hyperparams['T']-1) == t:
                done = True
            if done:
                obs['term_t'] = t
            t += 1

        if 'first_last_noarm' in self._hyperparams:
            end_img = self.hide_arm_store_image()
            obs["start_image"] = start_img
            obs["end_image"] = end_img

        traj_ok = True
        if not self.env.valid_rollout():
            traj_ok = False
        elif 'rejection_sample' in self._hyperparams:
            if self._hyperparams['rejection_sample'] < i_tr and not self.env.goal_reached():
                traj_ok = False

        return traj_ok, obs, policy_outputs

    def save_goal_image_conf(self, traj):
        div = .05
        quantized = np.around(traj.score/div)
        best_score = np.min(quantized)
        for i in range(traj.score.shape[0]):
            if quantized[i] == best_score:
                first_best_index = i
                break

        print('best_score', best_score)
        print('allscores', traj.score)
        print('goal index: ', first_best_index)

        goalimage = traj.images[first_best_index]
        goal_ballpos = np.concatenate([traj.X_full[first_best_index], np.zeros(2)])  #set velocity to zero

        goal_object_pose = traj.Object_pos[first_best_index]

        img = Image.fromarray(goalimage)

        dict = {}
        dict['goal_image'] = goalimage
        dict['goal_ballpos'] = goal_ballpos
        dict['goal_object_pose'] = goal_object_pose

        pickle.dump(dict, open(self._hyperparams['save_goal_image'] + '.pkl', 'wb'))
        img.save(self._hyperparams['save_goal_image'] + '.png',)

    def eval_action(self, traj, t):
        if 'ztarget' in self._hyperparams:
            obj_z = traj.Object_full_pose[t, 0, 2]
            pos_score = np.abs(obj_z - self._hyperparams['ztarget'])
            return pos_score, 0.
        abs_distances = []
        abs_angle_dist = []
        for i_ob in range(self.num_objects):
            goal_pos = self.goal_obj_pose[i_ob, :3]
            curr_pos = traj.Object_full_pose[t, i_ob, :3]
            abs_distances.append(np.linalg.norm(goal_pos - curr_pos))

            goal_quat = Quaternion(self.goal_obj_pose[i_ob, 3:])
            curr_quat = Quaternion(traj.Object_full_pose[t, i_ob, 3:])
            diff_quat = curr_quat.conjugate*goal_quat
            abs_angle_dist.append(np.abs(diff_quat.radians))

        return np.array(abs_distances), np.array(abs_angle_dist)

    def save_gif(self, itr, overlay=False):
        if self.traj_points is not None and overlay:
            colors = [tuple([np.random.randint(0, 256) for _ in range(3)]) for __ in range(self.num_objects + 1)]
            for pnts, img in zip(self.traj_points, self.large_images_traj):
                for i in range(self.num_objects + 1):
                    center = tuple([int(np.round(pnts[i, j])) for j in (1, 0)])
                    cv2.circle(img, center, 10, colors[i], -1)

        file_path = self._hyperparams['record']
        npy_to_gif(self.large_images_traj, file_path +'/video{}'.format(itr))

    def plot_ctrls(self, i_tr):
        # a = plt.gca()
        self.hf_qpos_l = np.stack(self.hf_qpos_l, axis=0)
        self.hf_target_qpos_l = np.stack(self.hf_target_qpos_l, axis=0)
        tmax = self.hf_target_qpos_l.shape[0]

        if not os.path.exists(self._hyperparams['record']):
            os.makedirs(self._hyperparams['record'])
        for i in range(self.adim):
            plt.subplot(self.adim,1,i+1)
            plt.plot(list(range(tmax)), self.hf_qpos_l[:,i], label='q_{}'.format(i))
            plt.plot(list(range(tmax)), self.hf_target_qpos_l[:, i], label='q_target{}'.format(i))
            plt.legend()
        plt.savefig(self._hyperparams['record'] + '/ctrls{}.png'.format(i_tr))
        plt.close()

    def plot_pix_dist(self, planstat):
        plt.figure()
        pix_dist = np.stack(self.pix_dist, -1)

        best_cost_perstep = planstat['best_cost_perstep']

        nobj = self.num_objects
        nplot = self.ncam*nobj
        for icam in range(self.ncam):
            for p in range(nobj):
                plt.subplot(1,nplot, 1 + icam*nobj+p)
                plt.plot(pix_dist[icam, p], label='gtruth')
                plt.plot(best_cost_perstep[icam,p], label='pred')

        plt.legend()
        plt.savefig(self._hyperparams['record'] + '/pixel_distcost.png')

    def _init(self):
        """
        Set the world to a given model
        """
        return


