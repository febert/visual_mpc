""" This file defines an agent for the MuJoCo simulator environment. """
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import shutil
import pickle as pkl
import copy
import numpy as np
import pickle
from PIL import Image
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
import os
import cv2
from python_visual_mpc.visual_mpc_core.algorithm.policy import get_policy_args
import pdb

def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class Image_Exception(Exception):
    def __init__(self):
        pass


class GeneralAgent(object):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.T = self._hyperparams['T']
        self.goal_obj_pose = None
        self.goal_images = None
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
        self._setup_world(0)

    def _setup_world(self, itr):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """

        if 'start_goal_confs' in self._hyperparams:
            reset_state = self._load_raw_data(itr)
        else: reset_state = None
        env_type, env_params = self._hyperparams['env']
        env_params['reset_state'] = reset_state
        self.env = env_type(env_params)

        self._hyperparams['adim'] = self.adim = self.env.adim
        self._hyperparams['sdim'] = self.sdim = self.env.sdim
        self._hyperparams['ncam'] = self.ncam = self.env.ncam
        self.num_objects = self.env.num_objects

    def sample(self, policy, i_tr):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """

        if "gen_xml" in self._hyperparams:
            if i_tr % self._hyperparams['gen_xml'] == 0 and i_tr > 0:
                self._setup_world(i_tr)

        traj_ok, obs_dict, policy_outs, agent_data = False, None, None, None
        i_trial = 0
        imax = 100
        while not traj_ok and i_trial < imax:
            i_trial += 1
            try:
                agent_data, obs_dict, policy_outs = self.rollout(policy, i_trial)
                traj_ok = agent_data['traj_ok']
            except Image_Exception:
                traj_ok = False

        print('needed {} trials'.format(i_trial))

        if 'make_final_gif' in self._hyperparams or 'make_final_gif_pointoverlay' in self._hyperparams:
            self.save_gif(i_tr, 'make_final_gif_pointoverlay' in self._hyperparams)

        if 'verbose' in self._hyperparams:
            self.plot_ctrls(i_tr)
            
        agent_data['stat_prop'] = self.env.obj_stat_prop
        return agent_data, obs_dict, policy_outs

    def hide_arm_store_image(self):
        highres_image = self.env.snapshot_noarm()
        target_dim = (self._hyperparams['image_width'], self._hyperparams['image_height'])
        return cv2.resize(highres_image, target_dim, interpolation=cv2.INTER_AREA)

    def _post_process_obs(self, env_obs, initial_obs=False):
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
                self.large_images_traj.append(env_obs['images'][0])  #only take first camera
                new_dims = (agent_img_width, agent_img_height)
                for i in range(env_obs['images'].shape[0]):
                    self._agent_cache['images'][t, i] = cv2.resize(env_obs['images'][i], new_dims,
                                                                    interpolation=cv2.INTER_AREA)

            # TODO: seems to be redundant with get_dsig_pix
            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  #only take first camera
                env_obs['obj_image_locations'] = (env_obs['obj_image_locations'] * agent_img_height / env_obs['images'].shape[1]).astype(np.int)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        obs['goal_image'] = self.goal_images
        obs['goal_pos'] = self.goal_obj_pose

        if self.goal_obj_pose is not None:
            obs['goal_pix'] = self.env.get_goal_pix(agent_img_width)

        obs['desig_pix'] = env_obs['obj_image_locations']

        print('desig pix:', obs['desig_pix'])

        return obs

    def _required_rollout_metadata(self, agent_data, traj_ok):
        """
        Adds meta_data into the agent dictionary that is MANDATORY for later parts of pipeline
        :param agent_data: Agent data dictionary
        :param traj_ok: Whether or not rollout succeeded
        :return: None
        """
        if self.env.has_goal():
            agent_data['goal_reached'] = self.env.goal_reached()
        agent_data['traj_ok'] = traj_ok
        agent_data['stat_prop'] = self.env.obj_stat_prop

    def rollout(self, policy, i_tr):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_tr: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        self._init()
        agent_img_height, agent_img_width = self._hyperparams['image_height'], self._hyperparams['image_width']
        self.env.goal_obj_pose = self.goal_obj_pose

        agent_data, policy_outputs = {}, []
        agent_data['stats'] = {}

        # Take the sample.
        t = 0
        done = False
        self.large_images_traj, self.traj_points= [], None
        obs = self._post_process_obs(self.env.reset(), True)

        while not done:
            """
            Currently refactoring the agent loop.
            Many features are being moved from agent into environment
            As a result many designated pixel related functions do not work
            This has implications for running MPC in sim
            """
            pi_t = policy.act(**get_policy_args(policy, obs, t, i_tr))
            policy_outputs.append(pi_t)

            try:
                obs = self._post_process_obs(self.env.step(copy.deepcopy(pi_t['actions'])))
            except ValueError:
                return {'traj_ok': False}, None, None

            if self.goal_obj_pose is not None:
                agent_data['stats'] = self.env.eval()

            if (self._hyperparams['T']-1) == t:
                done = True
            if done:
                agent_data['stats']['term_t'] = t
            t += 1

        traj_ok = True
        if not self.env.valid_rollout():
            traj_ok = False

        if 'rejection_sample' in self._hyperparams:
            if self._hyperparams['rejection_sample'] > i_tr:
                assert self.env.has_goal(), 'Rejection sampling enabled but env has no goal'
                traj_ok = self.env.goal_reached()
                print('reject test', traj_ok)

        self._required_rollout_metadata(agent_data, traj_ok)
        return agent_data, obs, policy_outputs


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


    def _load_raw_data(self, itr):
        """
        doing the reverse of save_raw_data
        :param itr:
        :return:
        """
        ngroup = 1000
        igrp = itr // ngroup
        group_folder = self._hyperparams['start_goal_confs'] + '/traj_group{}'.format(igrp)
        traj_folder = group_folder + '/traj{}'.format(itr)

        print('reading from: ', traj_folder)
        if 'num_load_steps' in self._hyperparams:
            num_images = self._hyperparams['num_load_steps']
        else:
            num_images = 2

        obs_dict = {}
        self.goal_images = np.zeros([num_images, self.ncam, self._hyperparams['image_height'], self._hyperparams['image_width'], 3])
        for t in range(num_images):  #TODO detect number of images automatically in folder
            for i in range(self.ncam):
                self.goal_images[t, i] = cv2.imread('{}/images{}/im_{}.png'.format(traj_folder, i, t))[...,::-1]
        self.goal_images = self.goal_images.astype(np.float32)/255.
        with open('{}/agent_data.pkl'.format(traj_folder), 'rb') as file:
            agent_data = pkl.load(file)
        with open('{}/obs_dict.pkl'.format(traj_folder), 'rb') as file:
            obs_dict.update(pkl.load(file))
        reset_state = {'object_qpos':obs_dict['object_qpos'][0], 'state':obs_dict['state'][0], 'stat_prop':agent_data['stat_prop']}

        self.goal_obj_pose = obs_dict['object_qpos'][-1]
        return reset_state

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


