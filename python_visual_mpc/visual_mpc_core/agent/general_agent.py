""" This file defines an agent for the MuJoCo simulator environment. """
import copy
import numpy as np
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
import cv2
from python_visual_mpc.visual_mpc_core.algorithm.policy import get_policy_args


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
        self._goal_obj_pose = None
        self._goal_image = None
        self._reset_state = None
        self._setup_world(0)

    def _setup_world(self, itr):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        env_type, env_params = self._hyperparams['env']
        self.env = env_type(env_params, self._reset_state)

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

        return agent_data, obs_dict, policy_outs

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
                if (agent_img_height, agent_img_width) == env_obs['images'].shape[1:3]:
                    for i in range(env_obs['images'].shape[0]):
                        self._agent_cache['images'][t, i] = env_obs['images'][i]
                else:
                    for i in range(env_obs['images'].shape[0]):
                        self._agent_cache['images'][t, i] = cv2.resize(env_obs['images'][i], new_dims,
                                                                        interpolation=cv2.INTER_AREA)
            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  #only take first camera
                env_obs['obj_image_locations'] = (env_obs['obj_image_locations'] * agent_img_height / env_obs['images'].shape[1]).astype(np.int)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        if self._goal_image is not None:
            obs['goal_image'] = self._goal_image

        if self._goal_obj_pose is not None:
            obs['goal_pos'] = self._goal_obj_pose
            obs['goal_pix'] = self.env.get_goal_pix(agent_img_width)
        return obs

    def _required_rollout_metadata(self, agent_data, traj_ok, t):
        """
        Adds meta_data into the agent dictionary that is MANDATORY for later parts of pipeline
        :param agent_data: Agent data dictionary
        :param traj_ok: Whether or not rollout succeeded
        :return: None
        """
        agent_data['term_t'] = t - 1

        if self._goal_obj_pose is not None:
            agent_data['stats'] = self.env.eval()
        if self.env.has_goal():
            agent_data['goal_reached'] = self.env.goal_reached()
        agent_data['traj_ok'] = traj_ok

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
        agent_data, policy_outputs = {}, []

        # Take the sample.
        t = 0
        done = False
        initial_env_obs, _ = self.env.reset()
        obs = self._post_process_obs(initial_env_obs, True)
        policy.reset()

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

            if 'rejection_sample' in self._hyperparams and 'rejection_end_early' in self._hyperparams:
                if self._hyperparams['rejection_sample'] > i_tr and not self.env.goal_reached():
                    return {'traj_ok': False}, None, None

            if (self._hyperparams['T']-1) == t:
                done = True
            t += 1

        traj_ok = True
        if not self.env.valid_rollout():
            traj_ok = False

        if 'rejection_sample' in self._hyperparams:
            if self._hyperparams['rejection_sample'] > i_tr:
                assert self.env.has_goal(), 'Rejection sampling enabled but env has no goal'
                traj_ok = self.env.goal_reached()
            print('goal_reached', self.env.goal_reached())

        self._required_rollout_metadata(agent_data, traj_ok, t)
        return agent_data, obs, policy_outputs


    def save_gif(self, itr, overlay=False):
        if self.traj_points is not None and overlay:
            colors = [tuple([np.random.randint(0, 256) for _ in range(3)]) for __ in range(self.num_objects + 1)]
            for pnts, img in zip(self.traj_points, self.large_images_traj):
                for i in range(self.num_objects + 1):
                    center = tuple([int(np.round(pnts[i, j])) for j in (1, 0)])
                    cv2.circle(img, center, 10, colors[i], -1)

        file_path = self._hyperparams['record']
        npy_to_gif(self.large_images_traj, file_path +'/video{}'.format(itr))

    def _init(self):
        """
        Set the world to a given model
        """
        self.large_images_traj, self.traj_points= [], None



