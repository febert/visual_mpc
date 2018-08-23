""" This file defines the linear Gaussian policy class. """
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
import copy
from .cem_controller_base import CEM_Controller_Base
from python_visual_mpc.visual_mpc_core.agent.general_agent import resize_store
import ray

@ray.remote
class SimWorker(object):
    def __init__(self):
        print('created worker')
        pass

    def create_sim(self, agentparams, reset_state, goal_pos, finalweight, len_pred):
    # def create_sim(self, a):
        print('create sim')
        self.M = None
        self.agentparams = agentparams
        self.current_reset_state = reset_state
        self._goal_pos = goal_pos
        self.len_pred = len_pred
        self.finalweight = finalweight
        env_type, env_params = self.agentparams['env']
        self.env = env_type(env_params, self.current_reset_state)
        self.env.set_goal_obj_pose(self._goal_pos)

    def eval_action(self):
        return self.env.get_distance_score()

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        """
        Copied from general_agent.py !!

        Handles conversion from the environment observations, to agent observation
        space. Observations are accumulated over time, and images are resized to match
        the given image_heightximage_width dimensions.

        Original images from cam index 0 are added to buffer for saving gifs (if needed)

        Data accumlated over time is cached into an observation dict and returned. Data specific to each
        time-step is returned in agent_data

        :param env_obs: observations dictionary returned from the environment
        :param initial_obs: Whether or not this is the first observation in rollout
        :return: obs: dictionary of observations up until (and including) current timestep
        """
        agent_img_height = self.agentparams['image_height']
        agent_img_width = self.agentparams['image_width']

        if initial_obs:
            T = self.len_pred + 1
            self._agent_cache = {}
            for k in env_obs:
                if k == 'images':
                    if 'obj_image_locations' in env_obs:
                        self.traj_points = []
                    n_cams = env_obs['images'].shape[0]
                    self._agent_cache['images'] = np.zeros((T, n_cams, agent_img_height, agent_img_width, 3),
                                                           dtype = np.uint8)
                elif isinstance(env_obs[k], np.ndarray):
                    obs_shape = [T] + list(env_obs[k].shape)
                    self._agent_cache[k] = np.zeros(tuple(obs_shape), dtype=env_obs[k].dtype)
                else:
                    self._agent_cache[k] = []
            self._cache_cntr = 0

        t = self._cache_cntr
        self._cache_cntr += 1

        point_target_width = float(self.agentparams.get('point_space_width', agent_img_width))
        obs = {}
        for k in env_obs:
            if k == 'images':
                resize_store(t, self._agent_cache['images'], env_obs['images'])
            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  #only take first camera
                env_obs['obj_image_locations'] = np.round((env_obs['obj_image_locations'] *
                                                           point_target_width / env_obs['images'].shape[2])).astype(np.int64)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        if 'obj_image_locations' in env_obs:
            agent_data['desig_pix'] = env_obs['obj_image_locations']
        return obs

    def sim_rollout_with_retry(self, actions):
        done = False
        while not done:
            try:
                costs, images = self.sim_rollout(actions)
                done = True
            except ValueError:
                print('sim error retrying')
        return costs, images

    def sim_rollout(self, actions):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_tr: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """

        agent_data = {}

        # Take the sample.
        t = 0
        done = False
        initial_env_obs, _ = self.env.reset(self.current_reset_state)
        obs = self._post_process_obs(initial_env_obs, agent_data, True)

        costs = []
        while not done:
            """
            Every time step send observations to policy, acts in environment, and records observations

            Policy arguments are created by
                - populating a kwarg dict using get_policy_arg
                - calling policy.act with given dictionary

            Policy returns an object (pi_t) where pi_t['actions'] is an action that can be fed to environment
            Environment steps given action and returns an observation
            """

            try:
                obs = self._post_process_obs(self.env.step(actions[t]), agent_data)
            except ValueError:
                return {'traj_ok': False}, None, None

            if (self.len_pred - 1) == t:
                done = True
            t += 1

            costs.append(self.eval_action())

        return costs, obs['images']


    def perform_rollouts(self, actions):
        self.M = actions.shape[0]
        all_scores = np.empty(self.M, dtype=np.float64)
        image_list = []

        for smp in range(self.M):
            score, images = self.sim_rollout_with_retry(actions[smp])
            image_list.append(images[1:].squeeze())
            # print('score', score)
            per_time_multiplier = np.ones([len(score)])
            per_time_multiplier[-1] = self.finalweight
            all_scores[smp] = np.sum(per_time_multiplier*score)

        return np.stack(image_list, 0), np.stack(all_scores, 0)


class CEM_Controller_Sim(CEM_Controller_Base):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(CEM_Controller_Sim, self).__init__(ag_params, policyparams)
        self.parallel = True
        if self.parallel:
            ray.init()

    def _default_hparams(self):
        default_dict = {
            'len_pred':15,
            'num_workers':40,
        }

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def create_sim(self):
        self.workers = []
        if self.parallel:
            self.n_worker = self._hp.num_workers
        else:
            self.n_worker = 1

        for i in range(self.n_worker):
            if self.parallel:
                self.workers.append(SimWorker.remote())
            else:
                self.workers.append(SimWorker())

        id_list = []
        for i, worker in enumerate(self.workers):
            if self.parallel:
                a = 0
                # id_list.append(worker.create_sim.remote(a))
                id_list.append(worker.create_sim.remote(self.agentparams, self.reset_state, self.goal_pos, self._hp.finalweight, self.len_pred))
                # id_list.append(worker.create_sim.remote())
            else:
                return worker.create_sim(self.agentparams, self.reset_state, self.goal_pos, self._hp.final_weight, self.len_pred)

        if self.parallel:
            # blocking call
            for id in id_list:
                ray.get(id)

    def get_rollouts(self, actions, cem_itr, itr_times):
        images, all_scores = self.sim_rollout_parallel(actions)
        if self.verbose:
            bestindices = all_scores.argsort()[:self.K]
            images = images[bestindices]
            vid = []
            for t in range(self.naction_steps * self.repeat):
                row = np.concatenate(np.split(images[:,t], images.shape[0], axis=0), axis=2).squeeze()
                vid.append(row)
            self.save_gif(vid, 't{}_iter{}'.format(self.t, cem_itr))
        return all_scores


    def save_gif(self, images, name):
        file_path = self.agentparams['record']
        npy_to_gif(images, file_path +'/video' + name)

    def sim_rollout_parallel(self, actions):
        n_smp = actions.shape[0]

        per_worker = int(n_smp / np.float32(self.n_worker))

        id_list = []
        for i, worker in enumerate(self.workers):
            subset_actions = actions[per_worker*i:(i+1)*per_worker]
            if self.parallel:
                id_list.append(worker.perform_rollouts.remote(subset_actions))
            else:
                return worker.perform_rollouts(subset_actions)

        # blocking call
        image_list, scores_list = [], []
        for id in id_list:
            images, scores = ray.get(id)
            image_list.append(images)
            scores_list.append(scores)
        scores = np.concatenate(scores_list, axis=0)
        images = np.concatenate(image_list, axis=0)
        return images, scores

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

    def act(self, t, i_tr, state, object_qpos, goal_pos, reset_state):
        self.reset_state = reset_state
        self.reset_state['state'] = state[t]
        self.reset_state['object_qpos'] = object_qpos[t]
        self.goal_pos = goal_pos

        if t == 0:
            self.create_sim()
        return super(CEM_Controller_Sim, self).act(t, i_tr)
