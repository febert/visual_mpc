""" This file defines the linear Gaussian policy class. """
import IPython
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
import copy
from .cem_controller_base import CEM_Controller_Base
from python_visual_mpc.visual_mpc_core.agent.general_agent import resize_store
import ray
import traceback


@ray.remote
class SimWorker(object):
    def __init__(self):
        print('created worker')
        pass

    def create_sim(self, agentparams, reset_state, goal_pos, finalweight, len_pred,
                   naction_steps, discrete_ind, action_bound, adim, repeat, initial_std):
        print('create sim')
        self.agentparams = agentparams
        self._goal_pos = goal_pos
        self.len_pred = len_pred
        self.finalweight = finalweight
        self.current_reset_state = reset_state
        env_type, env_params = self.agentparams['env']
        # env_params['verbose_dir'] = '/home/frederik/Desktop/'
        self.env = env_type(env_params, self.current_reset_state)
        self.env.set_goal_obj_pose(self._goal_pos)

        # hyperparams passed into sample_action function
        class HP(object):
            def __init__(self, naction_steps, discrete_ind, action_bound, adim, repeat, initial_std):
                self.naction_steps = naction_steps
                self.discrete_ind = discrete_ind
                self.action_bound = action_bound
                self.adim = adim
                self.repeat = repeat
                self.initial_std = initial_std
        self.hp = HP(naction_steps, discrete_ind, action_bound, adim, repeat, initial_std)

    def recreate_sim(self):
        env_type, env_params = self.agentparams['env']
        # env_params['verbose_dir'] = '/home/frederik/Desktop/'
        self.env = env_type(env_params, self.current_reset_state)
        self.env.set_goal_obj_pose(self._goal_pos)

    def eval_action(self):
        return self.env.get_distance_score()

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
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

    def sim_rollout(self, curr_qpos, curr_qvel, actions):
        agent_data = {}
        t = 0
        done = False
        # initial_env_obs, _ = self.env.reset(curr_reset_state)
        initial_env_obs , _ = self.env.qpos_reset(curr_qpos, curr_qvel)
        obs = self._post_process_obs(initial_env_obs, agent_data, initial_obs=True)
        costs = []
        while not done:
            # print('inner sim step', t)
            obs = self._post_process_obs(self.env.step(actions[t]), agent_data)
            if (self.len_pred - 1) == t:
                done = True
            t += 1
            costs.append(self.eval_action())
        return costs, obs['images']

    def perform_rollouts(self, curr_qpos, curr_qvel, actions, M):
        all_scores = np.empty(M, dtype=np.float64)
        image_list = []

        for smp in range(M):
            score, images = self.sim_rollout(curr_qpos, curr_qvel, actions[smp])
            image_list.append(images.squeeze())
            # print('score', score)
            per_time_multiplier = np.ones([len(score)])
            per_time_multiplier[-1] = self.finalweight
            all_scores[smp] = np.sum(per_time_multiplier*score)

        images = np.stack(image_list, 0)[:,1:].astype(np.float32)/255.
        return images, np.stack(all_scores, 0)


class CEM_Controller_Sim(CEM_Controller_Base):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(CEM_Controller_Sim, self).__init__(ag_params, policyparams)
        self.parallel = True
        # self.parallel = False
        if self.parallel:
            ray.init()

    def _default_hparams(self):
        default_dict = {
            'len_pred':15,
            'num_workers':10,
        }

        parent_params = super()._default_hparams()
        parent_params.ncam = 1
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
                id_list.append(worker.create_sim.remote(self.agentparams, self.curr_sim_state, self.goal_pos, self._hp.finalweight, self.len_pred,
                                                        self.naction_steps, self._hp.discrete_ind, self._hp.action_bound, self.adim, self.repeat, self._hp.initial_std))
            else:
                return worker.create_sim(self.agentparams, self.curr_sim_state, self.goal_pos, self._hp.finalweight, self.len_pred,
                                         self.naction_steps, self._hp.discrete_ind, self._hp.action_bound, self.adim, self.repeat, self._hp.initial_std)
        if self.parallel:
            # blocking call
            for id in id_list:
                ray.get(id)


    def get_rollouts(self, actions, cem_itr, itr_times):
        images, all_scores = self.sim_rollout_parallel(actions)

        if self.verbose:
            self.save_gif(images, all_scores, cem_itr)
        return all_scores

    def save_gif(self, images, all_scores, cem_itr):
        bestindices = all_scores.argsort()[:self.K]
        images = (images[bestindices]*255.).astype(np.uint8)  # select cam0
        vid = []
        for t in range(self.naction_steps * self.repeat):
            row = np.concatenate(np.split(images[:,t], images.shape[0], axis=0), axis=2).squeeze()
            vid.append(row)

        name = 't{}_iter{}'.format(self.t, cem_itr)
        file_path = self.agentparams['record']
        npy_to_gif(vid, file_path +'/video' + name)


    def sim_rollout_parallel(self, actions):
        per_worker = int(self.M / np.float32(self.n_worker))
        id_list = []
        for i, worker in enumerate(self.workers):
            if self.parallel:
                actions_perworker = actions[i*per_worker:(i+1)*per_worker]
                id_list.append(worker.perform_rollouts.remote(self.qpos_full, self.qvel_full, actions_perworker, per_worker))
            else:
                images, scores_mjc = worker.perform_rollouts(self.qpos_full, self.qvel_full, actions, self.M)

        # blocking call
        if self.parallel:
            image_list, scores_list = [], []
            for id in id_list:
                images, scores_mjc = ray.get(id)
                image_list.append(images)
                scores_list.append(scores_mjc)
            scores_mjc = np.concatenate(scores_list, axis=0)
            images = np.concatenate(image_list, axis=0)

        scores = self.get_scores(images, scores_mjc)
        return images, scores

    def get_scores(self, images, scores_mjc):
        return scores_mjc

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

    def act(self, t, i_tr, qpos_full, qvel_full, state, object_qpos, goal_pos, reset_state):
    # def act(self, t, i_tr, qpos_full, goal_pos, reset_state):
        self.curr_sim_state = reset_state
        self.qpos_full = qpos_full[t]
        self.qvel_full = qvel_full[t]
        self.goal_pos = goal_pos

        if t == 0:
            self.create_sim()
        return super(CEM_Controller_Sim, self).act(t, i_tr)
