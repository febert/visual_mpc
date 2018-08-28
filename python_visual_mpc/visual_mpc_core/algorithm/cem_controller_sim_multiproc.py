""" This file defines the linear Gaussian policy class. """
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
import copy
from .cem_controller_base import CEM_Controller_Base
from python_visual_mpc.visual_mpc_core.agent.general_agent import resize_store
import traceback
from python_visual_mpc.visual_mpc_core.algorithm.utils.cem_controller_utils import sample_actions
import time


from multiprocessing import Process, Queue, Manager



class SimWorker(Process):
    def __init__(self, id, queue, agentparams, reset_state, goal_pos, finalweight, len_pred,
                   naction_steps, discrete_ind, action_bound, adim, repeat, initial_std):
        print('created worker {}'.format(id))
        super(SimWorker, self).__init__()
        self.queue = queue
        self.id = id
        # self.agentparams = agentparams
        # self._goal_pos = goal_pos
        # self.len_pred = len_pred
        # self.finalweight = finalweight
        # self.current_reset_state = reset_state
        # env_type, env_params = self.agentparams['env']
        # # env_params['verbose_dir'] = '/home/frederik/Desktop/'
        # self.env = env_type(env_params, self.current_reset_state)
        # self.env.set_goal_obj_pose(self._goal_pos)
        #
        # # hyperparams passed into sample_action function
        # class HP(object):
        #     def __init__(self, naction_steps, discrete_ind, action_bound, adim, repeat, initial_std):
        #         self.naction_steps = naction_steps
        #         self.discrete_ind = discrete_ind
        #         self.action_bound = action_bound
        #         self.adim = adim
        #         self.repeat = repeat
        #         self.initial_std = initial_std
        # self.hp = HP(naction_steps, discrete_ind, action_bound, adim, repeat, initial_std)
        # print('finished creating sim')

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

    def sim_rollout_with_retry(self, curr_qpos, curr_qvel, mean, sigma):
        done = False
        attempts = 0
        while not done:
            try:
                attempts += 1
                actions = sample_actions(mean, sigma, self.hp, 1)
                costs, images = self.sim_rollout(curr_qpos, curr_qvel, actions[0])
                done = True
            except Exception as err:
                traceback.print_tb(err.__traceback__)
                print('sim error retrying')
        if attempts > 1:
            print('needed {} attempts'.format(attempts))
        return costs, images

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

    def run(self):
        print('launched {}'.format(self.id))

        for inp in iter(self.queue.get, None):
            req_id = inp['id']
            print('{} loaded queue reqid {}'.format(self.id, req_id))
            time.sleep(3)
            print('{} loaded queue reqid {} done'.format(self.id, req_id))
            # return_dict = inp['return_dict']
            # M = inp['M']
            # all_scores = np.empty(M, dtype=np.float64)
            # image_list = []
            #
            # for smp in range(M):
            #     score, images = self.sim_rollout_with_retry(inp['curr_qpos'], inp['curr_qvel'], inp['mean'], inp['sigma'])
            #     image_list.append(images.squeeze())
            #     # print('score', score)
            #     per_time_multiplier = np.ones([len(score)])
            #     per_time_multiplier[-1] = self.finalweight
            #     all_scores[smp] = np.sum(per_time_multiplier*score)
            #
            # print('filling return dict', id)
            # return_dict[id] = [np.stack(image_list, 0), np.stack(all_scores, 0)]


class CEM_Controller_Sim(CEM_Controller_Base):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(CEM_Controller_Sim, self).__init__(ag_params, policyparams)
        self.parallel = True
        # self.parallel = False

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

        self.procs = []
        for i in range(self.n_worker):
            if self.parallel:
                self.request_queue = Queue()
                self.procs.append(SimWorker(i, self.request_queue, self.agentparams, self.reset_state, self.goal_pos, self._hp.finalweight, self.len_pred,
                                        self.naction_steps, self._hp.discrete_ind, self._hp.action_bound, self.adim, self.repeat, self._hp.initial_std))
                self.procs[-1].start()
            else:
                self.worker = SimWorker(i, self.request_queue, self.agentparams, self.reset_state, self.goal_pos, self._hp.finalweight, self.len_pred,
                                        self.naction_steps, self._hp.discrete_ind, self._hp.action_bound, self.adim, self.repeat, self._hp.initial_std).start()

        pdb.set_trace()

    def get_rollouts(self, _, cem_itr, itr_times):
        images, all_scores = self.sim_rollout_parallel()

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

    def sim_rollout_parallel(self):
        per_worker = int(self.M / np.float32(self.n_worker))
        if self.parallel:
            manager = Manager()
            return_dict = manager.dict()
            input_dict = {
                'return_dict':return_dict,
                'curr_qpos':self.qpos_full,
                'curr_qvel':self.qvel_full,
                'mean':self.mean,
                'sigma':self.sigma,
                'M':per_worker
            }
            for i in range(self.n_worker):
                input_dict['id'] = i
                self.request_queue.put(input_dict)

        else:
            return_dict = {}
            id = 0
            self.worker.perform_rollouts([id, return_dict, self.qpos_full, self.qvel_full, self.mean, self.sigma, per_worker])

        # for i in range(self.n_worker):
        #     self.request_queue.put(None)

        print('waiting for simulation to complete')
        pdb.set_trace()
        for p in self.procs:
            p.join()

        print('completed')
        image_list, scores_list = [], []
        for key in return_dict.keys():
            images, scores = return_dict[key]
            scores_list.append(scores)
            image_list.append(images)
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

    def act(self, t, i_tr, qpos_full, qvel_full, state, object_qpos, goal_pos, reset_state):
    # def act(self, t, i_tr, qpos_full, goal_pos, reset_state):
        self.qpos_full = qpos_full[t]
        self.qvel_full = qvel_full[t]
        self.goal_pos = goal_pos
        if t == 0:
            self.reset_state = reset_state
            self.create_sim()
        return super(CEM_Controller_Sim, self).act(t, i_tr)
