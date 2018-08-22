""" This file defines the linear Gaussian policy class. """
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
import copy
from .cem_controller_base import CEM_Controller_Base
from python_visual_mpc.visual_mpc_core.agent.general_agent import resize_store


class CEM_Controller_Sim(CEM_Controller_Base):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(CEM_Controller_Sim, self).__init__(ag_params, policyparams)

    def _default_hparams(self):
        default_dict = {
            'len_pred':15
        }

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def create_sim(self):
        env_type, env_params = self.agentparams['env']
        self.env = env_type(env_params, self.current_reset_state)
        self.env.set_goal_obj_pose(self._goal_pos)

    def eval_action(self):
        return self.env.get_distance_score()

    def get_rollouts(self, actions, cem_itr, itr_times):
        all_scores = np.empty(self.M, dtype=np.float64)
        image_list = []
        for smp in range(self.M):
            self.env.reset(self.current_reset_state)
            score, images = self.sim_rollout(actions[smp])
            image_list.append(images)
            # print('score', score)
            per_time_multiplier = np.ones([len(score)])
            per_time_multiplier[-1] = self._hp.finalweight
            all_scores[smp] = np.sum(per_time_multiplier*score)

            # if smp % 10 == 0 and self.verbose:
            #     self.save_gif(images, '_{}'.format(smp))

        if self.verbose:
            bestindices = all_scores.argsort()[:self.K]
            images = np.concatenate(image_list, axis=0)
            images = images[:,:,0]  # only first cam
            images = images[bestindices]
            pdb.set_trace()
            vid = []
            for t in range(self.naction_steps * self.repeat):
                row = np.concatenate(np.split(images[:,t], images.shape[0], axis=0), axis=1)
                vid.append(row)
            self.save_gif(vid, 't{}_iter{}'.format(self.t, cem_itr))
        return all_scores

    def old_get_rollouts(self, actions, cem_itr, itr_times):
        all_scores = np.empty(self.M, dtype=np.float64)
        image_list = []
        for smp in range(self.M):
            self.reset_mujoco_env()
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
        initial_env_obs, _ = self.env.reset()
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

    def old_sim_rollout(self, actions):
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

    def act(self, i_tr, t, state, object_qpos, goal_pos, reset_state):
        self.current_reset_state = reset_state
        self.current_reset_state['state'] = state[t]
        self.current_reset_state['object_qpos'] = object_qpos[t]
        self._goal_pos = goal_pos

        if t == 0:
            self.create_sim()
        return super(CEM_Controller_Sim, self).act(i_tr, t)
