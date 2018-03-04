import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
import mujoco_py

class DeterministicGraspPolicy(Policy):
    def __init__(self, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.policyparams = policyparams
        self.adim = agentparams['adim']
        self.n_actions = policyparams['nactions']

        if 'num_samples' in self.policyparams:
            self.M = self.policyparams['num_samples']
        else: self.M = 20  # number of CEM Samples

        if 'best_to_take' in self.policyparams:
            self.K = self.policyparams['best_to_take']
        else: self.K = 5  # best samples to average for next sampling

        assert self.adim >= 4, 'must have at least x, y, z + gripper actions'

        self.moveto = True
        self.drop = False
        self.graspTime = 0
        self.grasp = False
        self.lift = False

        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

    def setup_CEM_model(self, t, init_model):
        if t == 0:
            if 'gen_xml' in self.agentparams:
                self.CEM_model = mujoco_py.MjModel(self.agentparams['gen_xml_fname'])
            else:
                self.CEM_model = mujoco_py.MjModel(self.agentparams['filename'])

            if 'debug_viewer' in self.policyparams and self.policyparams['debug_viewer']:
                # CEM viewer for debugging purposes
                gofast = True
                self.viewer = mujoco_py.MjViewer(visible=True,
                                                 init_width=700,
                                                 init_height=480,
                                                 go_fast=gofast)
                self.viewer.set_model(self.CEM_model)
                self.viewer.start()
                self.viewer.cam.camid = 0

        self.initial_qpos = init_model.data.qpos.copy()
        self.initial_qvel = init_model.data.qvel.copy()

        self.reset_CEM_model()

    def reset_CEM_model(self):
        self.CEM_model.data.qpos = self.initial_qpos.copy()
        self.CEM_model.data.qvel = self.initial_qvel.copy()
        self.viewer_refresh()

    def viewer_refresh(self):
        if 'debug_viewer' in self.policyparams and self.policyparams['debug_viewer']:
            self.viewer.loop_once()

    def perform_CEM(self, traj_t, traj):
        # print 'Beginning CEM'
        angle_samps = np.random.normal(0.0, 3.14 / 4, size = self.M)
        scores = np.zeros(self.M)
        best_score, best_ang = -1, None

        for n in range(self.niter):
            for s in range(self.M):
                # print 'On iter', n, 'sample', s
                self.reset_CEM_model()
                move = True
                drop = False
                grasp = False
                g_start = 0
                lift = False

                angle_actions = np.zeros((self.n_actions, self.adim))
                for t in range(self.n_actions):
                    cur_xy = self.CEM_model.data.qpos[:2].squeeze()

                    if move and np.linalg.norm(traj.Object_pose[traj_t, 0, :2] - cur_xy, 2) <= self.agentparams['drop_thresh']:
                        move = False
                        drop = True

                    if drop and self.CEM_model.data.qpos[2] <= -0.079:
                        drop = False
                        grasp = True
                        g_start = t
                    if grasp and t - g_start > 4:
                        grasp = False
                        lift = True
                    if move:
                        angle_actions[t, :2] = traj.Object_pose[traj_t, 0, :2]
                        angle_actions[t, 2] = self.agentparams['ztarget']
                        angle_actions[t, 3] = angle_samps[s]
                        angle_actions[t, 4] = -100
                    elif drop:
                        angle_actions[t, :2] = traj.Object_pose[traj_t, 0, :2]
                        angle_actions[t, 2] = -0.08
                        angle_actions[t, 3] = angle_samps[s]
                        angle_actions[t, 4] = -100
                    elif grasp:
                        angle_actions[t, :2] = traj.Object_pose[traj_t, 0, :2]
                        angle_actions[t, 2] = -0.08
                        angle_actions[t, 3] = angle_samps[s]
                        angle_actions[t, 4] = 10
                    elif lift:
                        angle_actions[t, :2] = traj.Object_pose[traj_t, 0, :2]
                        angle_actions[t, 2] = self.agentparams['ztarget']
                        angle_actions[t, 3] = 0.
                        angle_actions[t, 4] = 10

                    self.step_model(angle_actions[t])
                # print 'final z', self.CEM_model.data.qpos[8].squeeze(), 'with angle', angle_samps[s]

                scores[s] = self.CEM_model.data.qpos[8].squeeze() - 0.1 * np.abs(angle_samps[s])
                # print 'score',scores[s]

            best_scores = np.argsort(-scores)[:self.K]

            if scores[best_scores[0]] > best_score or best_ang is None:
                print scores[best_scores[0]]
                best_score = scores[best_scores[0]]
                best_ang = angle_samps[best_scores[0]]

            ang_mean = np.mean(angle_samps[best_scores])
            ang_var = np.var(angle_samps[best_scores])

            angle_samps = np.random.normal(ang_mean, np.sqrt(ang_var), size = self.M)

        return best_ang

    def step_model(self, actions):
        for _ in range(self.agentparams['substeps']):
            ctrl = actions.copy()

            ctrl[2] -= self.CEM_model.data.qpos[2].squeeze()

            self.CEM_model.data.ctrl = ctrl
            self.CEM_model.step()
        self.viewer_refresh()


    def act(self, traj, t, init_model = None):
        self.setup_CEM_model(t, init_model)

        if t == 0:
            self.moveto = True
            self.drop = False
            self.lift = False
            self.grasp = False
            self.graspTime = 0
            self.angle = self.perform_CEM(t, traj)

        if self.grasp and self.graspTime > 4:
            print 'lifting at time', t, '!', 'have z', traj.X_full[t, 2]
            self.grasp = False
            self.lift = True

        if self.drop and traj.X_full[t, 2] <= -0.079:
            print 'grasping at time', t, '!', 'have z', traj.X_full[t, 2]
            self.drop = False
            self.grasp = True

        if self.moveto and np.linalg.norm(traj.Object_pose[t, 0, :2] - traj.X_full[t, :2], 2) <= self.agentparams['drop_thresh']:
            print 'swapping at time', t, '!'
            self.moveto = False
            self.drop = True


        if self.moveto:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = self.agentparams['ztarget']
            actions[3] = self.angle
            actions[-1] = -100


        elif self.drop:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = -0.08
            actions[3] = self.angle
            actions[-1] = -100

            #self.sim_rollout(actions)

        elif self.lift:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = self.agentparams['ztarget']
            actions[3] = 0.
            actions[-1] = 10


        elif self.grasp:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = -0.08
            actions[3] = self.angle
            actions[-1] = 10. / 4 * self.graspTime
            self.graspTime += 1

        if 'debug_viewer' in self.policyparams and self.policyparams['debug_viewer'] and t == self.agentparams['T'] - 1:
            self.viewer.finish()

        return actions
