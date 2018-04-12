import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
import  matplotlib.pyplot as plt
class DeterministicGraspPolicy(Policy):
    def __init__(self, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.min_lift = agentparams.get('min_z_lift', 0.05)

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
        self.grasp = False
        self.lift = False

        if 'iterations' in self.policyparams:
            self.niter = self.policyparams['iterations']
        else: self.niter = 10  # number of iterations

    def setup_CEM_model(self, t, init_model):
        if t == 0:
            if 'gen_xml' in self.agentparams:

                self.CEM_model = MjSim(load_model_from_path(self.agentparams['gen_xml_fname']))
            else:
                self.CEM_model = MjSim(load_model_from_path(self.agentparams['filename']))

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
        sim_state = self.CEM_model.get_state()
        sim_state.qpos[:] = self.initial_qpos.copy()
        sim_state.qvel[:] = self.initial_qvel.copy()
        self.CEM_model.set_state(sim_state)

        self.prev_target = self.CEM_model.data.qpos[:self.adim].squeeze()
        self.target = self.CEM_model.data.qpos[:self.adim].squeeze()


        self.viewer_refresh()

    def viewer_refresh(self):
        if 'debug_viewer' in self.policyparams and self.policyparams['debug_viewer']:
            self.viewer.loop_once()

    def perform_CEM(self, targetxy):
        print('Beginning CEM')
        ang_dis_mean = self.policyparams['init_mean'].copy()
        ang_dis_cov = self.policyparams['init_cov'].copy()
        scores = np.zeros(self.M)
        best_score, best_ang, best_xy = -1, None, None

        for n in range(self.niter):
            ang_disp_samps = np.random.multivariate_normal(ang_dis_mean, ang_dis_cov, size=self.M)

            for s in range(self.M):
                print('On iter', n, 'sample', s)
                self.reset_CEM_model()
                move = True
                drop = False
                grasp = False
                g_start = 0
                lift = False

                angle_delta = ang_disp_samps[s, 0]
                targetxy_delta = targetxy + ang_disp_samps[s, 1:]
                for t in range(self.n_actions):
                    angle_action = np.zeros(self.adim)
                    cur_xy = self.CEM_model.data.qpos[:2].squeeze()

                    if move and np.linalg.norm(targetxy_delta - cur_xy, 2) <= self.policyparams['drop_thresh']:
                        move = False
                        drop = True

                    if drop and self.CEM_model.data.qpos[2] <= -0.079:
                        drop = False
                        grasp = True
                        g_start = t
                    if grasp and t - g_start > 2:
                        grasp = False
                        lift = True
                    if move:
                        angle_action[:2] = targetxy_delta
                        angle_action[2] = self.agentparams['ztarget']
                        angle_action[3] = angle_delta
                        angle_action[4] = -100
                    elif drop:
                        angle_action[:2] = targetxy_delta
                        angle_action[2] = -0.08
                        angle_action[3] = angle_delta
                        angle_action[4] = -100
                    elif grasp:
                        angle_action[:2] = targetxy_delta
                        angle_action[2] = -0.08
                        angle_action[3] = angle_delta
                        angle_action[4] = 21
                    elif lift:
                        angle_action[:2] = targetxy_delta
                        angle_action[2] = self.agentparams['ztarget']
                        angle_action[3] = angle_delta
                        angle_action[4] = 21

                    self.step_model(angle_action)
                # print 'final z', self.CEM_model.data.qpos[8].squeeze(), 'with angle', angle_samps[s]

                scores[s] = self.CEM_model.data.qpos[8].squeeze() - 0.1 * np.abs(angle_delta)

                if 'stop_iter_thresh' in self.policyparams and scores[s] > self.policyparams['stop_iter_thresh']:
                    return ang_disp_samps[s, 0], ang_disp_samps[s, 1:]
                # print 'score',scores[s]

            best_scores = np.argsort(-scores)[:self.K]

            if scores[best_scores[0]] > best_score or best_ang is None:

                print('best', scores[best_scores[0]])

                best_score = scores[best_scores[0]]
                best_ang = ang_disp_samps[best_scores[0], 0]
                best_xy = ang_disp_samps[best_scores[0], 1:]



            ang_dis_mean = np.mean(ang_disp_samps[best_scores, :], axis = 0)
            ang_dis_cov = np.cov(ang_disp_samps[best_scores, :].T)

        return best_ang, best_xy

    def step_model(self, input_actions):
        pos_clip = self.agentparams['targetpos_clip']

        self.prev_target = self.target.copy()
        self.target = input_actions.copy()
        self.target = np.clip(self.target, pos_clip[0], pos_clip[1])

        for s in range(self.agentparams['substeps']):
            step_action = s / float(self.agentparams['substeps']) * (self.target - self.prev_target) + self.prev_target
            self.CEM_model.data.ctrl[:] = step_action
            self.CEM_model.step()

        self.viewer_refresh()

        #print "end", self.CEM_model.data.qpos[:4].squeeze()

    def act(self, traj, t, init_model = None, goal_object_pose = None, hyperparams = None):
        self.setup_CEM_model(t, init_model)

        if t == 0:
            self.moveto = True
            self.drop = False
            self.lift = False
            self.grasp = False
            self.second_moveto = False
            self.final_drop = False
            self.switchTime = 0
            self.targetxy = traj.Object_pose[t, 0, :2]
            self.angle, self.disp = self.perform_CEM(self.targetxy)
            self.targetxy += self.disp

        if self.lift and traj.X_full[t, 2] >= self.min_lift:
            self.second_moveto = True
            self.lift = False
            self.targetxy = 0.9 * np.random.uniform(size=(2)) - 0.45

        if self.grasp and self.switchTime > 0:
            print('lifting at time', t, '!', 'have z', traj.X_full[t, 2])

            self.grasp = False
            self.lift = True

        if self.drop and traj.X_full[t, 2] <= -0.079:
            print('grasping at time', t, '!', 'have z', traj.X_full[t, 2])
            self.drop = False
            self.grasp = True

        if self.moveto and np.linalg.norm(traj.X_full[t, :2] - self.targetxy, 2) <= self.policyparams['drop_thresh']:
            if self.switchTime >= 0:
                print('swapping at time', t, '!')
                self.moveto = False
                self.drop = True
                self.switchTime = 0
            else:
                self.switchTime += 1

        if self.second_moveto and np.linalg.norm(traj.X_full[t, :2] - self.targetxy, 2) <= self.policyparams['drop_thresh']:
            self.second_moveto = False
            self.final_drop = True
            self.switchTime = 0

        actions = np.zeros(self.adim)
        if self.moveto or self.second_moveto:
            delta = self.targetxy - traj.target_qpos[t, :2]
            norm = np.sqrt(np.sum(np.square(delta)))
            if norm > self.policyparams['max_norm']:
                actions[:2] = traj.target_qpos[t, :2] + delta / norm * self.policyparams['max_norm']
            else:
                actions[:2] = traj.target_qpos[t, :2] + delta
            actions[2] = self.agentparams['ztarget']
            actions[3] = self.angle

            if self.moveto:
                actions[-1] = -100
            else:
                actions[-1] = 21

            if 'xyz_std' in self.policyparams and t < 9:
                actions[:3] += self.policyparams['xyz_std'] * np.random.normal(size=3)


        elif self.drop:
            actions[:2] = self.targetxy
            actions[2] = -0.08
            actions[3] = self.angle
            actions[-1] = -100

        elif self.lift:
            actions[:2] = self.targetxy
            actions[2] = self.agentparams['ztarget']
            actions[3] = self.angle
            actions[-1] = 21


        elif self.grasp:
            actions[:2] = self.targetxy
            actions[2] = -0.08
            actions[3] = self.angle
            actions[-1] = 21
            self.switchTime += 1
        elif self.final_drop:

            if self.switchTime > 1:
                print('on final drop, open')
                actions[:2] = self.targetxy
                actions[2] = -0.08
                actions[3] = self.angle
                actions[-1] = -100
            else:
                print('on final close')
                actions[:2] = self.targetxy
                actions[2] = -0.08
                actions[3] = self.angle
                actions[-1] = 21
                self.switchTime += 1

        if 'debug_viewer' in self.policyparams and self.policyparams['debug_viewer'] and t == self.agentparams['T'] - 1:
            self.viewer.finish()

        if 'angle_std' in self.policyparams:
            actions[3] += self.policyparams['angle_std'] * np.random.normal()

        return actions - traj.target_qpos[t, :] * traj.mask_rel
