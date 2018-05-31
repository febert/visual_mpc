import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
import os
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif

"""
TODO: Replace the implementation here with a more extensible varient.
Right now lots of duplicated code when defining new scripted motions.
"""
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
        self.imgs = []
        self.iter = 0

    def setup_CEM_model(self, t, init_model):
        if t == 0:
            if 'gen_xml' in self.agentparams:

                self.CEM_model = MjSim(load_model_from_path(self.agentparams['gen_xml_fname']))
            else:
                self.CEM_model = MjSim(load_model_from_path(self.agentparams['filename']))



        self.initial_qpos = init_model.data.qpos.copy()
        self.initial_qvel = init_model.data.qvel.copy()

        self.reset_CEM_model()

    def reset_CEM_model(self):
        if len(self.imgs) > 0:
            print('saving iter', self.iter, 'with frames:', len(self.imgs))
            npy_to_gif(self.imgs, os.path.join(self.agentparams['record'], 'iter_{}'.format(self.iter)))
            self.iter += 1


        sim_state = self.CEM_model.get_state()
        sim_state.qpos[:] = self.initial_qpos.copy()
        sim_state.qvel[:] = self.initial_qvel.copy()
        self.CEM_model.set_state(sim_state)

        self.prev_target = self.CEM_model.data.qpos[:self.adim].squeeze().copy()
        self.target = self.CEM_model.data.qpos[:self.adim].squeeze().copy()

        for _ in range(5):
            self.step_model(self.target)

        self.imgs = []

    def viewer_refresh(self):
        if 'debug_viewer' in self.policyparams and self.policyparams['debug_viewer']:
            large_img = self.CEM_model.render(640, 480, camera_name="maincam")[::-1, :, :]
            self.imgs.append(large_img)

    def perform_CEM(self, targetxy):
        self.reset_CEM_model()

        if 'object_meshes' in self.agentparams:
            targetxy = self.CEM_model.data.sensordata[:2].squeeze().copy()
        print('Beginning CEM')
        ang_dis_mean = self.policyparams['init_mean'].copy()
        ang_dis_cov = self.policyparams['init_cov'].copy()
        scores = np.zeros(self.M)
        best_score, best_ang, best_xy = -1, None, None

        for n in range(self.niter):
            ang_disp_samps = np.random.multivariate_normal(ang_dis_mean, ang_dis_cov, size=self.M)

            for s in range(self.M):
                #print('On iter', n, 'sample', s)
                self.reset_CEM_model()
                move = True
                drop = False
                grasp = False
                g_start = 0
                lift = False

                angle_delta = ang_disp_samps[s, 0]
                targetxy_delta = targetxy + ang_disp_samps[s, 1:]
                print('init iter')
                print(targetxy)
                print(angle_delta, targetxy_delta)
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

                if 'object_meshes' in self.agentparams:
                    obj_z = self.CEM_model.data.sensordata[2].squeeze()
                else:
                    obj_z = self.CEM_model.data.qpos[8].squeeze()

                print('obj_z at iter', n * self.M + s, 'is', obj_z)

                scores[s] = obj_z

                if 'stop_iter_thresh' in self.policyparams and scores[s] > self.policyparams['stop_iter_thresh']:
                    return ang_disp_samps[s, 0], ang_disp_samps[s, 1:]
                # print 'score',scores[s]

            best_scores = np.argsort(-scores)[:self.K]

            if scores[best_scores[0]] > best_score or best_ang is None:

                #print('best', scores[best_scores[0]])

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

    def act(self, traj, t, init_model = None, goal_object_pose = None, hyperparams = None, goal_image = None):
        """
        Scripted pick->place->wiggle trajectory. There's probably a better way to script this
        but booleans will suffice for now.
        """
        self.setup_CEM_model(t, init_model)

        if t == 0:
            self.moveto = True
            self.drop = False
            self.lift = False
            self.grasp = False
            self.second_moveto = False
            self.final_drop = False
            self.wiggle = False
            self.switchTime = 0
            print ('start pose', traj.Object_pose[t, 0, :3])
            self.targetxy = traj.Object_pose[t, 0, :2]
            self.angle, self.disp = self.perform_CEM(self.targetxy)
            print('best angle', self.angle, 'best target', self.targetxy)
            self.targetxy += self.disp
            traj.desig_pos = np.zeros((2, 2))
            traj.desig_pos[0] = self.targetxy.copy()


        if self.lift and traj.X_full[t, 2] >= self.min_lift:
            self.lift = False
            
            if traj.Object_full_pose[t, 0, 2] > -0.01: #lift occursed
                self.second_moveto = True
                self.targetxy = np.random.uniform(-0.2, 0.2, size=2)
                print("LIFTED OBJECT!")
                print('dropping at', self.targetxy)
                traj.desig_pos[1] = self.targetxy.copy()
            else:
                self.wiggle = True
            
        if self.grasp and self.switchTime > 0:
            print('lifting at time', t, '!', 'have z', traj.X_full[t, 2])

            self.grasp = False
            self.lift = True

        if self.drop and (traj.X_full[t, 2] <= -0.079 or self.switchTime >= 2):
            print('grasping at time', t, '!', 'have z', traj.X_full[t, 2])
            self.drop = False
            self.grasp = True
            self.switchTime = 0

        if self.moveto and (np.linalg.norm(traj.X_full[t, :2] - self.targetxy, 2) <= self.policyparams['drop_thresh']):
            if self.switchTime > 0:
                print('stopping moveto at time', t, '!')
                print(traj.X_full[t, :2], self.targetxy)
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
            delta = np.zeros(3)
            delta[:2] = self.targetxy - traj.target_qpos[t, :2]

            if 'xyz_std' in self.policyparams:
                delta += self.policyparams['xyz_std'] * np.random.normal(size=3)

            norm = np.sqrt(np.sum(np.square(delta)))
            if norm > self.policyparams['max_norm']:
                delta *= self.policyparams['max_norm'] / norm

            actions[:3] = traj.target_qpos[t, :3] + delta
            actions[2] = self.agentparams['ztarget']
            actions[3] = self.angle

            if self.moveto:
                actions[-1] = -1
            else:
                actions[-1] = 1


            self.switchTime += 1


        elif self.drop:
            actions[:2] = self.targetxy
            actions[2] = -0.08
            actions[3] = self.angle
            actions[-1] = -1
            self.switchTime += 1

        elif self.lift:
            actions[:2] = self.targetxy
            actions[2] = self.agentparams['ztarget']
            actions[3] = self.angle
            actions[-1] = 1


        elif self.grasp:
            actions[:2] = self.targetxy
            actions[2] = -0.08
            actions[3] = self.angle
            actions[-1] = 1
            self.switchTime += 1
        elif self.final_drop:

            if self.switchTime > 0:
                print('opening')
                actions[:2] = self.targetxy
                actions[2] = -0.08
                actions[3] = self.angle
                actions[-1] = -1
                self.switchTime += 1

            if self.switchTime > 1:
                print('up')
                actions[:2] = self.targetxy
                actions[2] = self.agentparams['ztarget'] + np.random.uniform(-0.03, 0.05)
                actions[3] = self.angle + np.random.uniform(-np.pi / 4., np.pi / 4.)
                actions[-1] = -1
                self.final_drop = False
                self.wiggle = True
                self.switchTime = 0
            else:
                actions[:2] = self.targetxy
                actions[2] = -0.08
                actions[3] = self.angle
                actions[-1] = -1
                self.switchTime += 1


        elif self.wiggle:
            delta_vec = np.random.normal(size = 2)
            norm = np.sqrt(np.sum(np.square(delta_vec)))
            if norm > self.policyparams['max_norm']:
                delta_vec *= self.policyparams['max_norm'] / norm
            actions[:2] = np.clip(traj.target_qpos[t, :2] + delta_vec, -0.15, 0.15)
            delta_z = np.random.uniform(-0.08 - traj.target_qpos[t, 2], 0.3 - traj.target_qpos[t, 2])
            actions[2] = traj.target_qpos[t, 2] + delta_z
            actions[3] = traj.target_qpos[t, 3] + np.random.uniform(-0.1, 0.1)
            if np.random.uniform() > 0.5:
                actions[4] = -1
            else:
                actions[4] = 1

        if 'angle_std' in self.policyparams:
            actions[3] += self.policyparams['angle_std'] * np.random.normal()

        return actions - traj.target_qpos[t, :] * traj.mask_rel


class CluteredGraspPolicy(Policy):
    def __init__(self, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.z_hover = agentparams['ztarget'] #gripper's "up" state
        self.min_lift = agentparams.get('min_z_lift', 0.05) #gripper's "up" state

        self.policyparams = policyparams
        self.adim = agentparams['adim']
        self.window = policyparams['z_window'] #gripper can wiggle between z_hover +- window while "up"

        self.angle_window = policyparams['angle_window']
        self.xyz_std = policyparams['xyz_std']
        
        """
        Gripper will stay up for z = min(max_drop_time, X) [X~Geom(3/T) --> E[X] = T/3]
        After z steps the gripper will attempt a grasp
        """
        self.max_drop_time = 2 * agentparams['T'] // 3   #time until gripper makes grasp attempt
        self.drop_rate = 2 / agentparams['T']      #rate at which gripper drop

        
    def act(self, traj, t, init_model = None, goal_object_pose = None, hyperparams = None, goal_image = None):
        """
        Handles up -> down -> wiggle motion for random grasping (cluttered environment)
        """
        if t == 0:
            self.up = True
            self.down = False
            self.grasp = False
            self.lift = False
            self.wiggle = False
            self.counter = 0
            self.drop_wait = np.random.choice(3)
        
        if self.up and (t >= self.max_drop_time or np.random.uniform() <= self.drop_rate):
            print('down at', t)
            self.down = True
            self.up = False
        
        if self.down and self.counter > self.drop_wait:
            print('grasping at', t)
            self.down = False
            self.grasp = True
            self.counter = 0
        if self.grasp and self.counter > 0:
            print('lifting at', t)
            self.grasp = False
            self.lift = True
            self.counter = 0
        if self.lift and traj.X_full[t, 2] >= self.min_lift:
            self.lift = False
            self.wiggle = True
        
        actions = np.zeros(self.adim)

        if self.up:
            delta_vec = self.xyz_std * np.random.normal(size = 2)
            norm = np.sqrt(np.sum(np.square(delta_vec)))

            if norm > self.policyparams['max_norm']:
                delta_vec *= self.policyparams['max_norm'] / norm
            
            actions[:2] = np.clip(traj.target_qpos[t, :2] + delta_vec, -0.15, 0.15)
            actions[2] = self.z_hover + np.random.uniform(low = -self.window, high = self.window)

            actions[3] = traj.target_qpos[t, 3] + np.random.uniform(-self.angle_window, self.angle_window)
            actions[4] = -1
        elif self.down:
            actions[:2] = traj.X_full[t, :2]
            actions[2] = -0.08
            actions[3] = traj.X_full[t, 3]
            actions[-1] = -1
            self.counter += 1
        elif self.grasp:
            actions[:2] = traj.X_full[t, :2]
            actions[2] = -0.08
            actions[3] = traj.X_full[t, 3]
            actions[-1] = 1
            self.counter += 1
        elif self.lift:
            actions[:2] = traj.X_full[t, :2]
            actions[2] = self.z_hover
            actions[3] = traj.X_full[t, 3]
            actions[-1] = 1
        elif self.wiggle:
            delta_vec = self.xyz_std * np.random.normal(size = 2)
            norm = np.sqrt(np.sum(np.square(delta_vec)))
            if norm > self.policyparams['max_norm']:
                delta_vec *= self.policyparams['max_norm'] / norm
            
            actions[:2] = np.clip(traj.target_qpos[t, :2] + delta_vec, -0.15, 0.15)

            delta_z = np.random.uniform(-0.08 - traj.target_qpos[t, 2], 0.3 - traj.target_qpos[t, 2])
            actions[2] = traj.target_qpos[t, 2] + delta_z

            actions[3] = traj.target_qpos[t, 3] + np.random.uniform(-self.angle_window, self.angle_window)

            if np.random.uniform() > 0.5:
                actions[4] = -1
            else:
                actions[4] = 1
        return actions - traj.target_qpos[t, :] * traj.mask_rel
