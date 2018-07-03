""" This file defines an agent for the MuJoCo simulator environment. """
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from python_visual_mpc.visual_mpc_core.agent.utils.gen_gtruth_desig import gen_gtruthdesig
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.agent.utils.convert_world_imspace_mj1_5 import project_point
import pickle
from PIL import Image
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
from pyquaternion import Quaternion
from mujoco_py import load_model_from_path, MjSim
from python_visual_mpc.visual_mpc_core.agent.utils.get_masks import get_obj_masks
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
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
        self.num_objects = hyperparams['env'][1]['num_objects']
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
        self._hyperparams['i_tr'] = i_tr

        traj_ok = False
        self.i_trial = 0
        imax = 100
        while not traj_ok and self.i_trial < imax:
            self.i_trial += 1
            try:
                traj_ok, traj = self.rollout(policy, i_tr)
            except Image_dark_except:
                traj_ok = False

        print('needed {} trials'.format(self.i_trial))

        if self.goal_obj_pose is not None:
            final_poscost, final_anglecost = self.eval_action(traj, traj.term_t)
            final_poscost = np.mean(final_poscost)
            initial_poscost, _ = self.eval_action(traj, 0)
            initial_poscost = np.mean(initial_poscost)
            traj.stats['scores'] = final_poscost
            traj.stats['initial_poscost'] = initial_poscost
            traj.stats['improvement'] = initial_poscost - final_poscost
            traj.stats['integrated_poscost'] = np.mean(traj.goal_dist)
            traj.stats['term_t'] = traj.term_t

        if 'save_goal_image' in self._hyperparams:
            self.save_goal_image_conf(traj)

        if 'make_final_gif' in self._hyperparams:
            self.save_gif(i_tr)

        if 'verbose' in self._hyperparams:
            self.plot_ctrls(i_tr)
            # self.plot_pix_dist(plan_stat)
        return traj

    def get_desig_pix(self, round=True):
        qpos_dim = self.sdim // 2  # the states contains pos and vel
        assert self.sim.data.qpos.shape[0] == qpos_dim + 7 * self.num_objects
        desig_pix = np.zeros([self.ncam, self.num_objects, 2], dtype=np.int)
        ratio = self._hyperparams['viewer_image_width'] / self._hyperparams['image_width']
        for icam in range(self.ncam):
            for i in range(self.num_objects):
                fullpose = self.sim.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze()
                d = project_point(fullpose[:3], icam)
                d = np.stack(d) / ratio
                if round:
                    d = np.around(d).astype(np.int)
                desig_pix[icam, i] = d
        return desig_pix

    def hide_arm_store_image(self, ind, traj):
        qpos = copy.deepcopy(self.sim.data.qpos)
        qpos[2] -= 10
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()
        width = self._hyperparams['image_width']
        height = self._hyperparams['image_height']
        traj.first_last_noarm[ind] = self.sim.render(width, height, camera_name='maincam')[::-1, :, :]
        qpos[2] += 10
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()

    def get_goal_pix(self, round=True):
        goal_pix = np.zeros([self.ncam, self.num_objects, 2], dtype=np.int)
        ratio = self._hyperparams['viewer_image_width'] / self._hyperparams['image_width']
        for icam in range(self.ncam):
            for i in range(self.num_objects):
                g = project_point(self.goal_obj_pose[i, :3], icam)
                g = np.stack(g) / ratio
                if round:
                    g= np.around(g).astype(np.int)
                goal_pix[icam, i] = g
        return goal_pix


    def get_int_targetpos(self, substep, prev, next):
        assert substep >= 0 and substep < self._hyperparams['substeps']
        return substep/float(self._hyperparams['substeps'])*(next - prev) + prev

    def rollout(self, policy, i_tr):
        self._init()
        if self.goal_obj_pose is not None:
            self.goal_pix = self.get_goal_pix()

        traj = Trajectory(self._hyperparams)
        traj.i_tr = i_tr

        if 'first_last_noarm' in self._hyperparams:
            self.hide_arm_store_image(0, traj)

        # Take the sample.
        t = 0
        done = False
        obs = self.env.reset()
        self.large_images_traj = []
        agent_img_height, agent_img_width = self._hyperparams['image_height'], self._hyperparams['image_width']
        while not done:
            traj.X_full[t] = obs['qpos']
            traj.Xdot_full[t] = obs['qvel']
            traj.X_Xdot_full[t] = np.concatenate([traj.X_full[t, :], traj.Xdot_full[t, :]])
            traj.target_qpos[t] = obs['target_qpos']
            traj.Object_full_pose[t] = obs['object_poses_full']
            traj.Object_pose[t] = obs['object_poses']

            if 'finger_sensors' in self._hyperparams['env'][1]:
                traj.touch_sensors[t] = obs['finger_sensors']


            for i, cam in enumerate(self._hyperparams['cameras']):
                traj.images[t, i] = cv2.resize(obs['images'][i], (agent_img_width, agent_img_height),
                                                                            interpolation = cv2.INTER_AREA)

            self.large_images_traj.append(obs['images'][0])
            # if 'get_curr_mask' in self._hyperparams:
            #     self.curr_mask, self.curr_mask_large = get_obj_masks(self.sim, self._hyperparams, include_arm=False) #get target object mask
            # else:
            #     self.desig_pix = self.get_desig_pix()

            # if 'gtruthdesig' in self._hyperparams:  # generate many designated pixel goal-pixel pairs
            #     self.desig_pix, self.goal_pix = gen_gtruthdesig(fullpose, self.goal_obj_pose,
            #                                                     self.curr_mask_large, traj.largedimage[t], self._hyperparams['gtruthdesig'],
            #                                                     self._hyperparams, traj.images[t], self.goal_image)
            #

            policy_args = {}
            policy_signature = signature(policy.act)         #Gets arguments required by policy
            for arg in policy_signature.parameters:          #Fills out arguments according to their keyword
                value = policy_signature.parameters[arg].default
                if arg == 'traj':
                    value = traj
                elif arg == 't':
                    value = t

                if value is Parameter.empty:
                    #required parameters MUST be set by agent
                    raise ValueError("Required Policy Param {} not set in agent".format(arg))
                policy_args[arg] = value

            mj_U = policy.act(**policy_args)

            traj.actions[t, :] = mj_U.copy()
            obs = self.env.step(mj_U)

            if self.goal_obj_pose is not None:
                traj.goal_dist.append(self.eval_action(traj, t)[0])

            if 'term_dist' in self._hyperparams:
                if traj.goal_dist[-1] < self._hyperparams['term_dist']:
                    done = True
            if (self._hyperparams['T']-1) == t:
                done = True
            if done:
                traj.term_t = t
            t += 1


        if 'first_last_noarm' in self._hyperparams:
            self.hide_arm_store_image(1, traj)

        if np.any(traj.Object_full_pose[:,:,2] > 0.01):
            lifted = True
        else: lifted = False
        traj.stats['lifted'] = lifted
        # only save trajectories which displace objects above threshold
        if 'displacement_threshold' in self._hyperparams:
            assert self._hyperparams['data_collection']
            disp_per_object = np.zeros(self.num_objects)
            for i in range(self.num_objects):
                pos_old = traj.Object_pose[0, i, :2]
                pos_new = traj.Object_pose[t, i, :2]
                disp_per_object[i] = np.linalg.norm(pos_old - pos_new)

            if np.sum(disp_per_object) > self._hyperparams['displacement_threshold']:
                traj_ok = True
            else:
                traj_ok = False
        elif 'lift_rejection_sample' in self._hyperparams:
            valid_frames = np.logical_and(traj.target_qpos[1:,-1] > 0.05, np.logical_and(traj.touch_sensors[:, 0] > 0, traj.touch_sensors[:, 1] > 0))
            off_ground = traj.target_qpos[1:,2] >= 0
            if not any(np.logical_and(valid_frames, off_ground)) and self.i_trial < self._hyperparams['lift_rejection_sample']:
                traj_ok = False
            else:
                traj_ok = True
        else:
            traj_ok = True

        #discarding trajecotries where an object falls out of the bin:
        end_zpos = [traj.Object_full_pose[-1, i, 2] for i in range(self.num_objects)]
        if any(zval < -2e-2 for zval in end_zpos):
            print('object fell out!!!')
            traj_ok = False

        if 'dist_ok_thresh' in self._hyperparams:
            if np.any(traj.goal_dist[-1] > self._hyperparams['dist_ok_thresh']):
                traj_ok = False
        return traj_ok, traj

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


    def zangle_to_quat(self, zangle):
        """
        :param zangle in rad
        :return: quaternion
        """
        return np.array([np.cos(zangle/2), 0, 0, np.sin(zangle/2) ])


    def calc_anglediff(self, alpha, beta):
        delta = alpha - beta
        while delta > np.pi:
            delta -= 2*np.pi
        while delta < -np.pi:
            delta += 2*np.pi
        return delta

    def get_world_coord(self, proj_mat, depth_image, pix_pos):
        depth = depth_image[pix_pos[0], pix_pos[1]]
        pix_pos = pix_pos.astype(np.float32) / depth_image.shape[0]
        clipspace = pix_pos*2 -1
        depth = depth*2 -1

        clipspace = np.concatenate([clipspace, depth, np.array([1.]) ])

        res = np.linalg.inv(proj_mat).dot(clipspace)
        res[:3] = 1 - res[:3]

        return res[:3]

    def get_point_cloud(self, depth_image, proj_mat):

        height = depth_image.shape[0]
        point_cloud = np.zeros([height, height,3])
        for r in range(point_cloud.shape[0]):
            for c in range(point_cloud.shape[1]):
                pix_pos = np.array([r, c])
                point_cloud[r, c] = self.get_world_coord(proj_mat,depth_image, pix_pos)[:3]

        return point_cloud

    def plot_point_cloud(self, point_cloud):

        height = point_cloud.shape[0]

        point_cloud = point_cloud.reshape([height**2, 3])
        px = point_cloud[:, 0]
        py = point_cloud[:, 1]
        pz = point_cloud[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.scatter(ax, px, py, pz)
        plt.show()



    def save_gif(self, itr):
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
        #create random starting poses for objects
        #Need to figure what this did.....
        # if self.start_conf is None and 'not_create_goals' not in self._hyperparams:
        #     self.goal_obj_pose = []
        #     dist_betwob_ok = False
        #     while not dist_betwob_ok:
        #         for i_ob in range(self._hyperparams['num_objects']):
        #             pos_ok = False
        #             while not pos_ok:
        #                 if 'ang_disp_range' in self._hyperparams:
        #                     angular_disp = self._hyperparams['ang_disp_range']
        #                 else: angular_disp = 0.2
        #                 delta_alpha = np.random.uniform(-angular_disp, angular_disp)
        #                 delta_rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_alpha)
        #                 pose = object_pos_l[i_ob]
        #                 curr_quat = Quaternion(pose[3:])
        #                 newquat = delta_rot*curr_quat
        #
        #                 alpha = np.random.uniform(-np.pi, np.pi, 1)
        #                 if 'const_dist' in self._hyperparams:
        #                     assert 'pos_disp_range' not in self._hyperparams
        #                     d = self._hyperparams['const_dist']
        #                     delta_pos = np.array([d*np.cos(alpha), d*np.sin(alpha), 0.])
        #                 else:
        #                     pos_disp = self._hyperparams['pos_disp_range']
        #                     delta_pos = np.concatenate([np.random.uniform(-pos_disp, pos_disp, 2), np.zeros([1])])
        #                 newpos = pose[:3] + delta_pos
        #
        #                 if 'lift_object' in self._hyperparams:
        #                     newpos[2] = 0.15
        #                 if np.any(newpos[:2] > 0.35) or np.any(newpos[:2] < -0.35):   # check if in field
        #                     continue
        #                 else:
        #                     self.goal_obj_pose.append(np.concatenate([newpos, newquat.elements]))
        #                     pos_ok = True
        #
        #         if self._hyperparams['num_objects'] == 2:
        #             #ensuring that the goal positions are far apart from each other
        #             if np.linalg.norm(self.goal_obj_pose[0][:3]- self.goal_obj_pose[1][:3]) < 0.2:
        #                 self.goal_obj_pose = []
        #                 continue
        #             dist_betwob_ok = True
        #         else:
        #             dist_betwob_ok = True
        #     self.goal_obj_pose = np.stack(self.goal_obj_pose, axis=0)

