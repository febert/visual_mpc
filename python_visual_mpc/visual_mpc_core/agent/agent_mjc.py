""" This file defines an agent for the MuJoCo simulator environment. """
from copy import deepcopy
from python_visual_mpc.visual_mpc_core.agent.utils.gen_gtruth_desig import gen_gtruthdesig
import copy
import numpy as np
import pdb
from python_visual_mpc.visual_mpc_core.agent.utils.convert_world_imspace_mj1_5 import project_point, get_3D
import pickle
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.misc.makegifs2 import assemble_gif, npy_to_gif
from pyquaternion import Quaternion
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
from python_visual_mpc.visual_mpc_core.agent.utils.get_masks import get_obj_masks, get_image
import time
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
from mpl_toolkits.mplot3d import Axes3D
from python_visual_mpc.visual_mpc_core.agent.utils.create_xml import create_object_xml, create_root_xml
import os
import cv2


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
        self._hyperparams = hyperparams
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
        self._setup_world()

    def _setup_world(self):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        if "gen_xml" in self._hyperparams:
            self.obj_statprop = create_object_xml(self._hyperparams, self.load_obj_statprop)
            xmlfilename = create_root_xml(self._hyperparams)
            self._hyperparams['gen_xml_fname'] = xmlfilename
        else:
            xmlfilename = self._hyperparams['filename']
        self.sim = MjSim(load_model_from_path(xmlfilename))

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

    def sample(self, policy, i_tr, verbose=True, save=True, noisy=False):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        if self.start_conf is not None:
            self.apply_start_conf(self.start_conf)

        if "gen_xml" in self._hyperparams:
            if i_tr % self._hyperparams['gen_xml'] == 0:
                self._setup_world()
        self._hyperparams['i_tr'] = i_tr

        traj_ok = False
        i_trial = 0
        imax = 100
        while not traj_ok and i_trial < imax:
            i_trial += 1
            try:
                traj_ok, traj = self.rollout(policy, i_tr)
            except Image_dark_except:
                traj_ok = False

        print('needed {} trials'.format(i_trial))

        tfinal = self._hyperparams['T'] -1
        if self.goal_obj_pose is not None:
            self.final_poscost, self.final_anglecost = self.eval_action(traj, tfinal)
            self.final_poscost = np.mean(self.final_poscost)
            self.initial_poscost, _ = self.eval_action(traj, 0)
            self.initial_poscost = np.mean(self.initial_poscost)
            self.improvement = self.initial_poscost - self.final_poscost
            traj.improvement = self.improvement
            traj.final_poscost = self.final_poscost
            traj.initial_poscost = self.initial_poscost

        if 'save_goal_image' in self._hyperparams:
            self.save_goal_image_conf(traj)

        if 'make_final_gif' in self._hyperparams:
            self.save_gif()
        return traj

    def get_desig_pix(self, round=True):
        qpos_dim = self.sdim // 2  # the states contains pos and vel
        assert self.sim.data.qpos.shape[0] == qpos_dim + 7 * self._hyperparams['num_objects']
        desig_pix = np.zeros([self.ncam, self._hyperparams['num_objects'], 2], dtype=np.int)
        ratio = self._hyperparams['viewer_image_width'] / self._hyperparams['image_width']
        for icam in range(self.ncam):
            for i in range(self._hyperparams['num_objects']):
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
        goal_pix = np.zeros([self.ncam, self._hyperparams['num_objects'], 2], dtype=np.int)
        ratio = self._hyperparams['viewer_image_width'] / self._hyperparams['image_width']
        for icam in range(self.ncam):
            for i in range(self._hyperparams['num_objects']):
                g = project_point(self.goal_obj_pose[i, :3], icam)
                g = np.stack(g) / ratio
                if round:
                    g= np.around(g).astype(np.int)
                goal_pix[icam, i] = g
        return goal_pix

    def clip_targetpos(self, pos):
        pos_clip = self._hyperparams['targetpos_clip']
        return np.clip(pos, pos_clip[0], pos_clip[1])

    def get_int_targetpos(self, substep, prev, next):
        assert substep >= 0 and substep < self._hyperparams['substeps']
        return substep/float(self._hyperparams['substeps'])*(next - prev) + prev

    def rollout(self, policy, i_tr):
        self._init()
        if self.goal_obj_pose is not None:
            self.goal_pix = self.get_goal_pix()   # has to occurr after self.viewer.cam.camid = 0 and set_model!!!

        traj = Trajectory(self._hyperparams)
        traj.i_tr = i_tr

        if 'gen_xml' in self._hyperparams:
            traj.obj_statprop = self.obj_statprop
        if 'compare_mj_planner_actions' in self._hyperparams:
            traj.mj_planner_actions = self.mj_planner_actions

        self.large_images_traj = []
        self.large_images = []

        self.hf_target_qpos_l = []
        self.hf_qpos_l = []

        # apply action of zero for the first few steps, to let the scene settle
        if 'skip_frist' not in self._hyperparams:
            skip_first = 10
        else: skip_first = self._hyperparams['skip_first']

        target_skip_first = copy.deepcopy(self.sim.data.qpos[:self.adim].squeeze())
        for t in range(skip_first):
            for _ in range(self._hyperparams['substeps']):
                if 'posmode' in self._hyperparams:
                    # keep gripper at default x,y positions
                    ctrl = target_skip_first
                else:
                    ctrl = np.zeros(self._hyperparams['adim'])
                self.sim.data.ctrl[:] = ctrl
                self.sim.step()

        self.gripper_closed = False
        self.gripper_up = False

        if 'first_last_noarm' in self._hyperparams:
            self.hide_arm_store_image(0, traj)

        # Take the sample.
        for t in range(self.T):
            qpos_dim = self.sdim // 2  # the states contains pos and vel
            traj.X_full[t, :] = self.sim.data.qpos[:qpos_dim].squeeze().copy()
            traj.Xdot_full[t, :] = self.sim.data.qvel[:qpos_dim].squeeze().copy()
            traj.X_Xdot_full[t, :] = np.concatenate([traj.X_full[t, :], traj.Xdot_full[t, :]])
            assert self.sim.data.qpos.shape[0] == qpos_dim + 7 * self._hyperparams['num_objects']

            for i in range(self._hyperparams['num_objects']):
                fullpose = self.sim.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze().copy()

                if 'object_meshes' in self._hyperparams:
                    fullpose[:3] = self.sim.data.sensordata[i * 3 :(i + 1) * 3].copy()

                traj.Object_full_pose[t, i, :] = fullpose
                zangle = self.quat_to_zangle(fullpose[3:])
                traj.Object_pose[t, i, :] = np.concatenate([fullpose[:2], zangle])  # save only xyz, theta

            if 'get_curr_mask' in self._hyperparams:
                self.curr_mask, self.curr_mask_large = get_obj_masks(self.sim, self._hyperparams, include_arm=False) #get target object mask
            else:
                self.desig_pix = self.get_desig_pix()

            self._store_image(t , traj, policy)
            if 'gtruthdesig' in self._hyperparams:  # generate many designated pixel goal-pixel pairs
                self.desig_pix, self.goal_pix = gen_gtruthdesig(fullpose, self.goal_obj_pose,
                                                                self.curr_mask_large, traj.largedimage[t], self._hyperparams['gtruthdesig'],
                                                                self._hyperparams, traj.images[t], self.goal_image)

            if 'not_use_images' in self._hyperparams:
                mj_U = policy.act(traj, t, self.sim, self.goal_obj_pose, self._hyperparams, self.goal_image)
            else:
                mj_U, plan_stat = policy.act(traj, t, desig_pix=self.desig_pix,goal_pix=self.goal_pix,
                                          goal_image=self.goal_image, goal_mask=self.goal_mask, curr_mask=self.curr_mask)
                traj.plan_stat.append(copy.deepcopy(plan_stat))

            self.large_images_traj.append(self.large_images[t])

            traj.actions[t, :] = mj_U
            if 'posmode' in self._hyperparams:  #if the output of act is a positions
                if t == 0:
                    self.prev_target_qpos = copy.deepcopy(self.sim.data.qpos[:self.adim].squeeze())
                    self.target_qpos = copy.deepcopy(self.sim.data.qpos[:self.adim].squeeze())
                else:
                    self.prev_target_qpos = copy.deepcopy(self.target_qpos)

                if 'discrete_adim' in self._hyperparams:
                    up_cmd = mj_U[2]
                    assert np.floor(up_cmd) == up_cmd
                    if up_cmd != 0:
                        self.t_down = t + up_cmd
                        self.target_qpos[2] = self._hyperparams['targetpos_clip'][1][2]
                        self.gripper_up = True
                    if self.gripper_up:
                        if t == self.t_down:
                            self.target_qpos[2] = self._hyperparams['targetpos_clip'][0][2]
                            self.gripper_up = False
                    self.target_qpos[:2] += mj_U[:2]
                    if self.adim == 4:
                        self.target_qpos[3] += mj_U[3]
                    assert self.adim <= 4
                elif 'close_once_actions' in self._hyperparams:
                    assert self.adim == 5
                    self.target_qpos[:4] = mj_U[:4] + self.target_qpos[:4]
                    grasp_thresh = 0.5
                    if mj_U[4] > grasp_thresh:
                        self.gripper_closed = True
                    if self.gripper_closed:
                        self.target_qpos[4] = 0.1
                    else:
                       self.target_qpos[4] = 0.0
                    print('target_qpos', self.target_qpos)
                else:
                    self.target_qpos = self.target_qpos + mj_U*self._hyperparams['mode_rel']
                self.target_qpos = self.clip_targetpos(self.target_qpos)
                traj.target_qpos[t] = self.target_qpos
            else:
                ctrl = mj_U.copy()

            for st in range(self._hyperparams['substeps']):
                if 'posmode' in self._hyperparams:
                    ctrl = self.get_int_targetpos(st, self.prev_target_qpos, self.target_qpos)
                self.sim.data.ctrl[:] = ctrl
                self.sim.step()
                # width = self._hyperparams['viewer_image_width']
                # height = self._hyperparams['viewer_image_height']
                # cv2.imwrite('test_rec/test{}.jpg'.format(t * self._hyperparams['substeps'] + st), self.sim.render(width, height, camera_name="maincam")[::-1, :, ::-1])

                self.hf_qpos_l.append(copy.deepcopy(self.sim.data.qpos))
                self.hf_target_qpos_l.append(copy.deepcopy(ctrl))

            if self.goal_obj_pose is not None:
                traj.goal_dist.append(self.eval_action(traj, t)[0])

        if 'first_last_noarm' in self._hyperparams:
            self.hide_arm_store_image(1, traj)

        # only save trajectories which displace objects above threshold
        if 'displacement_threshold' in self._hyperparams:
            assert self._hyperparams['data_collection']
            disp_per_object = np.zeros(self._hyperparams['num_objects'])
            for i in range(self._hyperparams['num_objects']):
                pos_old = traj.Object_pose[0, i, :2]
                pos_new = traj.Object_pose[t, i, :2]
                disp_per_object[i] = np.linalg.norm(pos_old - pos_new)

            if np.sum(disp_per_object) > self._hyperparams['displacement_threshold']:
                traj_ok = True
            else:
                traj_ok = False
        else:
            traj_ok = True


        #discarding trajecotries where an object falls out of the bin:
        end_zpos = [traj.Object_full_pose[-1, i, 2] for i in range(self._hyperparams['num_objects'])]
        if any(zval < -2e-2 for zval in end_zpos):
            print('object fell out!!!')
            traj_ok = False
        if 'verbose' in self._hyperparams:
            self.plot_ctrls()

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
        for i_ob in range(self._hyperparams['num_objects']):
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

    def quat_to_zangle(self, quat):
        """
        :param quat: quaternion with only
        :return: zangle in rad
        """
        theta = np.arctan2(2*quat[0]*quat[3], 1-2*quat[3]**2)
        return np.array([theta])

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

    def _store_image(self, t, traj, policy=None):
        """
        store image at time index t
        """
        assert self._hyperparams['viewer_image_width']/self._hyperparams['image_width'] == self._hyperparams['viewer_image_height']/self._hyperparams['image_height']
        width = self._hyperparams['viewer_image_width']
        height = self._hyperparams['viewer_image_height']

        if 'cameras' in self._hyperparams:
            for i, cam in enumerate(self._hyperparams['cameras']):
                large_img = self.sim.render(width, height, camera_name=cam)[::-1, :, :]

                if np.sum(large_img) < 1e-3:
                    print("image dark!!!")
                    raise Image_dark_except
                if cam == 'maincam':
                    self.large_images.append(large_img)
                traj.images[t, i] = cv2.resize(large_img, dsize=(self._hyperparams['image_width'],
                                        self._hyperparams['image_height']), interpolation = cv2.INTER_AREA)

                if 'make_gtruth_flows' in self._hyperparams:
                    traj.largeimage[t, i] = large_img
                    dlarge_img = self.sim.render(width, height, camera_name="maincam", depth=True)[1][::-1, :]
                    traj.largedimage[t, i] = dlarge_img
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            large_img = self.sim.render(width, height, camera_name="maincam")[::-1, :, :]
            if np.sum(large_img) < 1e-3:
                print("image dark!!!")
                raise Image_dark_except
            self.large_images.append(large_img)
            traj.images[t] = cv2.resize(large_img, dsize=(self._hyperparams['image_width'], self._hyperparams['image_height']), interpolation = cv2.INTER_AREA)
            if 'image_medium' in self._hyperparams:
                traj._image_medium[t] = cv2.resize(large_img, dsize=(self._hyperparams['image_medium'][1],
                                                                         self._hyperparams['image_medium'][0]), interpolation = cv2.INTER_AREA)
            if 'make_gtruth_flows' in self._hyperparams:
                traj.largeimage[t] = large_img
                dlarge_img = self.sim.render(width, height, camera_name="maincam", depth=True)[1][::-1, :]
                traj.largedimage[t] = dlarge_img

        if 'store_whole_pred' in self._hyperparams:
            if t > 1:
                traj.predicted_images = policy.best_gen_images
                traj.gtruth_images = policy.best_gtruth_images

    def save_gif(self):
        file_path = self._hyperparams['record']
        npy_to_gif(self.large_images_traj, file_path +'/video')

    def plot_ctrls(self):
        plt.figure()
        # a = plt.gca()
        self.hf_qpos_l = np.stack(self.hf_qpos_l, axis=0)
        self.hf_target_qpos_l = np.stack(self.hf_target_qpos_l, axis=0)
        tmax = self.hf_target_qpos_l.shape[0]

        i = 4
        plt.plot(list(range(tmax)), self.hf_qpos_l[:,i], label='q_{}'.format(i))
        plt.plot(list(range(tmax)), self.hf_target_qpos_l[:, i], label='q_target{}'.format(i))
        plt.legend()
        # plt.show()
        if not os.path.exists(self._hyperparams['record']):
            os.makedirs(self._hyperparams['record'])
        # for i in range(self.adim):
        #     plt.plot(list(range(tmax)), self.hf_qpos_l[:,i], label='q_{}'.format(i))
        #     plt.plot(list(range(tmax)), self.hf_target_qpos_l[:, i], label='q_target{}'.format(i))
        #     plt.legend()
        #     # plt.show()
        #     if not os.path.exists(self._hyperparams['record']):
        #         os.makedirs(self._hyperparams['record'])
        plt.savefig(self._hyperparams['record'] + '/ctrls.png')

    def _init(self):
        """
        Set the world to a given model
        """
        #create random starting poses for objects
        def create_pos():
            poses = []
            for i in range(self._hyperparams['num_objects']):
                pos = np.random.uniform(-.35, .35, 2)
                alpha = np.random.uniform(0, np.pi*2)
                ori = np.array([np.cos(alpha/2), 0, 0, np.sin(alpha/2) ])
                poses.append(np.concatenate((pos, np.array([0]), ori), axis= 0))
            return poses

        if 'sample_objectpos' in self._hyperparams: # if object pose explicit do not sample poses
            assert self.start_conf is None
            if 'object_object_mindist' in self._hyperparams:
                assert self._hyperparams['num_objects'] == 2
                ob_ob_dist = 0.
                while ob_ob_dist < self._hyperparams['object_object_mindist']:
                    object_pos_l = create_pos()
                    ob_ob_dist = np.linalg.norm(object_pos_l[0][:3] - object_pos_l[1][:3])
                object_pos = np.concatenate(object_pos_l)
            else:
                object_pos_l = create_pos()
                object_pos = np.concatenate(object_pos_l)
        else:
            object_pos = self._hyperparams['object_pos0'][:self._hyperparams['num_objects']]

        xpos0 = np.zeros(self._hyperparams['sdim']//2)
        if 'randomize_ballinitpos' in self._hyperparams:
            assert self.start_conf is None
            xpos0[:2] = np.random.uniform(-.4, .4, 2)
            xpos0[2] = np.random.uniform(-0.08, .14)
        elif 'arm_obj_initdist' in self._hyperparams:
            d = self._hyperparams['arm_obj_initdist']
            alpha = np.random.uniform(-np.pi, np.pi, 1)
            delta_pos = np.array([d*np.cos(alpha), d*np.sin(alpha)])
            xpos0[:2] = object_pos[:2] + delta_pos.squeeze()
            xpos0[2] = np.random.uniform(-0.08, .14)
        else:
            xpos0_true_len = (self.sim.get_state().qpos.shape[0] - self._hyperparams['num_objects']*7)
            len_xpos0 = self._hyperparams['xpos0'].shape[0]

            if len_xpos0 != xpos0_true_len:
                xpos0 = np.concatenate([self._hyperparams['xpos0'], np.zeros(xpos0_true_len - len_xpos0)], 0)  #testing in setting with updown rot, while data has only xyz
                print("appending zeros to initial robot configuration!!!")
            else:
                xpos0 = self._hyperparams['xpos0']
            assert xpos0.shape[0] == self._hyperparams['sdim']/2

        if 'arm_start_lifted' in self._hyperparams:
            xpos0[2] = self._hyperparams['arm_start_lifted']


        sim_state = self.sim.get_state()
        if 'goal_point' in self._hyperparams:
            goal = np.append(self._hyperparams['goal_point'], [.1])   # goal point
            ref = np.append(object_pos[:2], [.1]) # reference point on the block
            sim_state.qpos[:] = np.concatenate((xpos0, object_pos, goal, ref), 0)
        else:
            sim_state.qpos[:] = np.concatenate((xpos0, object_pos.flatten()), 0)

        sim_state.qvel[:] = np.zeros_like(sim_state.qvel)
        self.sim.set_state(sim_state)
        self.sim.forward()

        if self.start_conf is None and 'not_create_goals' not in self._hyperparams:
            self.goal_obj_pose = []
            dist_betwob_ok = False
            while not dist_betwob_ok:
                for i_ob in range(self._hyperparams['num_objects']):
                    pos_ok = False
                    while not pos_ok:
                        if 'ang_disp_range' in self._hyperparams:
                            angular_disp = self._hyperparams['ang_disp_range']
                        else: angular_disp = 0.2
                        delta_alpha = np.random.uniform(-angular_disp, angular_disp)
                        delta_rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_alpha)
                        pose = object_pos_l[i_ob]
                        curr_quat = Quaternion(pose[3:])
                        newquat = delta_rot*curr_quat

                        alpha = np.random.uniform(-np.pi, np.pi, 1)
                        if 'const_dist' in self._hyperparams:
                            assert 'pos_disp_range' not in self._hyperparams
                            d = self._hyperparams['const_dist']
                            delta_pos = np.array([d*np.cos(alpha), d*np.sin(alpha), 0.])
                        else:
                            pos_disp = self._hyperparams['pos_disp_range']
                            delta_pos = np.concatenate([np.random.uniform(-pos_disp, pos_disp, 2), np.zeros([1])])
                        newpos = pose[:3] + delta_pos

                        if np.any(newpos[:2] > 0.35) or np.any(newpos[:2] < -0.35):   # check if in field
                            continue
                        else:
                            self.goal_obj_pose.append(np.concatenate([newpos, newquat.elements]))
                            pos_ok = True

                if self._hyperparams['num_objects'] == 2:
                    #ensuring that the goal positions are far apart from each other
                    if np.linalg.norm(self.goal_obj_pose[0][:3]- self.goal_obj_pose[1][:3]) < 0.2:
                        self.goal_obj_pose = []
                        continue
                    dist_betwob_ok = True
                else:
                    dist_betwob_ok = True
            self.goal_obj_pose = np.stack(self.goal_obj_pose, axis=0)

