""" This file defines an agent for the MuJoCo simulator environment. """
from copy import deepcopy
import copy
import numpy as np
import mujoco_py
from mujoco_py.mjtypes import *
import pdb
import cPickle
from PIL import Image
import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.misc.makegifs2 import assemble_gif, npy_to_gif
from pyquaternion import Quaternion

import time
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
from mpl_toolkits.mplot3d import Axes3D

from utils.create_xml import create_object_xml, create_root_xml
import os
from time import sleep
import cv2


def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

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
            xmlfilename_nomarkers = xmlfilename
        else:
            xmlfilename = self._hyperparams['filename']
            xmlfilename_nomarkers = self._hyperparams['filename_nomarkers']

        self._model= mujoco_py.MjModel(xmlfilename)
        self.model_nomarkers = mujoco_py.MjModel(xmlfilename_nomarkers)

        gofast = True
        self.viewer = mujoco_py.MjViewer(visible=True,
                                         init_width=self._hyperparams['viewer_image_width'],
                                         init_height=self._hyperparams['viewer_image_height'],
                                         go_fast=gofast)
        self.viewer.start()
        self.viewer.cam.camid = 0

    def sample(self, policy, i_tr, verbose=True, save=True, noisy=False):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        if "gen_xml" in self._hyperparams:
            if i_tr % self._hyperparams['gen_xml'] == 0:
                self.viewer.finish()
                self._setup_world()

        traj_ok = False
        i_trial = 0
        imax = 100
        while not traj_ok and i_trial < imax:
            i_trial += 1
            traj_ok, traj = self.rollout(policy)

        print 'needed {} trials'.format(i_trial)

        tfinal = self._hyperparams['T'] -1
        if not 'data_collection' in self._hyperparams:
            self.final_poscost, self.final_anglecost = self.eval_action(traj, tfinal, getanglecost=True)

        if 'save_goal_image' in self._hyperparams:
            self.save_goal_image_conf(traj)

        if 'make_final_gif' in self._hyperparams:
            self.save_gif()

        return traj

    def get_max_move_pose(self, traj):
        """
        get pose trajectory of the object with maximum accumulated motion
        :param traj:
        :return:
        """
        delta_move = np.zeros(self._hyperparams['num_objects'])
        for i in range(self._hyperparams['num_objects']):
            for t in range(self.T-1):
                delta_move[i] += np.linalg.norm(traj.Object_pose[t+1,i,:2] -traj.Object_pose[t,i,:2])

        imax = np.argmax(delta_move)
        traj.max_move_pose = traj.Object_pose[:,imax]

        return traj

    def get_desig_pix(self, round=True):
        qpos_dim = self.sdim / 2  # the states contains pos and vel
        assert self._model.data.qpos.shape[0] == qpos_dim + 7 * self._hyperparams['num_objects']
        desigpix = []
        for i in range(self._hyperparams['num_objects']):
            fullpose = self._model.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze()
            desigpix.append(self.viewer.project_point(fullpose[:3]))
        ratio = self._hyperparams['viewer_image_width']/self._hyperparams['image_width']
        desig_pix = np.stack(desigpix) / ratio
        if round:
            desig_pix = np.around(desig_pix).astype(np.int)
        return desig_pix

    def get_goal_pix(self, round=True):
        goal_pix = []
        for i in range(self._hyperparams['num_objects']):
            goal_pix.append(self.viewer.project_point(self.goal_obj_pose[i, :3]))

        ratio = self._hyperparams['viewer_image_width'] / self._hyperparams['image_width']

        goal_pix = np.stack(goal_pix) / ratio
        if round:
            goal_pix = np.around(goal_pix).astype(np.int)

        # img = self.goal_image
        # for i in range(self._hyperparams['num_objects']):
        #     img[goal_pix[i, 0], goal_pix[i, 1]] = np.array([1., 1., 1.])
        # plt.imshow(img)
        # plt.show()
        # print 'goal_pix', goal_pix
        # print 'goal obj pose', self.goal_obj_pose
        return goal_pix

    def rollout(self, policy):
        self.viewer.set_model(self.model_nomarkers)
        self.viewer.cam.camid = 0

        self._init()
        if 'data_collection' not in self._hyperparams:
            self.goal_pix = self.get_goal_pix()   # has to occurr after self.viewer.cam.camid = 0 and set_model!!!

        traj = Trajectory(self._hyperparams)
        if 'gen_xml' in self._hyperparams:
            traj.obj_statprop = self.obj_statprop

        # apply action of zero for the first few steps, to let the scene settle
        for t in range(self._hyperparams['skip_first']):
            for _ in range(self._hyperparams['substeps']):
                self._model.data.ctrl = np.zeros(self._hyperparams['adim'])
                self._model.step()

        self.large_images_traj = []
        self.large_images = []

        # Take the sample.
        for t in range(self.T):
            qpos_dim = self.sdim / 2  # the states contains pos and vel
            traj.X_full[t, :] = self._model.data.qpos[:qpos_dim].squeeze()
            traj.Xdot_full[t, :] = self._model.data.qvel[:qpos_dim].squeeze()
            traj.X_Xdot_full[t, :] = np.concatenate([traj.X_full[t, :], traj.Xdot_full[t, :]])
            assert self._model.data.qpos.shape[0] == qpos_dim + 7 * self._hyperparams['num_objects']
            for i in range(self._hyperparams['num_objects']):
                fullpose = self._model.data.qpos[i * 7 + qpos_dim:(i+1) * 7 + qpos_dim].squeeze()
                traj.Object_full_pose[t, i, :] = fullpose
                zangle = self.quat_to_zangle(fullpose[3:])
                traj.Object_pose[t, i, :] = np.concatenate([fullpose[:2], zangle])  # save only xyz, theta

            self.desig_pix = self.get_desig_pix()
            self._store_image(t, traj, policy)

            if 'data_collection' in self._hyperparams or 'random_baseline' in self._hyperparams:
                mj_U, target_inc = policy.act(traj, t)
            elif 'gtruth_planner' in self._hyperparams:
                mj_U, pos, ind, targets = policy.act(traj, t, init_model=self._model)
            else:
                mj_U, bestindices_of_iter, rec_input_distrib = policy.act(traj, t, desig_pix=self.desig_pix,goal_pix=self.goal_pix, goal_image=self.goal_image)
                if 'add_traj_visual' in self._hyperparams:  # whether to add visuals for trajectory
                    self.large_images_traj += self.add_traj_visual(self.large_images[t], pos, ind, targets)
                else:
                    self.large_images_traj.append(self.large_images[t])

            if 'poscontroller' in self._hyperparams.keys():
                traj.actions[t, :] = target_inc
            else:
                traj.actions[t, :] = mj_U

            accum_touch = np.zeros_like(self._model.data.sensordata)

            for _ in range(self._hyperparams['substeps']):
                accum_touch += self._model.data.sensordata

                if 'vellimit' in self._hyperparams:
                    # calculate constraint enforcing force..
                    c_force = self.enforce(self._model)
                    mj_U += c_force
                self._model.data.ctrl = mj_U
                self._model.step()  # simulate the model in mujoco

            if 'touch' in self._hyperparams:
                traj.touchdata[t, :] = accum_touch.squeeze()
                print 'accumulated force', t
                print accum_touch

        traj = self.get_max_move_pose(traj)

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
            print 'object fell out!!!'
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

        print 'best_score', best_score
        print 'allscores', traj.score
        print 'goal index: ', first_best_index

        goalimage = traj._sample_images[first_best_index]
        goal_ballpos = np.concatenate([traj.X_full[first_best_index], np.zeros(2)])  #set velocity to zero

        goal_object_pose = traj.Object_pos[first_best_index]

        img = Image.fromarray(goalimage)

        dict = {}
        dict['goal_image'] = goalimage
        dict['goal_ballpos'] = goal_ballpos
        dict['goal_object_pose'] = goal_object_pose

        cPickle.dump(dict, open(self._hyperparams['save_goal_image'] + '.pkl', 'wb'))
        img.save(self._hyperparams['save_goal_image'] + '.png',)

    def eval_action(self, traj, t, getanglecost=False):

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

        return np.sum(np.array(abs_distances)), np.sum(np.array(abs_angle_dist))


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

    def _store_image(self,t, traj, policy=None):
        """
        store image at time index t
        """
        #small viewer:
        self.model_nomarkers.data.qpos = self._model.data.qpos
        self.model_nomarkers.data.qvel = self._model.data.qvel
        self.model_nomarkers.step()
        self.viewer.loop_once()

        img_string, width, height = self.viewer.get_image()
        large_img = np.fromstring(img_string, dtype='uint8').reshape((height, width, 3))[::-1,:,:]
        self.large_images.append(large_img)

        assert self._hyperparams['viewer_image_width']/self._hyperparams['image_width'] == self._hyperparams['viewer_image_height']/self._hyperparams['image_height']
        traj._sample_images[t,:,:,:] = cv2.resize(large_img, dsize=(self._hyperparams['image_width'], self._hyperparams['image_height']), interpolation = cv2.INTER_AREA)

        # img_string, width, height = self.viewer.get_depth()
        # largedimage = np.fromstring(img_string, dtype='uint8').reshape(
        #     (480, 640, 1))[::-1, :, :]
        # plt.imshow(np.squeeze(largedimage))
        # plt.show()

        # img = traj._sample_images[t,:,:,:] # verify desigpos
        # desig_pix = np.around(self.desig_pix).astype(np.int)
        # # img = large_img
        # for i in range(self._hyperparams['num_objects']):
        #     img[desig_pix[i][0], desig_pix[i][1]] = np.array([255, 255, 255])
        # print 'desig_pix', desig_pix
        # plt.imshow(img)
        # plt.show()

        if 'store_video_prediction' in self._hyperparams:
            if t > 1:
                traj.final_predicted_images.append((policy.terminal_pred*255.).astype(np.uint8))

        if 'store_whole_pred' in self._hyperparams:
            if t > 1:
                traj.predicted_images = policy.best_gen_images
                traj.gtruth_images = policy.best_gtruth_images


    def add_traj_visual(self, img, traj, bestindices, targets):

        large_sample_images_traj = []
        fig = plt.figure(figsize=(6, 6), dpi=80)
        fig.add_subplot(111)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        num_samples = traj.shape[0]
        niter = traj.shape[1]

        for itr in range(niter):

            axes = plt.gca()
            plt.cla()
            axes.axis('off')
            plt.imshow(img, zorder=0)
            axes.autoscale(False)

            for smp in range(num_samples):  # for each trajectory

                x = traj[smp, itr, :, 1]
                y = traj[smp, itr, :, 0]

                if smp == bestindices[itr][0]:
                    plt.plot(x, y, zorder=1, marker='o', color='y')
                elif smp in bestindices[itr][1:]:
                    plt.plot(x, y, zorder=1, marker='o', color='r')
                else:
                    if smp % 5 == 0:
                        plt.plot(x, y, zorder=1, marker='o', color='b')

            fig.canvas.draw()  # draw the canvas, cache the renderer

            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            large_sample_images_traj.append(data)

        return large_sample_images_traj

    def save_gif(self):
        file_path = self._hyperparams['record']
        if 'random_baseline' in self._hyperparams:
            npy_to_gif(self.large_images, file_path +'/video')
        else:
            npy_to_gif(self.large_images_traj, file_path +'/video')

    def _init(self):
        """
        Set the world to a given model, and run kinematics.
        Args:
        """

        #create random starting poses for objects
        def create_pos():
            poses = []
            for i in range(self._hyperparams['num_objects']):
                pos = np.random.uniform(-.35, .35, 2)
                alpha = np.random.uniform(0, np.pi*2)

                ori = np.array([np.cos(alpha/2), 0, 0, np.sin(alpha/2) ])
                poses.append(np.concatenate((pos, np.array([0]), ori), axis= 0))
            return np.concatenate(poses)

        if 'sample_objectpos' in self._hyperparams: # if object pose explicit do not sample poses
            object_pos = create_pos()
        else:
            object_pos = self._hyperparams['object_pos0']

        # Initialize world/run kinematics
        xpos0 = self._hyperparams['xpos0']
        if 'randomize_ballinitpos' in self._hyperparams:
            xpos0[:2] = np.random.uniform(-.35, .35, 2)

        if 'goal_point' in self._hyperparams:
            goal = np.append(self._hyperparams['goal_point'], [.1])   # goal point
            ref = np.append(object_pos[:2], [.1]) # reference point on the block
            self._model.data.qpos = np.concatenate((xpos0, object_pos,goal,ref), 0)
        else:
            self._model.data.qpos = np.concatenate((xpos0, object_pos.flatten()), 0)

        self._model.data.qvel = np.zeros_like(self._model.data.qvel)