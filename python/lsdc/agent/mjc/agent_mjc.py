""" This file defines an agent for the MuJoCo simulator environment. """
from copy import deepcopy

import numpy as np

import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *

import h5py
import cPickle

from PIL import Image

import matplotlib.pyplot as plt

from lsdc.agent.agent import Agent
from lsdc.agent.agent_utils import generate_noise, setup
from lsdc.agent.config import AGENT_MUJOCO
from lsdc.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE

from lsdc.sample.sample import Sample

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])

        # datastructure for storing all images of a whole sample trajectory;
        self._sample_images = np.zeros((self.T,
                                      self._hyperparams['image_height'],
                                      self._hyperparams['image_width'],
                                      self._hyperparams['image_channels']), dtype= 'uint8')

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._model = []

        # Initialize Mujoco models. If there's only one xml file, create a single model object,
        # otherwise create a different world for each condition.

        if not isinstance(filename, list):
            for i in range(self._hyperparams['conditions']):
                self._model.append(mujoco_py.MjModel(filename))
                self.model_nomarkers = mujoco_py.MjModel(self._hyperparams['filename_nomarkers'])

        else:
            for i in range(self._hyperparams['conditions']):
                self._model.append(mujoco_py.MjModel(self._hyperparams['filename'][i]))

        for i in range(self._hyperparams['conditions']):
            for j in range(len(self._hyperparams['pos_body_idx'][i])):
                idx = self._hyperparams['pos_body_idx'][i][j]
                temp = np.copy(self._model[i].body_pos)
                temp[idx, :] = temp[idx, :] + self._hyperparams['pos_body_offset'][i][j]
                self._model[i].body_pos = temp

        # changes here:
        self._joint_idx = range(self._hyperparams['joint_angles'])
        self._vel_idx = range( self._hyperparams['joint_angles'], self._hyperparams['joint_velocities'] + self._hyperparams['joint_angles'])
        #

        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                # TODO: this assumes END_EFFECTOR_VELOCITIES is also in datapoints right?
                self._init(i)
                eepts = self._model[i].data.site_xpos.flatten()
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
                )
            else:
                self.x0.append(self._hyperparams['x0'][i])

        self._small_viewer = mujoco_py.MjViewer(visible=True,
                                                init_width=self._hyperparams['image_width'],
                                                init_height=self._hyperparams['image_height'],
                                                go_fast=True)
        self._small_viewer.start()
        self._small_viewer.cam.camid = 0

        if self._hyperparams['additional_viewer']:
            self._large_viewer = mujoco_py.MjViewer(visible=True, init_width=480,
                                                    init_height=480, go_fast=True)
            self._large_viewer.start()


    def sample(self, policy, condition, verbose=True, save=True, noisy=False):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """

        # Create new sample, populate first time step.
        self._init_sample(condition)

        U = np.empty([self.T, self.dU])
        X_full = np.empty([self.T, 2])
        Xdot_full = np.empty([self.T, 2])

        self._small_viewer.set_model(self.model_nomarkers)

        if self._hyperparams['additional_viewer']:
            self._large_viewer.set_model(self._model[condition])
            self._large_viewer.cam = deepcopy(self._small_viewer.cam)

        # apply action of zero for the first few steps, to let the scene settle
        for t in range(self._hyperparams['skip_first']):
            for _ in range(self._hyperparams['substeps']):
                self._model[condition].data.ctrl = np.array([0. ,0.])
                self._model[condition].step()

        self.large_images_traj = []
        self.large_images = []

        # Take the sample.
        for t in range(self.T):

            X_full[t, :] = self._model[condition].data.qpos[:2].squeeze()
            Xdot_full[t, :] = self._model[condition].data.qvel[:2].squeeze()

            # self.reference_points_show(condition)
            if self._hyperparams['additional_viewer']:
                self._large_viewer.loop_once()

            self._store_image(t, condition)

            if self._hyperparams['data_collection']:
                mj_U = policy.act(X_full[t, :], Xdot_full[t, :], self._sample_images, t)
            else:
                mj_U, pos, ind, targets = policy.act(X_full, Xdot_full, self._sample_images, t, init_model=self._model[condition])
                add_traj = True
                if add_traj:
                    self.large_images_traj += self.add_traj_visual(self.large_images[t], pos, ind, targets)

            U[t, :] = mj_U

            for _ in range(self._hyperparams['substeps']):
                self._model[condition].data.ctrl = mj_U
                self._model[condition].step()         #simulate the model in mujoco

        if self._hyperparams['record']:
            self.save_gif()

        return X_full, Xdot_full, U, self._sample_images


    def _store_image(self,t, condition):
        """
        store image at time index t
        """
        self.model_nomarkers.data.qpos = self._model[condition].data.qpos
        self.model_nomarkers.data.qvel = self._model[condition].data.qvel
        self.model_nomarkers.step()
        self._small_viewer.loop_once()

        img_string, width, height = self._large_viewer.get_image()
        largeimage = np.fromstring(img_string, dtype='uint8').reshape(
                (480, 480, self._hyperparams['image_channels']))[::-1, :, :]

        # import pdb; pdb.set_trace()
        self.large_images.append(largeimage)

        img_string, width, height = self._small_viewer.get_image()
        img = np.fromstring(img_string, dtype='uint8').reshape((height, width, self._hyperparams['image_channels']))[::-1,:,:]

        self._sample_images[t,:,:,:] = img


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

                # target points #####
                # x = targets[smp, itr, :, 1]
                # y = targets[smp, itr, :, 0]
                #
                # if smp == bestindices[itr][0]:
                #     plt.plot(x, y, zorder=1, marker='o', color='y', linestyle='--')
                # elif smp in bestindices[itr][1:]:
                #     plt.plot(x, y, zorder=1, marker='o', color='r', linestyle='--')
                # else:
                #     if smp % 5 == 0:
                #         plt.plot(x, y, zorder=1, marker='o', color='b', linestyle='--')


            fig.canvas.draw()  # draw the canvas, cache the renderer

            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            large_sample_images_traj.append(data)

            # plt.show()
            # print 'timestep', t, 'iter', itr
            # from PIL import Image
            # Image.fromarray(data).show()

            # import pdb;
            # pdb.set_trace()

        return large_sample_images_traj

    def save_gif(self):
        file_path = self._hyperparams['record']
        from video_prediction.utils_vpred.create_gif import npy_to_gif
        npy_to_gif(self.large_images_traj, file_path)

    def _init(self, condition):
        """
        Set the world to a given model, and run kinematics.
        Args:
            condition: Which condition to initialize.
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

        if self._hyperparams['x0'][condition].shape[0] > 4: # if object pose explicit do not sample poses
            object_pos = self._hyperparams['x0'][condition][4:]
        else:
            object_pos= create_pos()

        # Initialize world/run kinematics
        x0 = self._hyperparams['x0'][condition]
        if 'goal_point' in self._hyperparams.keys():
            goal = np.append(self._hyperparams['goal_point'], [.1])   # goal point
            ref = np.append(object_pos[:2], [.1]) # reference point
            self._model[condition].data.qpos = np.concatenate((x0[:2], object_pos,goal, ref), 0)
        else:
            self._model[condition].data.qpos = np.concatenate((x0[:2], object_pos), 0)
        self._model[condition].data.qvel = np.zeros_like(self._model[condition].data.qvel)

        # self._model[condition].data_files.qpos[2:] = self._hyperparams['initial_object_pos']
        mjlib.mj_kinematics(self._model[condition].ptr, self._model[condition].data.ptr)
        mjlib.mj_comPos(self._model[condition].ptr, self._model[condition].data.ptr)
        mjlib.mj_tendon(self._model[condition].ptr, self._model[condition].data.ptr)
        mjlib.mj_transmission(self._model[condition].ptr, self._model[condition].data.ptr)
        
    def _init_sample(self, condition):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
        """
        sample = Sample(self)

        # Initialize world/run kinematics
        self._init(condition)

        # Initialize sample with stuff from _data
        data = self._model[condition].data
        #    ;
        sample.set(JOINT_ANGLES, data.qpos.flatten(), t=0)
        sample.set(JOINT_VELOCITIES, data.qvel.flatten(), t=0)
        eepts = data.site_xpos.flatten()
        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)
        jac = np.zeros([eepts.shape[0], self._model[condition].nq])
        for site in range(eepts.shape[0] // 3):
            idx = site * 3
            temp = np.zeros((3, jac.shape[1]))
            mjlib.mj_jacSite(self._model[condition].ptr, self._model[condition].data.ptr, temp.ctypes.data_as(POINTER(c_double)), 0, site)
            jac[idx:(idx+3), :] = temp
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)


        # save initial image to meta data_files
        img_string, width, height = self._small_viewer.get_image()
        img = np.fromstring(img_string, dtype='uint8').reshape(height, width, 3)[::-1,:,:]

        #downsampling the image
        img = Image.fromarray(img, 'RGB')
        img.thumbnail((80,60), Image.ANTIALIAS)
        img = np.array(img)
        img_data = np.transpose(img, (1, 0, 2)).flatten()

        #

        # if initial image is an observation, replicate it for each time step
        if CONTEXT_IMAGE in self.obs_data_types:
            sample.set(CONTEXT_IMAGE, np.tile(img_data, (self.T, 1)), t=None)
        else:
            sample.set(CONTEXT_IMAGE, img_data, t=None)
        sample.set(CONTEXT_IMAGE_SIZE, np.array([self._hyperparams['image_channels'],
                                                self._hyperparams['image_width'],
                                                self._hyperparams['image_height']]), t=None)
        # only save subsequent images if image is part of observation
        if RGB_IMAGE in self.obs_data_types:
            sample.set(RGB_IMAGE, img_data, t=0)
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
        return sample

    def _set_sample(self, sample, mj_X, t, condition):
        """
        Set the data_files for a sample for one time step.
        Args:
            sample: Sample object to set data_files for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
        """
        #

        sample.set(JOINT_ANGLES, np.array(mj_X[self._joint_idx]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(mj_X[self._vel_idx]), t=t+1)
        curr_eepts = self._data.site_xpos.flatten()
        sample.set(END_EFFECTOR_POINTS, curr_eepts, t=t+1)
        prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
        eept_vels = (curr_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)
        jac = np.zeros([curr_eepts.shape[0], self._model[condition].nq])
        for site in range(curr_eepts.shape[0] // 3):
            idx = site * 3
            temp = np.zeros((3, jac.shape[1]))
            mjlib.mj_jacSite(self._model[condition].ptr, self._model[condition].data.ptr, temp.ctypes.data_as(POINTER(c_double)), 0, site)
            jac[idx:(idx+3), :] = temp

        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t+1)