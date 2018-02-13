import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time

# Add lsdc/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# from lsdc.gui.gps_training_gui import GPSTrainingGUI
# from python_visual_mpc.video_prediction.setup_predictor_simple import setup_predictor
from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
from python_visual_mpc.visual_mpc_core.infrastructure.utility import *

from datetime import datetime
import pdb
import cPickle
import cv2
import shutil

class Sim(object):
    """ Main class to run algorithms and experiments. """

    def __init__(self, config, quit_on_end=False, gpu_id=0, ngpu=1):

        self._hyperparams = config
        self.agent = config['agent']['type'](config['agent'])
        self.agentparams = config['agent']
        self._data_save_dir = self.agentparams['data_save_dir']  # directory where to save trajectories

        if 'netconf' in config['policy']:
            params = imp.load_source('params', config['policy']['netconf'])
            netconf = params.configuration

        if 'usenet' in config['policy']:
            if 'setup_predictor' in netconf:
                if 'use_ray' in netconf:
                    self.predictor = netconf['setup_predictor'](netconf, config['policy'], ngpu)
                else:
                    self.predictor = netconf['setup_predictor'](netconf, gpu_id, ngpu)

            if 'warp_objective' in config['policy']:
                params = imp.load_source('params', config['policy']['gdnconf'])
                gdnconf = params.configuration
                self.goal_image_waper = setup_gdn(gdnconf, gpu_id)
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor, self.goal_image_waper)
            else:
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor)
        else:
            self.policy = config['policy']['type'](config['agent'], config['policy'])

        self.trajectory_list = []
        self.im_score_list = []

        try:
            os.remove(self._hyperparams['agent']['image_dir'])
        except:
            pass

    def run(self):
        for i in range(self._hyperparams['start_index'], self._hyperparams['end_index']+1):
            self._take_sample(i)

    def _take_sample(self, sample_index):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            sample_index: Sample index.
        Returns: None
        """
        start = datetime.now()
        traj = self.agent.sample(self.policy, sample_index)

        if self._hyperparams['save_data']:
            self.save_data(traj, sample_index)

        end = datetime.now()
        print 'time elapsed for one trajectory sim', end - start

    def save_data(self, traj, itr):
        """
        :param traj:
        :param itr: index of trajectory
        :return:
        """

        if 'save_raw_images' in self._hyperparams:
            ngroup = self._hyperparams['ngroup']
            if itr % ngroup == 0:
                self.igrp = itr / ngroup
                self.group_folder = self._data_save_dir + '/traj_group{}'.format(self.igrp)
                os.makedirs(self.group_folder)

            self.traj_folder = self.group_folder + '/traj{}'.format(itr)
            self.image_folder = self.traj_folder + '/images'
            self.depth_image_folder = self.traj_folder + '/depth_images'

            if os.path.exists(self.traj_folder):
                print 'trajectory folder {} already exists, deleting the folder'.format(self.traj_folder)
                shutil.rmtree(self.traj_folder)

            os.makedirs(self.traj_folder)
            os.makedirs(self.image_folder)
            os.makedirs(self.depth_image_folder)

            if traj.goal_mask != None:
                self.mask_folder = self.traj_folder + '/goal_masks'
                os.makedirs(self.mask_folder)

            self.state_action_pkl_file = self.traj_folder + '/state_action.pkl'

            #save pkl file:
            with open(self.state_action_pkl_file, 'wb') as f:
                dict = {'qpos': traj.X_full,
                        'qvel': traj.Xdot_full,
                        'actions': traj.actions,
                        'object_full_pose': traj.Object_full_pose
                        }
                if 'gen_xml' in self.agentparams:
                    dict['obj_statprop'] = traj.obj_statprop

                if 'goal_mask' in self.agentparams:
                    dict['goal_mask'] = traj.goal_mask
                cPickle.dump(dict, f)

            for t in range(traj.T):
                image_name = self.image_folder+ "/im{}.png".format(t)
                cv2.imwrite(image_name, traj._sample_images[t][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

            if traj.goal_mask != None:
                for i in range(traj.goal_mask.shape[0]):
                    name = self.mask_folder + "/goal_mask_ob{}.png".format(i)
                    cv2.imwrite(name, 255*traj.goal_mask[i][:, :], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
        else:
            #save tfrecords
            traj = copy.deepcopy(traj)
            self.trajectory_list.append(traj)
            if 'traj_per_file' in self._hyperparams:
                traj_per_file = self._hyperparams['traj_per_file']
            else:
                traj_per_file = 256
            print 'traj_per_file', traj_per_file
            if len(self.trajectory_list) == traj_per_file:
                filename = 'traj_{0}_to_{1}' \
                    .format(itr - traj_per_file + 1, itr)

                from utility.save_tf_record import save_tf_record
                save_tf_record(filename, self.trajectory_list, self.agentparams)

                self.trajectory_list = []