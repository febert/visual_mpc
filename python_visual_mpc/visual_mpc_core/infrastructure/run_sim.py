import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import pdb

# Add lsdc/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# from lsdc.gui.gps_training_gui import GPSTrainingGUI
#from python_visual_mpc.video_prediction.setup_predictor_simple import setup_predictor
from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
from python_visual_mpc.visual_mpc_core.infrastructure.utility import *


from datetime import datetime
import pickle
import cv2
import shutil
import numpy as np
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
from .utility.save_tf_record import save_tf_record


class Sim(object):
    """ Main class to run algorithms and experiments. """

    def __init__(self, config, gpu_id=0, ngpu=1, logger=None, mode='train'):
        self._hyperparams = config
        self.agent = config['agent']['type'](config['agent'])
        self.agentparams = config['agent']
        self.policyparams = config['policy']
        if logger == None:
            self.logger = Logger(printout=True)
        else: self.logger = logger
        self.logger.log('started sim')

        self.agentparams['gpu_id'] = gpu_id
        self.task_mode = mode

        if 'do_timing' in self._hyperparams:
            self._timing_file = self._hyperparams['current_dir'] + '/timing_file{}.txt'.format(os.getpid())
        else: self._timing_file = None

        if 'netconf' in config['policy']:
            params = imp.load_source('params', config['policy']['netconf'])
            netconf = params.configuration
            self.predictor = netconf['setup_predictor'](config, netconf, gpu_id, ngpu, self.logger)

            if 'warp_objective' in config['policy'] or 'register_gtruth' in config['policy']:
                params = imp.load_source('params', config['policy']['gdnconf'])
                gdnconf = params.configuration
                self.goal_image_warper = setup_gdn(gdnconf, gpu_id)
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor, self.goal_image_warper)
            elif 'actionproposal_conf' in config['policy']:
                self.actionproposal_policy = config['policy']['actionproposal_setup'](config['policy']['actionproposal_conf'], self.agentparams, self.policyparams)
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor, self.actionproposal_policy)
            else:
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor)
        else:
            imitationconf = {}
            self.policy = config['policy']['type'](imitationconf, self.agent._hyperparams, config['policy'])

        self.trajectory_list = []
        self.im_score_list = []
        try:
            os.remove(self._hyperparams['agent']['image_dir'])
        except:
            pass

    def _init_policy(self):
        if 'netconf' in self.policyparams:
            if 'warp_objective' in self.policyparams or 'register_gtruth' in self.policyparams:
                self.policy = self.policyparams['type'](self.agent._hyperparams,
                                                              self.policyparams, self.predictor, self.goal_image_warper)
            elif 'actionproposal_conf' in self.policyparams:
                self.policy = self.policyparams['type'](self.agent._hyperparams, self.policyparams, self.predictor, self.actionproposal_policy)
            else:
                self.policy = self.policyparams['type'](self.agent._hyperparams,
                                                              self.policyparams, self.predictor)
        else:
            imitationconf = {}
            self.policy = self.policyparams['type'](imitationconf, self.agent._hyperparams, self.policyparams)

    def run(self):
        for i in range(self._hyperparams['start_index'], self._hyperparams['end_index']+1):
            self.take_sample(i)

    def take_sample(self, sample_index):
        self._init_policy()

        t_traj = time.time()
        traj = self.agent.sample(self.policy, sample_index)
        t_traj = time.time() - t_traj

        t_save = time.time()
        if self._hyperparams['save_data']:
            self.save_data(traj, sample_index)
        t_save = time.time() - t_save

        if self._timing_file is not None:
            with open(self._timing_file,'a') as f:
                f.write("{} trajtime {} savetime {}\n".format(sample_index, t_traj, t_save))

        if 'verbose' in self.policyparams:
            if self.agent.goal_obj_pose is not None:
                plot_dist(traj, self.agentparams['record'])
            if 'register_gtruth' in self.policyparams:
                plot_warp_err(traj, self.agentparams['record'])
        return traj

    def save_data(self, traj, itr):
        """
        :param traj:
        :param itr: index of trajectory
        :return:
        """
        if 'RESULT_DIR' in os.environ:
            self._data_save_dir = os.environ['RESULT_DIR'] + '/data'
        else: self._data_save_dir = self.agentparams['data_save_dir']
        self._data_save_dir += '/' + self.task_mode

        if 'save_raw_images' in self._hyperparams:
            ngroup = self._hyperparams['ngroup']
            self.igrp = itr // ngroup
            self.group_folder = self._data_save_dir + '/traj_group{}'.format(self.igrp)
            if not os.path.exists(self.group_folder):
                os.makedirs(self.group_folder)

            self.traj_folder = self.group_folder + '/traj{}'.format(itr)

            if 'cameras' in self.agentparams:
                image_folders = [self.traj_folder + '/images{}'.format(i) for i in range(len(self.agentparams['cameras']))]
            else:
                self.image_folder = self.traj_folder + '/images'


            if os.path.exists(self.traj_folder):
                self.logger.log('trajectory folder {} already exists, deleting the folder'.format(self.traj_folder))
                shutil.rmtree(self.traj_folder)

            os.makedirs(self.traj_folder)
            self.logger.log('writing: ', self.traj_folder)
            if 'cameras' in self.agentparams:
                for f in image_folders:
                    os.makedirs(f)
            else:
                os.makedirs(self.image_folder)

            self.state_action_pkl_file = self.traj_folder + '/state_action.pkl'

            #save pkl file:
            with open(self.state_action_pkl_file, 'wb') as f:
                dict = {'qpos': traj.X_full,
                        'qvel': traj.Xdot_full,
                        'actions': traj.actions,
                        'object_full_pose': traj.Object_full_pose
                        }
                if hasattr(traj, 'obj_statprop'):
                    dict['obj_statprop'] = traj.obj_statprop

                if 'goal_mask' in self.agentparams:
                    dict['goal_mask'] = traj.goal_mask

                if 'make_gtruth_flows' in self.agentparams:
                    dict['bwd_flow'] = traj.bwd_flow
                    dict['ob_masks'] = traj.ob_masks
                    dict['arm_masks'] = traj.arm_masks
                if 'posmode' in self.agentparams:
                    dict['target_qpos'] = traj.target_qpos
                    if hasattr(traj, 'mask_rel'):
                        dict['mask_rel'] = traj.mask_rel
                    if hasattr(traj, 'desig_pos'):
                        dict['obj_start_end_pos'] = traj.desig_pos
                if hasattr(traj, 'touch_sensors'):
                    dict['finger_sensors'] = traj.touch_sensors
                if hasattr(traj, "plan_stat"):
                    dict['plan_stat'] = traj.plan_stat

                pickle.dump(dict, f)
            if 'cameras' in self.agentparams:
                for i, image_folder in enumerate(image_folders):
                    for t in range(traj.T):
                        image_name = image_folder + "/im{}.png".format(t)
                        cv2.imwrite(image_name, traj.images[t, i][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
                        if 'medium_image' in self.agentparams:
                            image_name = image_folder + "/im_med{}.png".format(t)
                            cv2.imwrite(image_name, traj._medium_images[t, i][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
            else:
                for t in range(traj.T):
                    image_name = self.image_folder+ "/im{}.png".format(t)
                    cv2.imwrite(image_name, traj.images[t,0][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
                    if 'medium_image' in self.agentparams:
                        image_name = self.image_folder+ "/im_med{}.png".format(t)
                        cv2.imwrite(image_name, traj._medium_images[t][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

            if 'first_last_noarm' in self.agentparams:
                for t in range(2):
                    image_name = self.image_folder + "/noarm{}.png".format(t)
                    cv2.imwrite(image_name, traj.first_last_noarm[t][:, :, ::-1],
                                     [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

            if traj.goal_mask != None:
                folder = self.traj_folder + '/goal_masks'
                os.makedirs(folder)
                for i in range(traj.goal_mask.shape[0]):
                    name = folder + "/goal_mask_ob{}.png".format(i)
                    cv2.imwrite(name, 255*traj.goal_mask[i][:, :], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

            if 'make_gtruth_flows' in self.agentparams:
                folder = self.traj_folder + '/gtruth_flows'
                os.makedirs(folder)
                for t in range(traj.T-1):
                    plt.imshow(traj.bwd_flow[t, :,:, 0])
                    plt.savefig(folder + "/t{}_bwd_flow_col.png".format(t))
                    plt.imshow(traj.bwd_flow[t, :, :, 1])
                    plt.savefig(folder + "/t{}_bwd_flow_row.png".format(t))
            self.logger.log('done writing: ', self.traj_folder)
        else:
            #save tfrecords
            traj = copy.deepcopy(traj)
            self.trajectory_list.append(traj)
            if 'traj_per_file' in self._hyperparams:
                traj_per_file = self._hyperparams['traj_per_file']
            else:
                traj_per_file = 256
            self.logger.log('traj_per_file', traj_per_file)
            if len(self.trajectory_list) == traj_per_file:
                filename = 'traj_{0}_to_{1}'.format(itr - traj_per_file + 1, itr)
                self.logger.log('Writing', self._data_save_dir + '/' + filename)
                if self.task_mode != 'val_task':   # do not save validation tasks (but do save validation runs on randomly generated tasks)
                    save_tf_record(self._data_save_dir, filename, self.trajectory_list, self.agentparams)
                if self.agent.goal_obj_pose is not None:
                    dir = '/'.join(str.split(self._data_save_dir, '/')[:-1])
                    dir += '/scores/' + self.task_mode
                    write_scores(dir, filename, self.trajectory_list, self.logger)
                self.trajectory_list = []


def write_scores(dir, filename, trajlist, logger):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    filename = filename.partition('.')[0] + '_score.pkl'
    filename = os.path.join(dir, filename)
    logger.log('writing scorefile {}'.format(filename))
    scores = {}
    improvements = []
    final_poscost = []
    initial_poscost = []
    for traj in trajlist:
        improvements.append(traj.improvement)
        final_poscost.append(traj.final_poscost)
        initial_poscost.append(traj.initial_poscost)
    scores['improvement'] = improvements
    scores['final_poscost'] = final_poscost
    scores['initial_poscost'] = initial_poscost
    pickle.dump(scores, open(filename, 'wb'))


def plot_warp_err(traj, dir):
    warperrs = []
    tradeoff = []
    for tstep in traj.plan_stat[1:]:
        warperrs.append(tstep['warperrs'])
        tradeoff.append(tstep['tradeoff'])

    pdb.set_trace()
    tradeoff = np.stack(tradeoff, 0)
    warperrs = np.stack(warperrs, 0)


    pickle.dump({'warperrs':warperrs, 'tradeoff':tradeoff}, open(dir +  '/warperrs_tradeoff.pkl', 'wb'))

    # warperrs shape: tstep, ncam, numtrack
    plt.figure()
    ax = plt.gca()
    ax.plot(warperrs[:,0,0], marker ='d', label='start')
    ax.plot(warperrs[:,0,1], marker='o', label='goal')
    ax.legend()
    plt.savefig(dir + '/warperrors.png')

    plt.figure()
    ax = plt.gca()

    ax.plot(tradeoff[:,0,0], marker='d', label='tradeoff for start')
    ax.plot(tradeoff[:,0,1], marker='d', label='tradeoff for goal')
    ax.legend()
    plt.savefig(dir + '/tradeoff.png')

def plot_dist(traj, dir):
    goal_dist = np.stack(traj.goal_dist, axis=0)
    plt.figure()
    for ob in range(goal_dist.shape[1]):
        plt.plot(goal_dist[:,ob])
    plt.savefig(dir + '/goal_dist.png')
