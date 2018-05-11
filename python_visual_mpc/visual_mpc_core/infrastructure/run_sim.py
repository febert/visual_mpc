import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import pdb

from timewarp_prediction.multipush.frame_cgan import testModel
# Add lsdc/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# from lsdc.gui.gps_training_gui import GPSTrainingGUI
#from python_visual_mpc.video_prediction.setup_predictor_simple import setup_predictor
from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
from python_visual_mpc.visual_mpc_core.infrastructure.utility import *

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import cv2
import shutil
import numpy as np
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger


class Sim(object):
    """ Main class to run algorithms and experiments. """

    def __init__(self, config, gpu_id=0, ngpu=1, logger=None):
        self._hyperparams = config
        self.agent = config['agent']['type'](config['agent'])
        self.agentparams = config['agent']
        self.policyparams = config['policy']
        if logger == None:
            self.logger = Logger(printout=True)
        else: self.logger = logger

        if 'RESULT_DIR' in os.environ:
            self.agentparams['data_save_dir'] = os.environ['RESULT_DIR'] + '/data/train'
        self._data_save_dir = self.agentparams['data_save_dir']
        self.agentparams['gpu_id'] = gpu_id

        if 'current_dir' in self._hyperparams:
            self._timing_file = self._hyperparams['current_dir'] + '/timing_file{}.txt'.format(os.getpid())
        else: self._timing_file = None

        if 'netconf' in config['policy']:
            params = imp.load_source('params', config['policy']['netconf'])
            netconf = params.configuration
            self.predictor = netconf['setup_predictor'](self._hyperparams, netconf, gpu_id, ngpu, self.logger)

            if 'warp_objective' in config['policy'] or 'register_gtruth' in config['policy']:
                params = imp.load_source('params', config['policy']['gdnconf'])
                gdnconf = params.configuration
                self.goal_image_warper = setup_gdn(gdnconf, gpu_id)
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor, self.goal_image_warper)
            else:
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor)

            if 'intmstep' in self.policyparams:
                if 'INTM_PRED_DATA' in os.environ:
                    modelpath = self.policyparams['intmstep']['pretrained'].partition('timewarp_prediction')[2]
                    model_path = os.environ['INTM_PRED_DATA'] + modelpath
                    pdb.set_trace()
                else:
                    model_path = self.policyparams['intmstep']['pretrained']
                self.intmstep_predictor = testModel(model_path)
                self.policy.intmstep_predictor = self.intmstep_predictor
        else:
            self.policy = config['policy']['type'](self.agent._hyperparams, config['policy'])

        self.trajectory_list = []
        self.im_score_list = []

        try:
            os.remove(self._hyperparams['agent']['image_dir'])
        except:
            pass

    def reset_policy(self):
        if 'netconf' in self.policyparams:
            if 'warp_objective' in self.policyparams or 'register_gtruth' in self.policyparams:
                self.policy = self.policyparams['type'](self.agent._hyperparams,
                                                              self.policyparams, self.predictor, self.goal_image_warper)
            else:
                self.policy = self.policyparams['type'](self.agent._hyperparams,
                                                              self.policyparams, self.predictor)
        else:
            self.policy = self.policyparams['type'](self.agent._hyperparams, self.policyparams)

        if 'intmstep' in self.policyparams:
            self.policy.intmstep_predictor = self.intmstep_predictor

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
        if 'save_raw_images' in self._hyperparams:
            ngroup = self._hyperparams['ngroup']
            self.igrp = itr // ngroup
            self.group_folder = self._data_save_dir + '/traj_group{}'.format(self.igrp)
            if not os.path.exists(self.group_folder):
                os.makedirs(self.group_folder)

            self.traj_folder = self.group_folder + '/traj{}'.format(itr)
            self.image_folder = self.traj_folder + '/images'
            self.depth_image_folder = self.traj_folder + '/depth_images'

            if os.path.exists(self.traj_folder):
                self.logger.log('trajectory folder {} already exists, deleting the folder'.format(self.traj_folder))
                shutil.rmtree(self.traj_folder)

            os.makedirs(self.traj_folder)
            self.logger.log('writing: ', self.traj_folder)
            os.makedirs(self.image_folder)
            os.makedirs(self.depth_image_folder)

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

                if 'make_gtruth_flows' in self.agentparams:
                    dict['bwd_flow'] = traj.bwd_flow
                    dict['ob_masks'] = traj.ob_masks
                    dict['arm_masks'] = traj.arm_masks
                if 'posmode' in self.agentparams:
                    dict['target_qpos'] = traj.target_qpos

                if hasattr(traj, "plan_stat"):
                    dict['plan_stat'] = traj.plan_stat

                pickle.dump(dict, f)

            for t in range(traj.T):
                image_name = self.image_folder+ "/im{}.png".format(t)
                cv2.imwrite(image_name, traj.images[t][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
                if 'medium_image' in self.agentparams:
                    image_name = self.image_folder+ "/im_med{}.png".format(t)
                    cv2.imwrite(image_name, traj._medium_images[t][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

            if 'first_last_noarm' in self.agentparams:
                for t in range(2):
                    image_name = self.image_folder+ "/noarm{}.png".format(t)
                    cv2.imwrite(image_name, traj.first_last_noarm[t][:,:,::-1], [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

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
                filename = 'traj_{0}_to_{1}' \
                    .format(itr - traj_per_file + 1, itr)
                from .utility.save_tf_record import save_tf_record
                self.logger.log('Writing', self.agentparams['data_save_dir'] + '/'+ filename)
                save_tf_record(filename, self.trajectory_list, self.agentparams)
                self.trajectory_list = []

                write_scores(itr, self.trajectory_list, filename, self.agentparams)


def write_scores(itr, trajlist, filename, agentparams):
    dir = '/'.join(str.split(agentparams['data_save_dir'], '/')[:-1])
    dir += '/scores'
    filename = filename.partition('.')[0] + '_score.pkl'
    scores = {}
    for itr, traj in zip(range(itr, len(trajlist)), trajlist):
        scores[itr] = {'improvement':traj.improvement,
                       'final_poscost':traj.final_poscost,
                       'initial_poscost':traj.initial_poscost}
    pickle.dump(scores, open(os.path.join(dir, filename)), 'wb')

def plot_warp_err(traj, dir):
    start_err = []
    goal_err = []
    tradeoff = []
    for tstep in traj.plan_stat[1:]:
        if 'start_warp_err' in tstep:
            start_err.append(tstep['start_warp_err'])
        if 'goal_warp_err' in tstep:
            goal_err.append(tstep['goal_warp_err'])
        tradeoff.append(tstep['tradeoff'])

    tradeoff = np.stack(tradeoff, 0)
    start_err = np.array(start_err)
    goal_err = np.array(goal_err)
    plt.figure()
    ax = plt.gca()
    ax.plot(start_err, marker ='d', label='start')
    ax.plot(goal_err, marker='o', label='goal')
    ax.legend()
    plt.savefig(dir + '/warperrors.png')

    plt.figure()
    ax = plt.gca()
    ax.plot(tradeoff[:,0], marker='d', label='tradeoff for start')
    ax.plot(tradeoff[:,1], marker='d', label='tradeoff for goal')
    ax.legend()
    plt.savefig(dir + '/tradeoff.png')

def plot_dist(traj, dir):
    goal_dist = np.stack(traj.goal_dist, axis=0)
    plt.figure()
    for ob in range(goal_dist.shape[1]):
        plt.plot(goal_dist[:,ob])
    plt.savefig(dir + '/goal_dist.png')
