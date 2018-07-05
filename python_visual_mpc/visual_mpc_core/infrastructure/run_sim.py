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
import pickle as pkl

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
        obs_dict, policy_out = self.agent.sample(self.policy, sample_index)
        t_traj = time.time() - t_traj

        t_save = time.time()
        if self._hyperparams['save_data']:
            self.save_data(obs_dict, policy_out, sample_index)
        t_save = time.time() - t_save

        if self._timing_file is not None:
            with open(self._timing_file,'a') as f:
                f.write("{} trajtime {} savetime {}\n".format(sample_index, t_traj, t_save))

        if 'verbose' in self.policyparams:
            if self.agent.goal_obj_pose is not None:
                plot_dist(traj, self.agentparams['record'])
            if 'register_gtruth' in self.policyparams:
                try:
                    plot_warp_err(traj, self.agentparams['record'])
                except:
                    print('plot warperr failed!!!')

    def save_data(self, obs_dict, policy_outputs, itr):
        """
        :param traj:
        :param itr: index of trajectory
        :return:
        """
        if 'RESULT_DIR' in os.environ:
            data_save_dir = os.environ['RESULT_DIR'] + '/data'
        else: data_save_dir = self.agentparams['data_save_dir']
        data_save_dir += '/' + self.task_mode

        ngroup = self._hyperparams['ngroup']
        igrp = itr // ngroup
        group_folder = data_save_dir + '/traj_group{}'.format(igrp)
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        traj_folder = group_folder + '/traj{}'.format(itr)
        if os.path.exists(traj_folder):
            self.logger.log('trajectory folder {} already exists, deleting the folder'.format(traj_folder))
            shutil.rmtree(traj_folder)

        os.makedirs(traj_folder)
        self.logger.log('writing: ', traj_folder)
        if 'images' in obs_dict:
            images = obs_dict.pop('images')
            T, n_cams = images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/images{}')
            for t in range(T):
                for i in range(n_cams):
                    cv2.imwrite('{}/images{}/im_{}.png'.format(traj_folder, i, t), images[t, i])

        with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(obs_dict, file)
        with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
            pkl.dump(policy_outputs, file)

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
