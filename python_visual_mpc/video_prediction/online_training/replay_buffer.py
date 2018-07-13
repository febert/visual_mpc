import numpy as np
import os
import tensorflow as tf
import random
import ray
from collections import namedtuple
from python_visual_mpc.video_prediction.read_tf_records2 import build_tfrecord_input
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
import pdb

import time

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pickle
from tensorflow.python.platform import gfile
Traj = namedtuple('Traj', 'images X_Xdot_full actions')
import copy

from tensorflow.python.framework.errors_impl import OutOfRangeError, DataLossError

class ReplayBuffer(object):
    def __init__(self, conf, data_collectors=None, todo_ids=None, printout=False, mode='train'):
        self.logger = Logger(conf['logging_dir'], 'replay_log.txt', printout=printout)
        self.conf = conf
        self.onpolconf = conf['onpolconf']
        if 'agent' in conf:
            self.agentparams = conf['agent']
        self.ring_buffer = []
        self.mode = mode
        self.maxsize = self.onpolconf['replay_size'][mode]
        self.batch_size = conf['batch_size']
        self.data_collectors = data_collectors
        self.todo_ids = todo_ids
        self.scores = []
        self.num_updates = 0
        self.logger.log('init Replay buffer')
        self.tstart = time.time()

    def push_back(self, traj):
        assert traj.images.dtype == np.float32 and np.max(traj.images) <= 1.0
        self.ring_buffer.append(traj)
        if len(self.ring_buffer) > self.maxsize:
            self.ring_buffer.pop(0)
        if len(self.ring_buffer) % 100 == 0:
            self.logger.log('current size {}'.format(len(self.ring_buffer)))

    def get_batch(self):
        images = []
        states = []
        actions = []
        current_size = len(self.ring_buffer)
        for b in range(self.batch_size):
            i = random.randint(0, current_size-1)
            traj = self.ring_buffer[i]
            images.append(traj.images)
            states.append(traj.X_Xdot_full)
            actions.append(traj.actions)
        return np.stack(images,0), np.stack(states,0), np.stack(actions,0)

    def update(self, sess):
        done_id, self.todo_ids = ray.wait(self.todo_ids, timeout=0)
        if len(done_id) != 0:
            self.logger.log("len doneid {}".format(len(done_id)))
            for id in done_id:
                traj, info = ray.get(id)
                self.logger.log("received trajectory from {}, pushing back traj".format(info['collector_id']))
                self.push_back(traj)
                self.scores.append(traj.final_poscost)
                # relauch the collector if it hasn't done all its work yet.
                returning_collector = self.data_collectors[info['collector_id']]
                self.todo_ids.append(returning_collector.run_traj.remote())
                self.logger.log('restarting {}'.format(info['collector_id']))

                self.num_updates += 1

                if self.num_updates % 100 == 0:
                    plot_scores(self.scores, self.agentparams['result_dir'])

                self.logger.log('traj_per hour: {}'.format(self.num_updates/((time.time() - self.tstart)/3600)))
                self.logger.log('avg time per traj {}s'.format((time.time() - self.tstart)/self.num_updates))


class ReplayBuffer_Loadfiles(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(ReplayBuffer_Loadfiles, self).__init__(*args, **kwargs)
        self.loaded_filenames = []
        self.conf['max_epoch'] = 1
        self.improvement_avg = []
        self.final_poscost_avg = []

    def update(self, sess):
        # check if new files arrived:
        all_filenames = gfile.Glob(self.conf['data_dir'] + '/' + self.mode + '/*.tfrecords')

        to_load_filenames = []
        for name in all_filenames:
            if name not in self.loaded_filenames:
                to_load_filenames.append(name)
                self.loaded_filenames.append(name)

        if len(to_load_filenames) != 0:
            self.logger.log('loading files')
            self.logger.log(to_load_filenames)
            self.logger.log('start filling replay')
            try:
                dict = build_tfrecord_input(self.conf, input_files=to_load_filenames)
                ibatch = 0
                while True:
                    try:
                        images, actions, endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos']])
                        self.logger.log('getting batch {}'.format(ibatch))
                        ibatch +=1
                    except OutOfRangeError:
                        self.logger.log('OutOfRangeError')
                        break
                    for b in range(self.conf['batch_size']):
                        t = Traj(images[b], endeff[b], actions[b])
                        self.push_back(t)
                        self.num_updates += 1
                self.logger.log('done filling replay')
            except DataLossError:
                self.logger.log('DataLossError')

            self.logger.log('reading scores')
            self.get_scores(to_load_filenames)
            self.logger.log('writing scores plot to {}'.format(self.conf['result_dir']))
            plot_scores(self.conf['result_dir'], self.final_poscost_avg, self.improvement_avg)
            self.logger.log('traj_per hour: {}'.format(self.num_updates/((time.time() - self.tstart)/3600)))
            self.logger.log('avg time per traj {}s'.format((time.time() - self.tstart)/self.num_updates))

    def get_scores(self, to_load_filenames):
        for file in to_load_filenames:
            filenum = file.partition('train')[2].partition('.')[0]
            path = file.partition('train')[0]
            scorefile = path + 'scores/' + self.mode + '/' + filenum + '_score.pkl'
            try:
                dict_ = pickle.load(open(scorefile, 'rb'))
            except FileNotFoundError:
                self.logger.log('scorefile: {} not found'.format(scorefile))
                continue
            self.improvement_avg.append(np.mean(dict_['improvement']))
            self.final_poscost_avg.append(np.mean(dict_['final_poscost']))

        with open(self.conf['result_dir'] + '/scores.txt', 'w') as f:
            f.write('improvement averaged over batch, final_pos_cost averaged over batch\n')
            for i in range(len(self.improvement_avg)):
                f.write('{}: {} {}'.format(i, self.improvement_avg[i], self.final_poscost_avg[i]) + '\n')

    def preload(self, sess):
        self.logger.log('start prefilling replay')
        conf = copy.deepcopy(self.conf)
        conf['data_dir'] = conf['preload_data_dir']
        dict = build_tfrecord_input(conf, mode=self.mode)
        for i_run in range(self.onpolconf['fill_replay_fromsaved'][self.mode] // conf['batch_size']):
            images, actions, endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos']])
            for b in range(conf['batch_size']):
                t = Traj(images[b], endeff[b], actions[b])
                self.push_back(t)
        self.logger.log('done prefilling replay')


def plot_scores(dir, scores, improvement=None):

    plt.subplot(2,1,1)
    plt.plot(scores)
    plt.title('scores over collected data')
    plt.xlabel('collected trajectories')
    plt.xlabel('avg distances trajectories')

    if improvement is not None:
        plt.subplot(2,1,2)
        plt.plot(improvement)
        plt.title('improvments over collected data')
        plt.xlabel('collected trajectories')
        plt.xlabel('avg improvment trajectories')

    plt.savefig(dir + '/scores.png')
