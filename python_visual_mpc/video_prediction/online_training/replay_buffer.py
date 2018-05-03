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
Traj = namedtuple('Traj', 'images X_Xdot_full actions')

class ReplayBuffer(object):
    def __init__(self, agentparams, maxsize, batch_size, data_collectors=None, todo_ids=None):
        self.logger = Logger(agentparams['logging_dir'], 'replay_log.txt')
        self.agentparams = agentparams
        self.ring_buffer = []
        self.maxsize = maxsize
        self.batch_size = batch_size
        self.data_collectors = data_collectors
        self.todo_ids = todo_ids
        self.scores = []
        self.num_updates = 0
        self.logger.log('init Replay buffer')
        self.tstart = time.time()

    def push_back(self, traj):
        self.ring_buffer.append(traj)
        if len(self.ring_buffer) > self.maxsize:
            self.ring_buffer.pop(0)
        self.logger.log('current size {}'.format(len(self.ring_buffer)))

    def get_batch(self):
        images = []
        states = []
        actions = []
        current_size = len(self.ring_buffer)
        for b in range(self.batch_size):
            i = random.randint(0, current_size-1)
            traj = self.ring_buffer[i]
            images.append(traj.images.astype(np.float32)/255.)
            states.append(traj.X_Xdot_full)
            actions.append(traj.actions)
        return np.stack(images,0), np.stack(states,0), np.stack(actions,0)

    def update(self):
        done_id, self.todo_ids = ray.wait(self.todo_ids, timeout=0)
        if len(done_id) != 0:
            pdb.set_trace()
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


def plot_scores(scores, dir):
    plt.plot(scores)
    plt.title('scores over time')
    plt.xlabel('collected trajectories')
    plt.savefig(dir + '/scores.png')


