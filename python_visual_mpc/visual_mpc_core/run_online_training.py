from multiprocessing import Pool
import sys
import argparse
import os
import importlib.machinery
import importlib.util
from python_visual_mpc.visual_mpc_core.infrastructure.run_sim import Sim
from python_visual_mpc.visual_mpc_core.benchmarks import perform_benchmark
import copy
import random
import numpy as np
import shutil
import python_visual_mpc
import pdb
import glob
import re

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import combine_scores
from python_visual_mpc.visual_mpc_core.infrastructure.utility.create_configs import CollectGoalImageSim
import time
import ray
from python_visual_mpc.video_prediction.online_training.replay_buffer import ReplayBuffer
import pickle
from python_visual_mpc.video_prediction.online_training.trainvid_online import trainvid_online
import matplotlib; matplotlib.use('Agg'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt


@ray.remote
class Data_Collector(object):
    def __init__(self, conf, collector_id):
        print('started process with PID {} and collector_id {}'.format(os.getpid(), collector_id))
        self.itraj = conf['start_index']
        self.maxtraj = conf['end_index']
        self.colllector_id = collector_id
        random.seed(None)
        np.random.seed(None)
        self.conf = conf
        self.sim = Sim(conf, gpu_id=conf['gpu_id'])
        print('init data collectors done.')

    def run_traj(self):
        assert self.itraj < self.maxtraj
        if self.itraj % self.conf['onpolconf']['infnet_reload_freq'] == 0 and self.itraj != 0:
            self.conf['load_latest'] = ''
            self.sim = Sim(self.conf, gpu_id=self.conf['gpu_id'])
        print('-------------------------------------------------------------------')
        print('run number ', self.itraj)
        print('-------------------------------------------------------------------')

        # reinitilize policy between rollouts
        self.sim.reset_policy()
        record_dir = self.sim.agentparams['result_dir'] + '/verbose/traj{0}'.format(self.itraj)
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        self.sim.agent._hyperparams['record'] = record_dir
        traj = self.sim._take_sample(self.itraj)

        self.itraj += 1
        info = {'collector_id':self.colllector_id, 'itraj':self.itraj, 'maxtraj':self.maxtraj}
        return traj, info


def main():
    parser = argparse.ArgumentParser(description='run parllel data collection')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
    parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)
    parser.add_argument('--nsplit', type=int, help='number of splits', default=-1)
    parser.add_argument('--isplit', type=int, help='split id', default=-1)
    parser.add_argument('--iex', type=int, help='if different from -1 use only do example', default=-1)

    args = parser.parse_args()
    hyperparams_file = args.experiment
    exp_dir = '/'.join(str.split(hyperparams_file, '/')[:-1])
    trainvid_conf_file = exp_dir + '/trainvid_conf.py'
    gpu_id = args.gpu_id

    n_worker = args.nworkers
    if args.nworkers == 1:
        parallel = False
    else:
        parallel = True
    print('parallel ', bool(parallel))

    hyperparams = load_module(hyperparams_file, 'mod_hyper')
    trainvid_conf = load_module(trainvid_conf_file, 'trainvid_conf')

    if args.nsplit != -1:
        n_persplit = (hyperparams['end_index']+1)//args.nsplit
        hyperparams['start_index'] = args.isplit * n_persplit
        hyperparams['end_index'] = (args.isplit+1) * n_persplit -1

    n_traj = hyperparams['end_index'] - hyperparams['start_index'] +1
    traj_per_worker = int(n_traj // np.float32(n_worker))
    start_idx = [hyperparams['start_index'] + traj_per_worker * i for i in range(n_worker)]
    end_idx = [hyperparams['start_index'] + traj_per_worker * (i+1)-1 for i in range(n_worker)]

    if 'gen_xml' in hyperparams['agent']: #remove old auto-generated xml files
        try:
            os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))
        except: pass

    if 'RESULT_DIR' in os.environ:
        result_dir = os.environ['RESULT_DIR']
        if 'verbose' in hyperparams['policy'] and not os.path.exists(result_dir + '/verbose'):
            os.makedirs(result_dir + '/verbose')
        hyperparams['agent']['result_dir'] = result_dir
        hyperparams['agent']['data_save_dir'] = os.environ['RESULT_DIR'] + 'data/train'
    else:
        hyperparams['agent']['result_dir'] = hyperparams['current_dir']

    onpolconf = hyperparams['onpolconf']

    ray.init()
    data_collectors = []
    for i in range(n_worker):
        modconf = copy.deepcopy(hyperparams)
        modconf['start_index'] = start_idx[i]
        modconf['end_index'] = end_idx[i]
        modconf['gpu_id'] = i + gpu_id
        data_collectors.append(Data_Collector.remote(modconf, i))

    todo_ids = [d.run_traj.remote() for d in data_collectors]
    rb = ReplayBuffer(maxsize=onpolconf['replay_size'],
                      batch_size=16, data_collectors=data_collectors, todo_ids=todo_ids)

    if 'prefil_replay' in onpolconf:
        print('prefilling replay')
        path = trainvid_conf['data_dir'].partition('pushing_data')[2]
        trainvid_conf['data_dir'] = os.environ['VMPC_DATA_DIR'] + path
        rb.prefil(onpolconf['replay_size'], trainvid_conf)
        print('prefilling replay done.')

    while len(rb.ring_buffer) < onpolconf['replay_size']:
        rb.update()
    print("Replay buffer filled")
    trainvid_online(rb, trainvid_conf, i + gpu_id + 1)


def load_module(hyperparams_file, name):
    loader = importlib.machinery.SourceFileLoader(name, hyperparams_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    hyperparams = mod.config
    return hyperparams


if __name__ == '__main__':
    main()