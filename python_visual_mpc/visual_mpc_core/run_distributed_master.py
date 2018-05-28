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
from python_visual_mpc.video_prediction.online_training.replay_buffer import ReplayBuffer_Loadfiles
import pickle
from python_visual_mpc.video_prediction.online_training.trainvid_online import trainvid_online
import matplotlib; matplotlib.use('Agg'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser(description='run parllel data collection')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=1)
    parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)
    parser.add_argument('--nsplit', type=int, help='number of splits', default=-1)
    parser.add_argument('--isplit', type=int, help='split id', default=-1)
    parser.add_argument('--iex', type=int, help='if different from -1 use only do example', default=-1)
    parser.add_argument('--printout', type=int, help='print to console if 1', default=0)


    args = parser.parse_args()
    trainvid_conf_file = args.experiment
    trainvid_conf = load_module(trainvid_conf_file, 'trainvid_conf')

    if 'RESULT_DIR' in os.environ:
        trainvid_conf['result_dir'] = os.environ['RESULT_DIR']
    else:
        trainvid_conf['result_dir'] = trainvid_conf['current_dir']

    printout = bool(args.printout)
    gpu_id = args.gpu_id

    logging_dir = trainvid_conf['current_dir'] + '/logging'
    trainvid_conf['logging_dir'] = logging_dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    onpolconf = trainvid_conf['onpolconf']
    train_rb = ReplayBuffer_Loadfiles(trainvid_conf, mode='train', maxsize=onpolconf['replay_size'], batch_size=16, printout=printout)
    val_rb = ReplayBuffer_Loadfiles(trainvid_conf, mode='val', maxsize=onpolconf['replay_size'], batch_size=16, printout=printout)
    trainvid_online(train_rb, val_rb, trainvid_conf, logging_dir, onpolconf, gpu_id, printout=True)

def load_module(hyperparams_file, name):
    loader = importlib.machinery.SourceFileLoader(name, hyperparams_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    hyperparams = mod.config
    return hyperparams


if __name__ == '__main__':
    main()