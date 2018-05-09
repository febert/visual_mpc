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
import os
import matplotlib; matplotlib.use('Agg'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import combine_scores
from python_visual_mpc.visual_mpc_core.infrastructure.utility.create_configs import CollectGoalImageSim
import pickle

def worker(conf, iex=-1):
    print('started process with PID:', os.getpid())
    print('making trajectories {0} to {1}'.format(
        conf['start_index'],
        conf['end_index'],
    ))

    random.seed(None)
    np.random.seed(None)
    if 'simulator' in conf:
        Simulator = CollectGoalImageSim
        print('use collect goalimage sim')
    else:
        Simulator = Sim

    s = Simulator(conf)
    s.run()

def bench_worker(conf, iex=-1):
    print('started process with PID:', os.getpid())
    random.seed(None)
    np.random.seed(None)
    perform_benchmark(conf, iex, gpu_id=conf['gpu_id'])

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
    exp_name = '/'.join(str.split(hyperparams_file, '/')[:-1])
    gpu_id = args.gpu_id

    n_worker = args.nworkers
    if args.nworkers == 1:
        parallel = False
    else:
        parallel = True
    print('parallel ', bool(parallel))

    if 'benchmarks' in hyperparams_file:
        do_benchmark = True
    else: do_benchmark = False
    loader = importlib.machinery.SourceFileLoader('mod_hyper', hyperparams_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    hyperparams = mod.config

    if args.nsplit != -1:
        n_persplit = (hyperparams['end_index']+1)//args.nsplit
        hyperparams['start_index'] = args.isplit * n_persplit
        hyperparams['end_index'] = (args.isplit+1) * n_persplit -1

    n_traj = hyperparams['end_index'] - hyperparams['start_index'] +1
    traj_per_worker = int(n_traj // np.float32(n_worker))
    start_idx = [hyperparams['start_index'] + traj_per_worker * i for i in range(n_worker)]
    end_idx = [hyperparams['start_index'] + traj_per_worker * (i+1)-1 for i in range(n_worker)]

    conflist = []

    if 'gen_xml' in hyperparams['agent']: #remove old auto-generated xml files
        try:
            os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))
        except: pass

    if do_benchmark:
        use_worker = bench_worker
    else: use_worker = worker

    if 'RESULT_DIR' in os.environ:
        result_dir = os.environ['RESULT_DIR']
        if 'verbose' in hyperparams['policy'] and not os.path.exists(result_dir + '/verbose'):
            os.makedirs(result_dir + '/verbose')

        data_save_path = hyperparams['agent']['data_save_dir'].partition('pushing_data')[2]
        hyperparams['agent']['data_save_dir'] = os.environ['RESULT_DIR'] + data_save_path

    for i in range(n_worker):
        modconf = copy.deepcopy(hyperparams)
        modconf['start_index'] = start_idx[i]
        modconf['end_index'] = end_idx[i]
        modconf['gpu_id'] = i + gpu_id
        conflist.append(modconf)
    if parallel:
        p = Pool(n_worker)
        p.map(use_worker, conflist)
    else:
        use_worker(conflist[0], args.iex)

    if do_benchmark:
        if 'RESULT_DIR' in os.environ:
            result_dir = os.environ['RESULT_DIR']
        else: result_dir = hyperparams['current_dir']
        combine_scores(result_dir)
        sys.exit()

    traindir = modconf['agent']["data_save_dir"]
    testdir = '/'.join(traindir.split('/')[:-1] + ['/test'])
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    import shutil
    files = glob.glob(traindir + '/*')
    files = sorted_alphanumeric(files)
    if os.path.isfile(files[0]): #don't do anything if directory
        shutil.move(files[0], testdir)

def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

if __name__ == '__main__':
    main()