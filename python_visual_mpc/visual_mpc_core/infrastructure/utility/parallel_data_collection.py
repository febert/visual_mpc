from multiprocessing import Pool
import argparse
import imp
import os
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

from python_visual_mpc.visual_mpc_core.infrastructure.utility.create_configs import CollectGoalImageSim
import ray

def worker(conf):
    print 'started process with PID:', os.getpid()
    print 'making trajectories {0} to {1}'.format(
        conf['start_index'],
        conf['end_index'],
    )

    random.seed(None)
    np.random.seed(None)

    if 'simulator' in conf:
        Simulator = CollectGoalImageSim
        print 'use collect goalimage sim'
    else:
        Simulator = Sim

    s = Simulator(conf)
    s.run()

# @ray.remote
def bench_worker(conf):
    print 'started process with PID:', os.getpid()
    random.seed(None)
    np.random.seed(None)
    perform_benchmark(conf, gpu_id=conf['gpu_id'])


def main():
    parser = argparse.ArgumentParser(description='run parllel data collection')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--nworkers', type=int, help='use multiple threads or not', default=10)

    args = parser.parse_args()
    exp_name = args.experiment

    n_worker = args.nworkers
    if args.nworkers == 1:
        parallel = False
    else:
        parallel = True
    print 'parallel ', bool(parallel)

    basepath = os.path.abspath(python_visual_mpc.__file__)
    basepath = '/'.join(str.split(basepath, '/')[:-2])
    data_coll_dir = basepath + '/pushing_data/' + exp_name
    hyperparams_file = data_coll_dir + '/hyperparams.py'
    do_benchmark = False

    if os.path.isfile(hyperparams_file):
        hyperparams = imp.load_source('hyperparams', hyperparams_file).config
        n_traj = hyperparams['end_index']
    else:
        print 'doing benchmark ...'
        do_benchmark = True
        experimentdir = basepath + '/experiments/cem_exp/benchmarks/' + exp_name
        hyperparams_file = experimentdir + '/mod_hyper.py'
        hyperparams = imp.load_source('hyperparams', hyperparams_file).config
        n_traj = hyperparams['end_index']
        hyperparams['bench_dir'] = experimentdir

    traj_per_worker = int(n_traj / np.float32(n_worker))
    start_idx = [traj_per_worker * i for i in range(n_worker)]
    end_idx =  [traj_per_worker * (i+1)-1 for i in range(n_worker)]

    conflist = []

    hyperparams['agent']['data_save_dir'] = os.path.join(os.environ['VMPC_DATA_DIR'], hyperparams['agent']['data_save_dir'])  # directory where to save trajectories

    if 'gen_xml' in hyperparams['agent']: #remove old auto-generated xml files
        os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))


    if do_benchmark:
        use_worker = bench_worker
    else: use_worker = worker

    use_ray = False
    if use_ray:
        ray.init()
        id_list = []
        for i in range(n_worker):
            modconf = copy.deepcopy(hyperparams)
            modconf['start_index'] = start_idx[i]
            modconf['end_index'] = end_idx[i]
            modconf['gpu_id'] = i
            id_list.append(use_worker.remote(modconf))

        res = [ray.get(id) for id in id_list]
    else:
        for i in range(n_worker):
            modconf = copy.deepcopy(hyperparams)
            modconf['start_index'] = start_idx[i]
            modconf['end_index'] = end_idx[i]
            modconf['gpu_id'] = i
            conflist.append(modconf)
        if parallel:
            p = Pool(n_worker)
            p.map(use_worker, conflist)
        else:
            use_worker(conflist[0])

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