from python_visual_mpc.visual_mpc_core.infrastructure.synchronize_tfrecs import sync
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
from python_visual_mpc.visual_mpc_core.infrastructure.utility.combine_scores import combine_scores
from python_visual_mpc.visual_mpc_core.infrastructure.utility.create_configs import CollectGoalImageSim
import pickle


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
    gpu_id = args.gpu_id

    n_worker = args.nworkers
    # if args.nworkers == 1:
    #     parallel = False
    # else:
    #     parallel = True
    parallel = True
    print('parallel ', bool(parallel))


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


    if 'gen_xml' in hyperparams['agent']: #remove old auto-generated xml files
        try:
            os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))
        except: pass


    if 'RESULT_DIR' in os.environ:
        result_dir = os.environ['RESULT_DIR']
        if 'verbose' in hyperparams['policy'] and not os.path.exists(result_dir + '/verbose'):
            os.makedirs(result_dir + '/verbose')

        data_save_path = hyperparams['agent']['data_save_dir'].partition('pushing_data')[2]
        hyperparams['agent']['data_save_dir'] = os.environ['RESULT_DIR'] + data_save_path


    if 'master_datadir' in hyperparams['agent']:
        import ray
        ray.init()
        sync_todo_id = sync.remote(hyperparams['agent'])
        print('launched sync')

    for i in range(n_worker):
        if i == n_worker - 1:
            detach = ''
        else: detach = '&'
        script = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-1]) + '/visual_mpc_core/bench_worker_cmd.py'
        cmd = "python {} {} {} {} {} {}".format(script, args.experiment, i, start_idx[i], end_idx[i], detach)
        print(cmd)
        os.system(cmd)


    if 'master_datadir' in hyperparams['agent']:
        ray.wait([sync_todo_id])

    if 'benchmarks' in hyperparams_file:
        if 'RESULT_DIR' in os.environ:
            result_dir = os.environ['RESULT_DIR']
        else:
            result_dir = hyperparams['current_dir']
        combine_scores(hyperparams, result_dir)
        sys.exit()

    traindir = hyperparams['agent']["data_save_dir"]
    testdir = '/'.join(traindir.split('/')[:-1] + ['/test'])
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    import shutil
    files = glob.glob(traindir + '/*')
    files = sorted_alphanumeric(files)
    if os.path.isfile(files[0]):  # don't do anything if directory
        shutil.move(files[0], testdir)




def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

if __name__ == '__main__':
    main()