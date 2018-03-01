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
import cPickle
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
    parser.add_argument('--gpu_id', type=int, help='the starting gpu_id', default=0)

    args = parser.parse_args()
    exp_name = args.experiment
    gpu_id = args.gpu_id

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

    if os.path.isfile(hyperparams_file):  # if not found in data_coll_dir
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


    if do_benchmark:
        try:
            combine_scores(hyperparams['current_dir'], start_idx, end_idx, )
        except:
            pass

    conflist = []

    hyperparams['agent']['data_save_dir'] = os.path.join(os.environ['VMPC_DATA_DIR'], hyperparams['agent']['data_save_dir'])  # directory where to save trajectories

    if 'gen_xml' in hyperparams['agent']: #remove old auto-generated xml files
        os.system("rm {}".format('/'.join(str.split(hyperparams['agent']['filename'], '/')[:-1]) + '/auto_gen/*'))


    if do_benchmark:
        use_worker = bench_worker
    else: use_worker = worker

    use_ray = False  # ray can cause black images!!
    if use_ray:
        ray.init()
        id_list = []
        for i in range(n_worker):
            modconf = copy.deepcopy(hyperparams)
            modconf['start_index'] = start_idx[i]
            modconf['end_index'] = end_idx[i]
            modconf['gpu_id'] = i + gpu_id
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

    if do_benchmark:
        combine_scores(hyperparams['current_dir'], start_idx, end_idx, exp_name)

    traindir = modconf['agent']["data_save_dir"]
    testdir = '/'.join(traindir.split('/')[:-1] + ['/test'])
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    import shutil
    files = glob.glob(traindir + '/*')
    files = sorted_alphanumeric(files)
    if os.path.isfile(files[0]): #don't do anything if directory
        shutil.move(files[0], testdir)


def combine_scores(dir, start_idx, end_idx, exp_name):

    full_scores = []
    full_anglecost = []
    for st_ind, end_ind in zip(start_idx, end_idx):
        filename = dir + '/scores_{}to{}.pkl'.format(st_ind, end_ind)
        dict_ = cPickle.load(open(filename, "rb"))

        full_scores.append(dict_['scores'])
        full_anglecost.append(dict_['anglecost'])

    scores = np.concatenate(full_scores, axis=0)
    sorted_ind = scores.argsort()
    anglecost = np.concatenate(full_anglecost, axis=0)

    f = open(dir + '/results_all.txt', 'w')
    f.write('experiment name: ' + exp_name + '\n')
    f.write('overall best pos score: {0} of traj {1}\n'.format(scores[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst pos score: {0} of traj {1}\n'.format(scores[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average pos score: {0}\n'.format(np.mean(scores)))
    f.write('standard deviation of population {0}\n'.format(np.std(scores)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(scores) / np.sqrt(scores.shape[0])))
    f.write('---\n')
    f.write('average angle cost: {0}\n'.format(np.mean(anglecost)))
    f.write('----------------------\n')
    f.write('traj: score, anglecost, rank\n')
    f.write('----------------------\n')
    for n, t in enumerate(range(0, end_idx[-1]+1)):
        f.write('{0}: {1}, {2}, :{3}\n'.format(t, scores[n], anglecost[n], np.where(sorted_ind == n)[0][0]))
    f.close()

def sorted_alphanumeric(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

if __name__ == '__main__':
    main()

    # n_worker = 1
    # n_traj = 51
    # dir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/experiments/cem_exp/benchmarks/appflow'
    #
    # traj_per_worker = int(n_traj / np.float32(n_worker))
    # start_idx = [traj_per_worker * i for i in range(n_worker)]
    # end_idx = [traj_per_worker * (i + 1) - 1 for i in range(n_worker)]
    #
    # combine_scores(dir, start_idx, end_idx, 'test')