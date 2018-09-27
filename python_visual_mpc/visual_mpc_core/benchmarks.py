from python_visual_mpc.visual_mpc_core.infrastructure.sim import Sim
import argparse
import importlib.machinery
import importlib.util
import os
import numpy as np
import pdb
import copy
import random
import pickle
from PIL import Image
from python_visual_mpc.video_prediction.utils_vpred.online_reader import read_trajectory

from python_visual_mpc import __file__ as python_vmpc_path
from python_visual_mpc.data_preparation.gather_data import make_traj_name_list
from collections import OrderedDict
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
import cv2


def perform_benchmark(conf=None, iex=-1, gpu_id=None):
    """
    :param conf:
    :param iex:  if not -1 use only rollout this example
    :param gpu_id:
    :return:
    """
    result_dir = conf['result_dir']
    cem_exp_dir = '/'.join(str.split(python_vmpc_path, '/')[:-2])  + '/experiments/cem_exp'

    if conf != None:
        benchmark_name = 'parallel'
        ngpu = 1

    print('-------------------------------------------------------------------')
    print('name of algorithm setting: ' + benchmark_name)
    print('agent settings')
    for key in list(conf['agent'].keys()):
        print(key, ': ', conf['agent'][key])
    print('------------------------')
    print('------------------------')
    print('policy settings')
    for key in list(conf['policy'].keys()):
        print(key, ': ', conf['policy'][key])
    print('-------------------------------------------------------------------')

    # sample intial conditions and goalpoints

    sim = Sim(conf, gpu_id=gpu_id, ngpu=ngpu)

    if iex == -1:
        i_traj = conf['start_index']
        nruns = conf['end_index']
        print('started worker going from ind {} to in {}'.format(conf['start_index'], conf['end_index']))
    else:
        i_traj = iex
        nruns = iex

    stats_lists = OrderedDict()

    if 'sourcetags' in conf:  # load data per trajectory
        if 'VMPC_DATA_DIR' in os.environ:
            datapath = conf['source_basedirs'][0].partition('pushing_data')[2]
            conf['source_basedirs'] = [os.environ['VMPC_DATA_DIR'] + datapath]
        traj_names = make_traj_name_list({'source_basedirs': conf['source_basedirs'],
                                                  'ngroup': conf['ngroup']}, shuffle=False)

    result_file = result_dir + '/results_{}to{}.txt'.format(conf['start_index'], conf['end_index'])
    final_dist_pkl_file = result_dir + '/scores_{}to{}.pkl'.format(conf['start_index'], conf['end_index'])
    if os.path.isfile(result_dir + '/result_file'):
        raise ValueError("the file {} already exists!!".format(result_file))

    while i_traj <= nruns:

        print('run number ', i_traj)
        print('loading done')

        print('-------------------------------------------------------------------')
        print('run number ', i_traj)
        print('-------------------------------------------------------------------')

        record_dir = result_dir + '/verbose/traj{0}'.format(i_traj)
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        sim.agent._hyperparams['record'] = record_dir

        agent_data = sim.take_sample(i_traj)

        stats_data = agent_data['stats']
        stat_arrays = OrderedDict()
        for key in stats_data.keys():
            if key not in stats_lists:
                stats_lists[key] = []
            stats_lists[key].append(stats_data[key])
            stat_arrays[key] = np.array(stats_lists[key])

        i_traj +=1 #increment trajectories every step!

        pickle.dump(stat_arrays, open(final_dist_pkl_file, 'wb'))
        write_scores(conf, result_file, stat_arrays, i_traj)


def write_scores(conf, result_file, stat, i_traj=None):
    improvement = stat['improvement']

    final_dist = stat['final_dist']
    if 'initial_dist' in stat:
        initial_dist = stat['initial_dist']
    else: initial_dist = None

    if 'term_t' in stat:
        term_t = stat['term_t']

    sorted_ind = improvement.argsort()[::-1]

    if i_traj == None:
        i_traj = improvement.shape[0]

    mean_imp = np.mean(improvement)
    med_imp = np.median(improvement)
    mean_dist = np.mean(final_dist)
    med_dist = np.median(final_dist)

    if 'lifted' in stat:
        lifted = stat['lifted'].astype(np.int)
    else: lifted = np.zeros_like(improvement)

    print('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))

    f = open(result_file, 'w')
    if 'term_dist' in conf['agent']:
        tlen = conf['agent']['T']
        nsucc_frac = np.where(term_t != (tlen - 1))[0].shape[0]/ improvement.shape[0]
        f.write('percent success: {}%\n'.format(nsucc_frac * 100))
        f.write('---\n')
    if 'lifted' in stat:
        f.write('---\n')
        f.write('fraction of traj lifted: {0}\n'.format(np.mean(lifted)))
        f.write('---\n')
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(final_dist) / np.sqrt(final_dist.shape[0])))
    f.write('---\n')
    f.write('overall best pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average pos improvemnt: {0}\n'.format(mean_imp))
    f.write('median pos improvement {}'.format(med_imp))
    f.write('standard deviation of population {0}\n'.format(np.std(improvement)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(improvement) / np.sqrt(improvement.shape[0])))
    f.write('---\n')
    f.write('average pos score: {0}\n'.format(mean_dist))
    f.write('median pos score {}'.format(med_dist))
    f.write('standard deviation of population {0}\n'.format(np.std(final_dist)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(final_dist) / np.sqrt(final_dist.shape[0])))
    f.write('---\n')
    f.write('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))
    f.write('---\n')
    if initial_dist is not None:
        f.write('average initial dist: {0}\n'.format(np.mean(initial_dist)))
        f.write('median initial dist: {0}\n'.format(np.median(initial_dist)))
        f.write('----------------------\n')
    f.write('traj: improv, final_d, term_t, lifted, rank\n')
    f.write('----------------------\n')

    for n, t in enumerate(range(conf['start_index'], i_traj)):
        f.write('{}: {}, {}:{}\n'.format(t, improvement[n], final_dist[n], np.where(sorted_ind == n)[0][0]))
    f.close()


if __name__ == '__main__':
    perform_benchmark()