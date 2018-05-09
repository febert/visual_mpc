from python_visual_mpc.visual_mpc_core.infrastructure.run_sim import Sim
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
import cv2


def perform_benchmark(conf = None, iex=-1, gpu_id=None):
    """
    :param conf:
    :param iex:  if not -1 use only rollout this example
    :param gpu_id:
    :return:
    """
    cem_exp_dir = '/'.join(str.split(python_vmpc_path, '/')[:-2])  + '/experiments/cem_exp'

    if conf != None:
        benchmark_name = 'parallel'
        ngpu = 1
        bench_dir = conf['current_dir']
    else:
        parser = argparse.ArgumentParser(description='Run benchmarks')
        parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
        parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
        parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
        args = parser.parse_args()

        benchmark_name = args.benchmark
        gpu_id = args.gpu_id
        ngpu = args.ngpu

        # load specific agent settings for benchmark:
        bench_dir = cem_exp_dir + '/benchmarks/' + benchmark_name
        if not os.path.exists(bench_dir):
            print('performing goal image benchmark ...')
            bench_dir = cem_exp_dir + '/benchmarks_goalimage/' + benchmark_name
            if not os.path.exists(bench_dir):
                raise ValueError('benchmark directory does not exist')

        loader = importlib.machinery.SourceFileLoader('mod_hyper', bench_dir + '/mod_hyper.py')
        spec = importlib.util.spec_from_loader(loader.name, loader)
        conf = importlib.util.module_from_spec(spec)
        loader.exec_module(conf)

        conf = conf.config

    if 'RESULT_DIR' in os.environ:
        result_dir = os.environ['RESULT_DIR']
    elif 'EXPERIMENT_DIR' in os.environ:
        subpath = conf['current_dir'].partition('experiments')[2]
        result_dir = os.path.join(os.environ['EXPERIMENT_DIR'] + subpath)
    else: result_dir = bench_dir
    print('result dir {}'.format(result_dir))

    conf['agent']['skip_first'] = 10

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
        traj = conf['start_index']
        nruns = conf['end_index']
        print('started worker going from ind {} to in {}'.format(conf['start_index'], conf['end_index']))
    else:
        traj = iex
        nruns = iex

    scores_l = []
    anglecost_l = []
    improvment_l = []
    initial_dist_l = []

    if 'sourcetags' in conf:  # load data per trajectory
        if 'VMPC_DATA_DIR' in os.environ:
            datapath = conf['source_basedirs'][0].partition('pushing_data')[2]
            conf['source_basedirs'] = [os.environ['VMPC_DATA_DIR'] + datapath]
        traj_names = make_traj_name_list({'source_basedirs': conf['source_basedirs'],
                                                  'ngroup': conf['ngroup']}, shuffle=False)

    result_file = result_dir + '/results_{}to{}.txt'.format(conf['start_index'], conf['end_index'])
    scores_pkl_file = result_dir + '/scores_{}to{}.pkl'.format(conf['start_index'], conf['end_index'])
    if os.path.isfile(result_dir + '/result_file'):
        raise ValueError("the file {} already exists!!".format(result_file))

    while traj <= nruns:
        if 'sourcetags' in conf:  # load data per trajectory
            dict = read_trajectory(conf, traj_names[traj])
            sim.agent.start_conf = dict

        print('run number ', traj)
        print('loading done')

        print('-------------------------------------------------------------------')
        print('run number ', traj)
        print('-------------------------------------------------------------------')

        record_dir = result_dir + '/verbose/traj{0}'.format(traj)
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        sim.agent._hyperparams['record'] = record_dir
        sim.reset_policy()

        sim._take_sample(traj)

        scores_l.append(sim.agent.final_poscost)
        anglecost_l.append(sim.agent.final_anglecost)
        improvment_l.append(sim.agent.improvement)
        initial_dist_l.append(sim.agent.initial_poscost)

        print('improvement of traj{},{}'.format(traj, improvment_l[-1]))
        traj +=1 #increment trajectories every step!

        score = np.array(scores_l)
        anglecost = np.array(anglecost_l)
        improvement = np.array(improvment_l)
        initial_dist = np.array(initial_dist_l)
        sorted_ind = improvement.argsort()[::-1]

        pickle.dump({'improvement':improvement, 'scores':score, 'anglecost':anglecost}, open(scores_pkl_file, 'wb'))

        mean_imp = np.mean(improvement)
        med_imp = np.median(improvement)
        mean_dist = np.mean(score)
        med_dist = np.median(score)

        f = open(result_file, 'w')
        f.write('experiment name: ' + benchmark_name + '\n')
        f.write('overall best pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[0]], sorted_ind[0]))
        f.write('overall worst pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[-1]], sorted_ind[-1]))
        f.write('average pos improvemnt: {0}\n'.format(mean_imp))
        f.write('median pos improvement {}'.format(med_imp))
        f.write('standard deviation of population {0}\n'.format(np.std(improvement)))
        f.write('standard error of the mean (SEM) {0}\n'.format(np.std(improvement)/np.sqrt(improvement.shape[0])))
        f.write('---\n')
        f.write('average pos score: {0}\n'.format(mean_dist))
        f.write('median pos score {}'.format(med_dist))
        f.write('standard deviation of population {0}\n'.format(np.std(score)))
        f.write('standard error of the mean (SEM) {0}\n'.format(np.std(score)/np.sqrt(score.shape[0])))
        f.write('---\n')
        f.write('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))
        f.write('---\n')
        f.write('average initial dist: {0}\n'.format(np.mean(initial_dist)))
        f.write('median initial dist: {0}\n'.format(np.median(initial_dist)))
        f.write('---\n')
        f.write('average angle cost: {0}\n'.format(np.mean(anglecost)))
        f.write('----------------------\n')
        f.write('traj: improv, score, anglecost, rank\n')
        f.write('----------------------\n')

        for n, t in enumerate(range(conf['start_index'], traj)):
            f.write('{}: {}, {}, {}, :{}\n'.format(t, improvement[n], score[n], anglecost[n], np.where(sorted_ind == n)[0][0]))
        f.close()

    print('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))

if __name__ == '__main__':
    perform_benchmark()