from infrastructure.run_sim import Sim
import argparse
import imp
import os
import numpy as np
import pdb
import copy
import random
import cPickle
from PIL import Image
from python_visual_mpc.video_prediction.utils_vpred.online_reader import read_trajectory

from python_visual_mpc import __file__ as python_vmpc_path
from python_visual_mpc.data_preparation.gather_data import make_traj_name_list


def perform_benchmark(conf = None, gpu_id=None):
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
            print 'performing goal image benchmark ...'
            bench_dir = cem_exp_dir + '/benchmarks_goalimage/' + benchmark_name
            if not os.path.exists(bench_dir):
                raise ValueError('benchmark directory does not exist')

        conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        conf = conf.config

    conf['agent']['skip_first'] = 10

    print '-------------------------------------------------------------------'
    print 'name of algorithm setting: ' + benchmark_name
    print 'agent settings'
    for key in conf['agent'].keys():
        print key, ': ', conf['agent'][key]
    print '------------------------'
    print '------------------------'
    print 'policy settings'
    for key in conf['policy'].keys():
        print key, ': ', conf['policy'][key]
    print '-------------------------------------------------------------------'

    # sample intial conditions and goalpoints

    if 'n_reseed' in conf['policy']:
        n_reseed = conf['policy']['n_reseed']
    else:
        n_reseed = 1

    anglecost = []
    sim = Sim(conf, gpu_id= gpu_id, ngpu= ngpu)

    traj = conf['start_index']
    i_conf = conf['start_index']
    nruns = conf['end_index']
    print 'started worker going from ind {} to in {}'.format(conf['start_index'], conf['end_index'])

    # if 'verbose' in conf['policy']:
    #     print 'verbose mode!! just running 1 configuration'
    #     nruns = 1

    scores_l = []
    anglecost_l = []

    if 'sourcetags' in conf:  # load data per trajectory
        traj_names = make_traj_name_list({'source_basedirs': conf['source_basedirs'],
                                                  'ngroup': conf['ngroup']}, shuffle=False)

    result_file = bench_dir + '/results_{}to{}.txt'.format(conf['start_index'], conf['end_index'])
    scores_pkl_file = bench_dir + '/scores_{}to{}.pkl'.format(conf['start_index'], conf['end_index'])
    if os.path.isfile(bench_dir + '/result_file'):
        raise ValueError("the file {} already exists!!".format(result_file))

    while traj <= nruns:
        dict = read_trajectory(conf, traj_names[traj])
        sim.agent.load_obj_statprop = dict['obj_statprop']
        if 'reverse_action' in conf:
            init_index = -1
            goal_index = 0
        else:
            init_index = 0
            goal_index = -1

        if sim.agent._hyperparams['sdim']/2 != dict['qpos'].shape[1]:
            sim.agent._hyperparams['xpos0'] = np.concatenate([dict['qpos'][init_index], np.array([0.,0.,0.])])  #testing in setting with updown rot, while data has only xyz
        else:
            sim.agent._hyperparams['xpos0'] = dict['qpos'][init_index]
        sim.agent._hyperparams['object_pos0'] = dict['object_full_pose'][init_index]
        sim.agent.object_full_pose_t = dict['object_full_pose']
        sim.agent.goal_obj_pose = dict['object_full_pose'][goal_index]   #needed for calculating the score
        sim.agent.goal_image = dict['images'][goal_index]  # assign last image of trajectory as goalimage
        if 'goal_mask' in conf['agent']:
            sim.agent.goal_mask = dict['goal_mask'][goal_index]  # assign last image of trajectory as goalimage

        print 'run number ', traj
        print 'loading done'

        print '-------------------------------------------------------------------'
        print 'run number ', traj
        print '-------------------------------------------------------------------'

        record_dir = bench_dir + '/verbose/traj{0}'.format(traj)
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        sim.agent._hyperparams['record'] = record_dir

        # reinitilize policy between rollouts
        if 'usenet' in conf['policy']:
            if 'warp_objective' in conf['policy']:
                sim.policy = conf['policy']['type'](sim.agent._hyperparams,
                                        conf['policy'], sim.predictor, sim.goal_image_waper)
            else:
                sim.policy = conf['policy']['type'](sim.agent._hyperparams,
                                                 conf['policy'], sim.predictor)
        else:
            sim.policy = conf['policy']['type'](sim.agent._hyperparams, conf['policy'])

        sim.policy.policyparams['rec_distrib'] = bench_dir + '/videos_distrib/traj{0}'.format(traj)

        sim._take_sample(traj)

        scores_l.append(sim.agent.final_poscost)
        anglecost_l.append(sim.agent.final_anglecost)

        print 'score of traj{},{} anglecost{}'.format(traj, scores_l[-1], anglecost_l[-1])

        traj +=1 #increment trajectories every step!

        scores = np.array(scores_l)
        sorted_ind = scores.argsort()
        anglecost = np.array(anglecost_l)

        cPickle.dump({'scores':scores, 'anglecost':anglecost}, open(scores_pkl_file, 'wb'))

        f = open(result_file, 'w')
        f.write('experiment name: ' + benchmark_name + '\n')
        f.write('overall best pos score: {0} of traj {1}\n'.format(scores[sorted_ind[0]], sorted_ind[0]))
        f.write('overall worst pos score: {0} of traj {1}\n'.format(scores[sorted_ind[-1]], sorted_ind[-1]))
        f.write('average pos score: {0}\n'.format(np.mean(scores)))
        f.write('standard deviation of population {0}\n'.format(np.std(scores)))
        f.write('standard error of the mean (SEM) {0}\n'.format(np.std(scores)/np.sqrt(scores.shape[0])))
        f.write('---\n')
        f.write('average angle cost: {0}\n'.format(np.mean(anglecost)))
        f.write('----------------------\n')
        f.write('traj: score, anglecost, rank\n')
        f.write('----------------------\n')
        for n, t in enumerate(range(conf['start_index'], traj)):
            f.write('{0}: {1}, {2}, :{3}\n'.format(t, scores[n], anglecost[n], np.where(sorted_ind == n)[0][0]))
        f.close()

    print 'overall best score: {0} of traj {1}'.format(scores[sorted_ind[0]], sorted_ind[0])
    print 'overall worst score: {0} of traj {1}'.format(scores[sorted_ind[-1]], sorted_ind[-1])
    print 'overall average score:', np.sum(scores)/scores.shape
    print 'standard deviation {0}\n'.format(np.sqrt(np.var(scores)))


if __name__ == '__main__':
    perform_benchmark()