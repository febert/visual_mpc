from lsdc import lsdc_main_mod
from lsdc.lsdc_main_mod import LSDCMain
import argparse
import imp
import os
import numpy as np
import pdb
import copy
import random
import cPickle
from PIL import Image
from video_prediction.correction.setup_corrector import setup_corrector


def main():
    from lsdc import __file__ as lsdc_filepath
    lsdc_dir = '/'.join(str.split(lsdc_filepath, '/')[:-3])
    cem_exp_dir = lsdc_dir + '/experiments/cem_exp'
    hyperparams = imp.load_source('hyperparams', cem_exp_dir + '/base_hyperparams.py')

    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
    parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
    parser.add_argument('--ngpu', type=int, default=None, help='number of gpus to use')
    args = parser.parse_args()

    benchmark_name = args.benchmark
    gpu_id = args.gpu_id
    ngpu = args.ngpu

    conf = hyperparams.config
    # load specific agent settings for benchmark:

    bench_dir = cem_exp_dir + '/benchmarks/' + benchmark_name
    if not os.path.exists(bench_dir):
        print 'performing goal image benchmark ...'
        bench_dir = cem_exp_dir + '/benchmarks_goalimage/' + benchmark_name
        goalimg_save_dir = cem_exp_dir + '/benchmarks_goalimage/' + benchmark_name + '/goalimage'
        if not os.path.exists(bench_dir):
            raise ValueError('benchmark directory does not exist')

    bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
    conf['policy'].update(bench_conf.policy)

    if hasattr(bench_conf, 'agent'):
        conf['agent'].update(bench_conf.agent)

    if hasattr(bench_conf, 'config'):
        conf.update(bench_conf.config)

    if hasattr(bench_conf, 'common'):
        conf['common'].update(bench_conf.common)


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

    if 'verbose' in conf['policy']:
        print 'verbose mode!! just running 1 configuration'
        nruns = 1


    traj = 0
    if 'n_reseed' in conf['policy']:
        n_reseed = conf['policy']['n_reseed']
    else:
        n_reseed = 3
    i_conf = 0


    anglecost = []
    lsdc = LSDCMain(conf, gpu_id= gpu_id, ngpu= ngpu)

    if 'start_confs' not in conf['agent']:
        benchconfiguration = cPickle.load(open('python/lsdc/utility/benchmarkconfigs', "rb"))
    else:
        benchconfiguration = cPickle.load(open(conf['agent']['start_confs'], "rb"))

    nruns = len(benchconfiguration['initialpos'])*n_reseed  # 60 in standard benchmark

    scores = np.zeros(nruns)

    if 'load_goal_image' in conf['policy']:
        goalimg_load_dir = cem_exp_dir +'/benchmarks_goalimage/' +\
                           conf['policy']['load_goal_image'] +'/goalimage'

        if 'ballinvar' in conf['policy']:
            goalimg_load_dir = cem_exp_dir + '/benchmarks_goalimage/' + \
                               conf['policy']['load_goal_image'] + '/goalimage_var_ballpos'


    goalpoints = benchconfiguration['goalpoints']
    initialposes = benchconfiguration['initialpos']

    while traj < nruns:

        lsdc.agent._hyperparams['x0'] = initialposes[i_conf]
        if 'use_goalimage' not in conf['policy']:
            lsdc.agent._hyperparams['goal_point'] = goalpoints[i_conf]

        for j in range(n_reseed):
            if traj > nruns -1:
                break

            seed = traj
            random.seed(seed)
            np.random.seed(seed)
            print '-------------------------------------------------------------------'
            print 'run number ', traj
            print 'configuration No. ', i_conf
            print 'using random seed', seed
            print '-------------------------------------------------------------------'

            lsdc.agent._hyperparams['record'] = bench_dir + '/videos/traj{0}_conf{1}'.format(traj, i_conf)
            if 'save_goal_image' in conf['agent']:
                lsdc.agent._hyperparams['save_goal_image'] = goalimg_save_dir + '/goalimg{0}_conf{1}'.format(traj, i_conf)
            if 'use_goalimage' in conf['policy']:
                conf['policy']['use_goalimage'] = goalimg_load_dir + '/goalimg{0}_conf{1}.pkl'.format(traj, i_conf)
                goal_dict = cPickle.load(open(conf['policy']['use_goalimage'], "rb"))
                lsdc.agent._hyperparams['goal_object_pose'] = goal_dict['goal_object_pose']

                # if 'pixelmover' in conf['policy'] or 'random_baseline' in conf['agent']:
                lsdc.agent._hyperparams['goal_point'] = goal_dict['goal_object_pose'][0][:2]

            if 'usenet' in conf['policy']:
                if conf['policy']['usenet']:
                    lsdc.policy = conf['policy']['type'](lsdc.agent._hyperparams,
                                                         conf['policy'], lsdc.predictor)
                else:
                    lsdc.policy = conf['policy']['type'](lsdc.agent._hyperparams,
                                                         conf['policy'])
            else:
                lsdc.policy = conf['policy']['type'](lsdc.agent._hyperparams,
                                                     conf['policy'])

            if 'correctorconf' in conf['policy']:
                lsdc.policy.corrector = lsdc.corrector

            lsdc.policy.policyparams['rec_distrib'] =  bench_dir + '/videos_distrib/traj{0}_conf{1}'.format(traj, i_conf)
            lsdc._take_sample(traj)
            scores[traj] = lsdc.agent.final_poscost

            if 'use_goalimage' in conf['agent']:
                anglecost.append(lsdc.agent.final_anglecost)

            print 'score of traj', traj, ':', scores[traj]

            traj +=1 #increment trajectories every step!

        i_conf += 1 #increment configurations every three steps!

        rel_scores = scores[:traj]
        sorted_ind = rel_scores.argsort()
        f = open(bench_dir + '/results', 'w')
        f.write('experiment name: ' + benchmark_name + '\n')
        f.write('overall best pos score: {0} of traj {1}\n'.format(rel_scores[sorted_ind[0]], sorted_ind[0]))
        f.write('overall worst pos score: {0} of traj {1}\n'.format(rel_scores[sorted_ind[-1]], sorted_ind[-1]))
        f.write('average pos score: {0}\n'.format(np.sum(rel_scores) / traj))
        f.write('standard deviation {0}\n'.format(np.sqrt(np.var(rel_scores))))
        f.write('----------------------\n')
        f.write('traj: score, anglecost, rank\n')
        f.write('----------------------\n')
        for t in range(traj):
            if 'use_goalimage' in conf['agent']:
                f.write('{0}: {1}, {2}, :{3}\n'.format(t, rel_scores[t], anglecost[t], np.where(sorted_ind == t)[0][0]))
            else:
                f.write('{0}: {1}, :{2}\n'.format(t, rel_scores[t], np.where(sorted_ind == t)[0][0]))
        f.close()

    print 'overall best score: {0} of traj {1}'.format(scores[sorted_ind[0]], sorted_ind[0])
    print 'overall worst score: {0} of traj {1}'.format(scores[sorted_ind[-1]], sorted_ind[-1])
    print 'overall average score:', np.sum(scores)/scores.shape
    print 'standard deviation {0}\n'.format(np.sqrt(np.var(scores)))

if __name__ == '__main__':
    main()