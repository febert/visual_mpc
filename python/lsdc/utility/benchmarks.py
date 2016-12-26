from lsdc import lsdc_main_mod
from lsdc.lsdc_main_mod import LSDCMain
import argparse
import imp
import os
import numpy as np
import copy
import random



def main():
    from lsdc import __file__ as lsdc_filepath
    lsdc_dir = '/'.join(str.split(lsdc_filepath, '/')[:-3])
    cem_exp_dir = lsdc_dir + '/experiments/cem_exp'
    hyperparams = imp.load_source('hyperparams', cem_exp_dir + '/benchmarks/base_hyperparams.py')

    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
    args = parser.parse_args()
    benchmark_name = args.benchmark

    conf = hyperparams.config
    # load specific agent settings for benchmark:
    bench_dir = cem_exp_dir + '/benchmarks/' + benchmark_name
    bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
    conf['policy'].update(bench_conf.policy)

    if hasattr(bench_conf, 'agent'):
        conf['agent'].update(bench_conf.agent)

    conf['agent']['skip_first'] = 10

    print '-------------------------------------------------------------------'
    print 'name of algorithm setting: ' + benchmark_name
    print 'agent settings'
    for key in conf['agent'].keys():
        print key, ': ', conf['agent'][key]
    print '-------'
    print 'policy settings'
    for key in conf['policy'].keys():
        print key, ': ', conf['policy'][key]
    print '-------------------------------------------------------------------'


    # sample intial conditions and goalpoints

    nruns = 50

    traj = 0
    n_reseed = 3
    n_conf = -1

    scores = np.empty(nruns)
    lsdc = LSDCMain(conf)

    while traj < nruns:

        alpha = np.random.uniform(0,360.0)
        ob_pos = np.random.uniform(-0.4,0.4,2)
        goalpoint = np.random.uniform(-0.4,0.4,2)

        n_conf += 1

        lsdc.agent._hyperparams['x0'] = np.array([0., 0., 0., 0.,
                    ob_pos[0], ob_pos[1], 0., np.cos(alpha/2), 0, 0, np.sin(alpha/2)  #object pose (x,y,z, quat)
                     ])
        lsdc.agent._hyperparams['goal_point'] = goalpoint

        for j in range(n_reseed):
            if traj > nruns -1:
                break

            seed = traj
            random.seed(seed)
            np.random.seed(seed)
            print '-------------------------------------------------------------------'
            print 'run number ', traj
            print 'configuration No. ', n_conf
            print 'using random seed', seed
            print '-------------------------------------------------------------------'

            lsdc.agent._hyperparams['record'] = bench_dir + '/videos/traj{0}_conf{1}'.format(traj, n_conf)

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

            lsdc.agent.sample(lsdc.policy)

            scores[traj] = lsdc.agent.final_score
            print scores[traj]

            traj +=1

    sorted_ind = scores.argsort()
    print 'overall best score: {0} of traj {1}'.format(scores[sorted_ind[0]], sorted_ind[0])
    print 'overall worst score: {0} of traj {1}'.format(scores[sorted_ind[-1]], sorted_ind[-1])
    print 'overall average score:', np.sum(scores)/scores.shape
    print 'standard deviation {0}\n'.format(np.sqrt(np.var(scores)))

    f = open(bench_dir +'/results', 'w')
    f.write('experiment name: ' + benchmark_name + '\n')
    f.write('overall best score: {0} of traj {1}\n'.format(scores[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst score: {0} of traj {1}\n'.format(scores[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average score: {0}\n'.format(np.sum(scores) / scores.shape))
    f.write('standard deviation {0}\n'.format(np.sqrt(np.var(scores))))
    f.write('----------------------\n')
    f.write('traj: score, rank\n')
    f.write('----------------------\n')
    for traj in range(nruns):
        f.write('{0}: {1}, {2}\n'.format(traj,scores[traj], np.where(sorted_ind == traj)[0][0]))
    f.close()

if __name__ == '__main__':
    main()