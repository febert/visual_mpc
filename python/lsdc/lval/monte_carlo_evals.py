from lsdc import lsdc_main_mod
from lsdc.lsdc_main_mod import LSDCMain
import argparse
import imp
import os
import numpy as np
import copy
import random
import cPickle
from video_prediction.correction.setup_corrector import setup_corrector

from multiprocessing import Pool, Process, Queue, Manager

def worker(conf):

    lsdc = LSDCMain(conf, gpu_id=conf['gpu_id'], ngpu=1)

    confs = cPickle.load(open('lval_configs_10000', "rb"))
    goalpoints = confs['goalpoints']
    initialposes = confs['initialpos']


    print 'started process with PID: {0} using gpu_id: {1}'.format(os.getpid(),conf['gpu_id'])
    print 'making trajectories {0} to {1}'.format(
        conf['start_index'],
        conf['end_index'],
    )

    for i_conf in range(conf['start_index'], conf['end_index']):

        lsdc.agent._hyperparams['x0'] = initialposes[i_conf]
        lsdc.agent._hyperparams['goal_point'] = goalpoints[i_conf]


        seed = i_conf
        random.seed(seed)
        np.random.seed(seed)
        print '-------------------------------------------------------------------'
        print 'configuration No. ', i_conf
        print 'using random seed', seed
        print '-------------------------------------------------------------------'

        _dir = conf['policy']['current_dir']
        lsdc.agent._hyperparams['record'] = _dir + '/videos/conf{}'.format(i_conf)

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

        trajectory = lsdc.agent.sample(lsdc.policy)

        desig_pos = trajectory.desig_pos[0]
        init_state = trajectory.X_full[0]
        score = lsdc.agent.final_score
        print 'score of conf', i_conf, ':', score

        lsdc.save_data_lval(trajectory, score, goalpoints[i_conf], desig_pos, init_state, i_conf)
        q = conf['queue']
        q.put([i_conf, score])

def statistics_writer(q, conf, setting_name):
    print 'started statistics writer'
    index_score_list = []

    while len(index_score_list)< conf['nruns']:
        index_score_list.append(q.get())
        indexes = np.array([el[0] for el in index_score_list])
        scores = np.array([el[1] for el in index_score_list])

        _dir = conf['policy']['current_dir']

        sorted_ind = scores.argsort()
        f = open(_dir + '/results', 'w')
        f.write('experiment name: ' + setting_name + '\n')
        f.write('number of collected trajectories: {}\n'.format(len(index_score_list)))
        f.write('overall best score: {0} of traj {1}\n'.format(scores[sorted_ind[0]], sorted_ind[0]))
        f.write('overall worst score: {0} of traj {1}\n'.format(scores[sorted_ind[-1]], sorted_ind[-1]))
        f.write('average score: {0}\n'.format(np.sum(scores) / len(index_score_list)))
        f.write('standard deviation {0}\n'.format(np.sqrt(np.var(scores))))
        f.write('----------------------\n')
        f.write('traj: score, rank\n')
        f.write('----------------------\n')

        for idx, item in enumerate(index_score_list):
            f.write('{0}: {1}, {2}\n'.format(idx, scores[idx], np.where(sorted_ind == idx)[0][0]))
        f.close()

    print 'overall best score: {0} of traj {1}'.format(scores[sorted_ind[0]], sorted_ind[0])
    print 'overall worst score: {0} of traj {1}'.format(scores[sorted_ind[-1]], sorted_ind[-1])
    print 'overall average score:', np.sum(scores) / scores.shape
    print 'standard deviation {0}\n'.format(np.sqrt(np.var(scores)))

def main():
    from lsdc import __file__ as lsdc_filepath
    lsdc_dir = '/'.join(str.split(lsdc_filepath, '/')[:-3])
    lval_exp_dir = lsdc_dir + '/experiments/val_exp'

    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('lval_setting', type=str, help='the name of the folder with agent setting for the learning the value function')
    # parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
    parser.add_argument('--parallel', type=str, help='use multiple threads or not', default=True)
    args = parser.parse_args()
    setting_name = args.lval_setting
    # gpu_id = args.gpu_id
    parallel = args.parallel

    n_worker = 4
    print 'using ', n_worker, ' workers'
    if parallel == 'True':
        parallel = True
    if parallel == 'False':
        parallel = False
        n_worker = 1


    # load specific agent settings for benchmark:
    _dir = lval_exp_dir + '/' + setting_name
    hyperparams = imp.load_source('mod_hyper', _dir + '/hyperparams.py')
    conf = hyperparams.config

    print '-------------------------------------------------------------------'
    print 'name of algorithm setting: ' + setting_name
    print 'agent settings'
    for key in conf['agent'].keys():
        print key, ': ', conf['agent'][key]
    print '-------'
    print 'policy settings'
    for key in conf['policy'].keys():
        print key, ': ', conf['policy'][key]
    print '-------------------------------------------------------------------'

    # sample intial conditions and goalpoints
    nruns = conf['nruns']

    traj_per_worker = int(nruns / np.float32(n_worker))
    start_idx = [traj_per_worker * i for i in range(n_worker)]
    end_idx = [traj_per_worker * (i + 1) - 1 for i in range(n_worker)]

    m = Manager()
    queue = m.Queue()

    conflist = []
    for i in range(n_worker):
        modconf = copy.deepcopy(conf)
        modconf['nruns'] = traj_per_worker
        modconf['start_index'] = start_idx[i]
        modconf['end_index'] = end_idx[i]
        modconf['gpu_id'] = i
        modconf['queue'] = queue
        conflist.append(modconf)

    p_statistics_writer = Process(target=statistics_writer, args=(queue, conf, setting_name))
    p_statistics_writer.daemon = True
    p_statistics_writer.start()

    if parallel:
        p = Pool(n_worker)
        p.map(worker, conflist)
    else:
        worker(conflist[0])


if __name__ == '__main__':
    main()