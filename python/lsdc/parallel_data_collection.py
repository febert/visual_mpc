from multiprocessing import Pool
import argparse
import imp
import os
from lsdc.lsdc_main_mod import LSDCMain
import copy
import random
import numpy as np

def worker(conf):
    print 'started process with PID:', os.getpid()
    print 'making trajectories {0} to {1}'.format(
        conf['start_index'],
        conf['end_index'],
    )

    random.seed(None)
    np.random.seed(None)

    lsdc = LSDCMain(conf)
    lsdc.run()


n_worker = 10

def main():
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str, help='experiment name')
    args = parser.parse_args()
    exp_name = args.experiment

    from lsdc import __file__ as lsdc_filepath
    lsdc_filepath = os.path.abspath(lsdc_filepath)
    lsdc_dir = '/'.join(str.split(lsdc_filepath, '/')[:-4]) + '/'
    data_coll_dir = lsdc_dir + 'pushing_data/' + exp_name + '/'
    hyperparams_file = data_coll_dir + 'hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    n_traj = 60000
    traj_per_worker = int(n_traj / n_worker)
    start_idx = [traj_per_worker * i for i in range(n_worker)]
    end_idx =  [traj_per_worker * (i+1)-1 for i in range(n_worker)]

    conflist = []
    for i in range(n_worker):
        modconf = copy.deepcopy(hyperparams.config)
        modconf['start_index'] = start_idx[i]
        modconf['end_index'] = end_idx[i]
        conflist.append(modconf)

    parallel = True
    if parallel:
        p = Pool(n_worker)
        p.map(worker, conflist)
    else:
        worker(conflist[0])


if __name__ == '__main__':


    main()