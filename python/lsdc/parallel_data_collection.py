from multiprocessing import Pool
import argparse
import imp
import os
from lsdc.lsdc_main_mod import LSDCMain
import copy
import random
import numpy as np
import shutil

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


def main():
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--parallel', type=str, help='use multiple threads or not', default=True)
    args = parser.parse_args()
    exp_name = args.experiment
    parallel= args.parallel


    n_worker = 10
    print 'using ', n_worker,' workers'
    if parallel == 'True':
        parallel = True
    if parallel == 'False':
        parallel = False
    print 'parallel ', bool(parallel)

    from lsdc import __file__ as lsdc_filepath
    lsdc_filepath = os.path.abspath(lsdc_filepath)
    lsdc_dir = '/'.join(str.split(lsdc_filepath, '/')[:-3]) + '/'
    data_coll_dir = lsdc_dir + 'pushing_data/' + exp_name + '/'
    hyperparams_file = data_coll_dir + 'hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    n_traj = 60000
    traj_per_worker = int(n_traj / np.float32(n_worker))
    start_idx = [traj_per_worker * i for i in range(n_worker)]
    end_idx =  [traj_per_worker * (i+1)-1 for i in range(n_worker)]

    conflist = []
    for i in range(n_worker):
        modconf = copy.deepcopy(hyperparams.config)
        modconf['start_index'] = start_idx[i]
        modconf['end_index'] = end_idx[i]
        conflist.append(modconf)

    if parallel:
        p = Pool(n_worker)
        p.map(worker, conflist)
    else:
        worker(conflist[0])

    import pdb;
    pdb.set_trace()
    file = hyperparams['data_files_dir']+ '/traj_0_to_255.tfrecords'
    dest_file = '/'.join(str.split(hyperparams['data_files_dir'], '/')[:-1]) + '/test/traj_0_to_255.tfrecords'
    shutil.move(file, dest_file)

if __name__ == '__main__':
    main()