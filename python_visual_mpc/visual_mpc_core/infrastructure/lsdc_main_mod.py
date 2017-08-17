""" This file defines the main object that runs experiments. """

import matplotlib as mpl
# mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time

# Add lsdc/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# from lsdc.gui.gps_training_gui import GPSTrainingGUI
from python_visual_mpc.video_prediction.setup_predictor import setup_predictor
from python_visual_mpc.visual_mpc_core.infrastructure.utility import *

from datetime import datetime
import pdb

import random
import numpy as np
import matplotlib.pyplot as plt

class LSDCMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False, gpu_id= 0, ngpu= None):


        self._quit_on_end = quit_on_end
        self._hyperparams = config
        self._conditions = config['common']['conditions']

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.agentparams = config['agent']

        if 'netconf' in config['policy']:
            params = imp.load_source('params', config['policy']['netconf'])
            netconf = params.configuration

        if 'usenet' in config['policy']:
            if config['policy']['usenet']:
                if 'setup_predictor' in netconf:
                    if ngpu ==None:
                        self.predictor = netconf['setup_predictor'](netconf, gpu_id)
                    else:
                        self.predictor = netconf['setup_predictor'](netconf, gpu_id, ngpu)
                else:
                    self.predictor = setup_predictor(netconf, gpu_id)
                self.policy = config['policy']['type'](config['agent'], config['policy'], self.predictor)
            else:
                self.policy = config['policy']['type'](config['agent'], config['policy'])
        else:
            self.policy = config['policy']['type'](config['agent'], config['policy'])

        self.trajectory_list = []
        self.im_score_list = []

        try:
            os.remove(self._hyperparams['agent']['image_dir'])
        except:
            pass

    def run(self):

        for i in range(self._hyperparams['start_index'],self._hyperparams['end_index']):
            self._take_sample(i)


    def _take_sample(self, sample_index):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            sample_index: Sample index.
        Returns: None
        """

        start = datetime.now()

        traj = self.agent.sample(self.policy)

        if self._hyperparams['save_data']:
            self.save_data(traj, sample_index)

        end = datetime.now()
        print 'time elapsed for one trajectory sim', end-start


    def save_data(self, traj, sample_index):
        """
        saves all the images of a sample-trajectory in a separate dataset within the same hdf5-file
        Args:
            image_data: the numpy structure
            sample_index: sample number
        """
        
        traj = copy.deepcopy(traj)
        self.trajectory_list.append(traj)
        if 'traj_per_file' in self._hyperparams:
            traj_per_file = self._hyperparams['traj_per_file']
        else:
            traj_per_file = 256
        print 'traj_per_file', traj_per_file
        if len(self.trajectory_list) == traj_per_file:
            filename = 'traj_{0}_to_{1}'\
                .format(sample_index - traj_per_file + 1, sample_index)

            from utility.save_tf_record import save_tf_record
            save_tf_record(self._data_files_dir, filename, self.trajectory_list, self.agentparams)

            self.trajectory_list = []


def main():

    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')

    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume

    from python_visual_mpc import __file__ as python_vmpc_path
    exp_dir = '/'.join(str.split(python_vmpc_path + '/experiments', '/')[:-1])
    hyperparams_file = exp_dir + '/' + exp_name + '/hyperparams.py'

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/lsdc/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))



    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    gps = LSDCMain(hyperparams.config, args.quit)

    if hyperparams.config['gui_on']:
        run_gps = threading.Thread(
            target=lambda: gps.run(itr_load=resume_training_itr)
        )
        run_gps.daemon = True
        run_gps.start()

        plt.ioff()
        # plt.show()
    else:
        gps.run(itr_load=resume_training_itr)

if __name__ == "__main__":
    main()
