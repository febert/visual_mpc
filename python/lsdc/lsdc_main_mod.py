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

import h5py

# Add lsdc/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# from lsdc.gui.gps_training_gui import GPSTrainingGUI
from lsdc.utility.data_logger import DataLogger
from lsdc.sample.sample_list import SampleList
from lsdc.algorithm.policy.random_policy import Randompolicy
from lsdc.algorithm.policy.random_impedance_point import Random_impedance_point
from video_prediction.setup_predictor import setup_predictor
from video_prediction.correction.setup_corrector import setup_corrector
from lsdc.utility.save_tf_record import *

from datetime import datetime
import pdb

import random
import numpy as np
import matplotlib.pyplot as plt

import numpy as np


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

        if 'rewardnetconf' in config['policy']:
            params = imp.load_source('params', config['policy']['rewardnetconf'])
            rewardnetconf = params.configuration
            config['policy']['rewardnet_func'] = rewardnetconf['setup_rewardnet'](rewardnetconf, gpu_id)

        if 'correctorconf' in config['policy']:
            self.corrector = setup_corrector(config['policy']['correctorconf'])
            self.policy.corrector = self.corrector

        self.trajectory_list = []
        self.im_score_list = []

        try:
            os.remove(self._hyperparams['agent']['image_dir'])
        except:
            pass

    def run(self):

        for i in range(self._hyperparams['start_index'],self._hyperparams['end_index']):
            self._take_sample(i)


    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                if self.algorithm.cur[0].pol_info:
                    pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                        ('pol_sample_itr_%02d.pkl' % itr_load))
                else:
                    pol_sample_lists = None
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

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

        traj = self.agent.sample(
            self.policy)

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

            if 'store_whole_pred' in self.agentparams:
                save_tf_record_gtruthpred(self._data_files_dir,filename, self.trajectory_list, self.agentparams)
            else:
                save_tf_record(self._data_files_dir, filename, self.trajectory_list, self.agentparams)


            self.trajectory_list = []

    def save_data_lval(self, traj, score, goalpos, desig_pos, init_state, sample_index):
        """
        save starting image, task parameters (configuration, goalposition) and return tuples in tf records
        :param traj:
        :param sample_index:
        """
        # get first image
        first_img = copy.deepcopy(traj._sample_images[0])
        self.im_score_list.append([first_img, score, goalpos, desig_pos, init_state])
        traj_per_file = 100
        print '_per_file', traj_per_file
        if len(self.im_score_list) == traj_per_file:
            filename = 'traj_{0}_to_{1}' \
                .format(sample_index - traj_per_file + 1, sample_index)
            save_tf_record_lval(self._data_files_dir, filename, self.im_score_list)
            self.im_score_list = []


    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False)
        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, traj_sample_lists):
        """
        Log data_files and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """

        if 'no_sample_logging' in self._hyperparams['common']:
            return
        # self.data_logger.pickle(
        #     self._data_files_dir + ('algorithm_itr_%02d.pkl' ),
        #     copy.copy(self.algorithm)
        # )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' ),
            copy.copy(traj_sample_lists)
        )
        # if pol_sample_lists:
        #     self.data_logger.pickle(
        #         self._data_files_dir + ('pol_sample_itr_%02d.pkl'),
        #         copy.copy(pol_sample_lists)
        #     )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end:
                # Quit automatically (for running sequential expts)
                os._exit(1)

def main():


    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    from lsdc import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

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
