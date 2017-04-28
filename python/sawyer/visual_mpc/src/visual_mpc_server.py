#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image as Image_msg
import os
import shutil
import socket
import thread
import numpy as np
import pdb
from PIL import Image
import cPickle
import imp
import argparse

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
from lsdc.utility.trajectory import Trajectory
from lsdc import __file__ as lsdc_filepath
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg


from berkeley_sawyer.srv import *

class Visual_MPC_Server(object):
    def __init__(self):
        """
        Similar functionality to mjc_agent and lsdc_main_mod, calling the policy
        """

        # if it is an auxiliary node advertise services
        rospy.init_node('visual_mpc_server')
        rospy.loginfo("init visual mpc server")

        # initializing the servives:
        rospy.Service('get_action', get_action, self.get_action_handler)
        rospy.Service('get_action', init_traj_visualmpc, self.init_traj_visualmpc_handler)

        self.traj = Trajectory(self._hyperparams)

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

        netconf = imp.load_source('params', conf['policy']['netconf']).configuration

        self.predictor = netconf['setup_predictor'](netconf, gpu_id, ngpu)
        self.cem_controller = CEM_controller(conf['agent'], conf['policy'], self.predictor)

        self.t = None

        ###
        rospy.spin()



    def init_traj_visualmpc_handler(self, req):
        self.igrp = req.grp
        self.i_traj = req.itr
        self.t = 0

    def get_action_handler(self, req):
        self.traj.X_Xdot_full[self.t,:] = np.concatenate(req.x, req.xdot)

        self.traj._sample_images[self.t] = req.image

        mj_U, pos, ind, targets = self.cem_controller.act(self.traj.X_full,
                                                          self.traj.Xdot_full,
                                                          self.traj._sample_images,
                                                          self.t)
        self.traj.U[self.t, :] = mj_U

        self.t += 1






if __name__ ==  '__main__':
    print 'started'
    Visual_MPC_Server()
