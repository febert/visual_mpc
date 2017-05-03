#!/usr/bin/env python
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
import cv2
from cv_bridge import CvBridge, CvBridgeError

import socket
if socket.gethostname() == 'newton1':
    from lsdc.algorithm.policy.cem_controller_goalimage_sawyer import CEM_controller

from lsdc.utility.trajectory import Trajectory
from lsdc import __file__ as lsdc_filepath
import rospy

import rospy.numpy_msg

from visual_mpc.srv import *

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
        rospy.Service('init_traj_visualmpc', init_traj_visualmpc, self.init_traj_visualmpc_handler)


        lsdc_dir = '/'.join(str.split(lsdc_filepath, '/')[:-3])
        cem_exp_dir = lsdc_dir + '/experiments/cem_exp/benchmarks_sawyer'
        hyperparams = imp.load_source('hyperparams', cem_exp_dir + '/base_hyperparams_sawyer.py')

        parser = argparse.ArgumentParser(description='Run benchmarks')
        parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
        parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
        parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
        args = parser.parse_args()

        benchmark_name = args.benchmark
        gpu_id = args.gpu_id
        ngpu = args.ngpu

        self.conf = hyperparams.config
        self.policyparams = hyperparams.policy
        self.agentparams = hyperparams.agent
        # load specific agent settings for benchmark:

        print 'performing goal image benchmark ...'
        bench_dir = cem_exp_dir + '/' + benchmark_name
        goalimg_save_dir = cem_exp_dir + '/benchmarks_goalimage/' + benchmark_name + '/goalimage'

        if not os.path.exists(bench_dir):
            raise ValueError('benchmark directory does not exist')

        bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        if hasattr(bench_conf, 'policy'):
            self.policyparams.update(bench_conf.policy)
        if hasattr(bench_conf, 'agent'):
            self.agentparams.update(bench_conf.agent)

        netconf = imp.load_source('params', self.policyparams['netconf']).configuration
        self.predictor = netconf['setup_predictor'](netconf, gpu_id, ngpu)
        self.cem_controller = CEM_controller(self.agentparams, self.policyparams, self.predictor)
        self.t = None
        self.traj = Trajectory(self.agentparams)
        self.bridge = CvBridge()

        ###
        print 'spinning'
        rospy.spin()

    def init_traj_visualmpc_handler(self, req):
        self.igrp = req.igrp
        self.i_traj = req.itr
        self.t = 0
        self.cem_controller.goal_image = np.concatenate([
            req.goalmain,
            req.goalaux1
        ], axis=2)

        return init_traj_visualmpcResponse()

    def get_action_handler(self, req):

        self.traj.X_full[self.t, :] = req.state
        main_img = self.bridge.imgmsg_to_cv2(req.main)
        aux1_img = self.bridge.imgmsg_to_cv2(req.aux1)

        self.traj._sample_images[self.t] = np.concatenate((main_img, aux1_img), 2)

        mj_U, pos, ind, targets = self.cem_controller.act(self.traj, self.t)
        self.traj.U[self.t, :] = mj_U
        self.t += 1

        return get_actionResponse(tuple(mj_U))

if __name__ ==  '__main__':
    Visual_MPC_Server()
