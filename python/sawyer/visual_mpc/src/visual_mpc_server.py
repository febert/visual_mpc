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

from video_prediction.utils_vpred.create_gif import *
import socket
# if socket.gethostname() == 'newton1':
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

        bench_dir = cem_exp_dir + '/' + benchmark_name
        goalimg_save_dir = cem_exp_dir + '/benchmarks_goalimage/' + benchmark_name + '/goalimage'

        if not os.path.exists(bench_dir):
            raise ValueError('benchmark directory does not exist')

        bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        if hasattr(bench_conf, 'policy'):
            self.policyparams.update(bench_conf.policy)
        if hasattr(bench_conf, 'agent'):
            self.agentparams.update(bench_conf.agent)

        self.netconf = imp.load_source('params', self.policyparams['netconf']).configuration
        self.predictor = self.netconf['setup_predictor'](self.netconf, gpu_id, ngpu)
        self.cem_controller = CEM_controller(self.agentparams, self.policyparams, self.predictor)
        self.t = 0
        self.traj = Trajectory(self.agentparams, self.netconf)
        self.bridge = CvBridge()
        self.initial_pix_distrib = []

        # initializing the servives:
        rospy.Service('get_action', get_action, self.get_action_handler)
        rospy.Service('init_traj_visualmpc', init_traj_visualmpc, self.init_traj_visualmpc_handler)

        ###
        print 'visual mpc server ready for taking requests!'
        rospy.spin()

    def init_traj_visualmpc_handler(self, req):
        self.igrp = req.igrp
        self.i_traj = req.itr
        self.t = 0
        goal_main = self.bridge.imgmsg_to_cv2(req.goalmain)
        pdb.set_trace()
        goal_main = cv2.cvtColor(goal_main, cv2.COLOR_BGR2RGB)
        # goal_aux1 = self.bridge.imgmsg_to_cv2(req.goalaux1)
        # goal_aux1 = cv2.cvtColor(goal_aux1, cv2.COLOR_BGR2RGB)
        Image.fromarray(goal_main).show()
        goal_main = goal_main.astype(np.float32) / 255.
        self.cem_controller.goal_image = goal_main
        print 'init traj{} group{}'.format(self.i_traj, self.igrp)
        return init_traj_visualmpcResponse()

    def get_action_handler(self, req):


        self.traj.X_full[self.t, :] = req.state
        main_img = self.bridge.imgmsg_to_cv2(req.main)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        aux1_img = self.bridge.imgmsg_to_cv2(req.aux1)
        aux1_img = cv2.cvtColor(aux1_img, cv2.COLOR_BGR2RGB)

        if 'single_view' in self.netconf:
            self.traj._sample_images[self.t] = main_img
        else:
            # flip order of main and aux1 to match training of double view architecture
            self.traj._sample_images[self.t] = np.concatenate((aux1_img, main_img), 2)

        self.desig_pos_aux1 = req.desig_pos_aux1
        self.goal_pos_aux1 = req.goal_pos_aux1

        mj_U, pos, best_ind, pix_distrib = self.cem_controller.act(self.traj, self.t,
                                                          req.desig_pos_aux1,
                                                          req.goal_pos_aux1)

        if 'predictor_propagation' in self.policyparams and self.t > 0:
            self.initial_pix_distrib.append(pix_distrib[-1][0])

        self.traj.U[self.t, :] = mj_U

        if self.t == self.agentparams['T'] -1:
            self.save_video()

        self.t += 1
        return get_actionResponse(tuple(mj_U))

    def save_video(self):
        file_path = self.netconf['current_dir'] + '/videos'
        imlist = np.split(self.traj._sample_images, self.agentparams['T'], axis=0)

        imfilename = file_path + '/traj{0}_gr{1}'.format(self.i_traj, self.igrp)
        cPickle.dump(imlist, open(imfilename+ '.pkl', 'wb'))

        if 'predictor_propagation' in self.policyparams:
            cPickle.dump(self.initial_pix_distrib, open(file_path + '/initial_pix_distrib.pkl'.format(self.t), 'wb'))
            self.initial_pix_distrib = [im.reshape((1,64,64)) for im in self.initial_pix_distrib]
            pdb.set_trace()
            pix_distrib = make_color_scheme(self.initial_pix_distrib, convert_to_float=False)
            gif = assemble_gif([imlist, pix_distrib], num_exp=1, convert_from_float=False)
            npy_to_gif(gif, file_path +'/traj{0}_gr{1}_withpixdistrib'.format(self.i_traj, self.igrp))
        else:
            imlist = [np.squeeze(im) for im in imlist]
            npy_to_gif(imlist, imfilename)



if __name__ ==  '__main__':
    Visual_MPC_Server()
