#!/usr/bin/env python
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

from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import *
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller

from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
from python_visual_mpc import __file__ as base_filepath


import rospy
import rospy.numpy_msg
from visual_mpc_rospkg.srv import *
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg

class Visual_MPC_Server(object):
    def __init__(self):
        base_dir = '/'.join(str.split(base_filepath, '/')[:-2])

        cem_exp_dir = base_dir + '/experiments/cem_exp/benchmarks_sawyer'
        hyperparams = imp.load_source('hyperparams', cem_exp_dir + '/base_hyperparams_sawyer.py')

        parser = argparse.ArgumentParser(description='Run benchmarks')
        parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
        parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
        parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
        parser.add_argument('--userobot', type=str, default='True', help='number of gpus to use')

        parser.add_argument('--redis', type=str, default='', help='necessary when using ray: the redis address of the head node')

        args = parser.parse_args()

        if args.userobot == 'True':
            self.use_robot = True
        elif args.userobot == 'False':
            self.use_robot = False

        if self.use_robot:
            rospy.init_node('visual_mpc_server')
            rospy.loginfo("init visual mpc server")

        benchmark_name = args.benchmark
        gpu_id = args.gpu_id
        ngpu = args.ngpu

        self.conf = hyperparams.config
        self.policyparams = hyperparams.policy
        self.agentparams = hyperparams.agent

        # load specific agent settings for benchmark:
        bench_dir = cem_exp_dir + '/' + benchmark_name

        if not os.path.exists(bench_dir):
            raise ValueError('benchmark directory does not exist')

        bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        if hasattr(bench_conf, 'policy'):
            self.policyparams.update(bench_conf.policy)
        if hasattr(bench_conf, 'agent'):
            self.agentparams.update(bench_conf.agent)
        print '-------------------------------------------------------------------'
        print 'verify planner settings!! '
        for key in self.policyparams.keys():
            print key, ': ', self.policyparams[key]
        for key in self.agentparams.keys():
            print key, ': ', self.agentparams[key]
        print '-------------------------------------------------------------------'

        if self.policyparams['usenet']:
            self.netconf = imp.load_source('params', self.policyparams['netconf']).configuration
            if 'multmachine' in self.policyparams:
                self.predictor = self.netconf['setup_predictor'](netconf=self.netconf, policyparams=self.policyparams, ngpu=ngpu,redis_address=args.redis)
            else:
                self.predictor = self.netconf['setup_predictor'](self.netconf, gpu_id, ngpu)
        else:
            self.netconf = {}
            self.predictor = None
        self.cem_controller = CEM_controller(self.agentparams, self.policyparams, self.predictor)
        ###########
        self.t = 0
        self.traj = Trajectory(self.agentparams, self.netconf)

        if self.use_robot:
            self.bridge = CvBridge()

        if 'ndesig' in self.policyparams:
            self.initial_pix_distrib1 = []
            self.initial_pix_distrib2 = []
        else:
            self.initial_pix_distrib = []


        self.save_subdir = None

        if self.use_robot:
            # initializing the servives:
            rospy.Service('get_action', get_action, self.get_action_handler)
            rospy.Service('init_traj_visualmpc', init_traj_visualmpc, self.init_traj_visualmpc_handler)

            ###
            print 'visual mpc server ready for taking requests!'
            rospy.spin()
        else:
            self.test_canon_examples()

    def test_canon_examples(self):
        b_exp = 2 #5drill  #2
        file_path_canon = '/home/frederik/Documents/catkin_ws/src/lsdc/pushing_data/canonical_examples'
        dict = cPickle.load(open(file_path_canon + '/pkl/example{}.pkl'.format(b_exp), 'rb'))
        desig_pix = np.stack([dict['desig_pix'], np.zeros(2)]).astype(np.int32)
        goal_pix = np.stack([dict['goal_pix'], np.zeros(2)]).astype(np.int32)

        sel_img = dict['images']
        sel_img = sel_img[:2]
        sel_img = (sel_img*255.).astype(np.uint8)
        state = dict['endeff']
        sel_state = state[:2]

        # for i in range(self.policyparams['T']):
        for t in range(2):
            self.traj.X_full[t, :] = sel_state[t]
            self.traj._sample_images[t] = sel_img[t]

            mj_U, pos, best_ind, init_pix_distrib = self.cem_controller.act(self.traj, t,
                                                                        desig_pix,
                                                                        goal_pix)

    def init_traj_visualmpc_handler(self, req):
        self.igrp = req.igrp
        self.i_traj = req.itr

        self.t = 0
        if 'use_goalimage' in self.policyparams:
            goal_main = self.bridge.imgmsg_to_cv2(req.goalmain)
            goal_main = cv2.cvtColor(goal_main, cv2.COLOR_BGR2RGB)
            # goal_aux1 = self.bridge.imgmsg_to_cv2(req.goalaux1)
            # goal_aux1 = cv2.cvtColor(goal_aux1, cv2.COLOR_BGR2RGB)
            Image.fromarray(goal_main).show()
            goal_main = goal_main.astype(np.float32) / 255.
            self.cem_controller.goal_image = goal_main

        print 'init traj{} group{}'.format(self.i_traj, self.igrp)

        if 'ndesig' in self.policyparams:
            self.initial_pix_distrib = []
        else:
            self.initial_pix_distrib1 = []
            self.initial_pix_distrib2 = []

        self.cem_controller = CEM_controller(self.agentparams, self.policyparams, self.predictor, save_subdir=req.save_subdir)
        self.save_subdir = req.save_subdir
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

        if 'ndesig' in self.policyparams:
            mj_U, pos, best_ind, init_pix_distrib1, init_pix_distrib2  = self.cem_controller.act(self.traj, self.t,
                                                                                  req.desig_pos_aux1,
                                                                                  req.goal_pos_aux1)

            if 'predictor_propagation' in self.policyparams and self.t > 0:
                self.initial_pix_distrib1.append(init_pix_distrib1[-1][0])
                self.initial_pix_distrib2.append(init_pix_distrib2[-1][0])
        else:
            mj_U, pos, best_ind, init_pix_distrib = self.cem_controller.act(self.traj, self.t,
                                                                            req.desig_pos_aux1,
                                                                            req.goal_pos_aux1)

            if 'predictor_propagation' in self.policyparams and self.t > 0:
                self.initial_pix_distrib.append(init_pix_distrib[-1][0])

        self.traj.U[self.t, :] = mj_U

        if self.t == self.agentparams['T'] -1:
            if 'no_pixdistrib_video' not in self.policyparams:
                self.save_video()

        self.t += 1
        return get_actionResponse(tuple(mj_U))

    def save_video(self):
        file_path = self.netconf['current_dir'] + '/videos'
        if self.save_subdir != None:
            file_path = self.netconf['current_dir'] + "/"+ self.save_subdir +'/videos'

        imlist = np.split(self.traj._sample_images, self.agentparams['T'], axis=0)

        imfilename = file_path + '/traj{0}_gr{1}'.format(self.i_traj, self.igrp)
        cPickle.dump(imlist, open(imfilename+ '.pkl', 'wb'))

        if 'predictor_propagation' in self.policyparams:
            if 'ndesig' in self.policyparams:
                cPickle.dump(self.initial_pix_distrib1,
                             open(file_path + '/initial_pix_distrib1.pkl'.format(self.t), 'wb'))
                cPickle.dump(self.initial_pix_distrib2,
                             open(file_path + '/initial_pix_distrib2.pkl'.format(self.t), 'wb'))
                self.initial_pix_distrib1 = [im.reshape((1, 64, 64)) for im in self.initial_pix_distrib1]
                pix_distrib1 = make_color_scheme(self.initial_pix_distrib1, convert_to_float=False)

                self.initial_pix_distrib2 = [im.reshape((1, 64, 64)) for im in self.initial_pix_distrib2]
                pix_distrib2 = make_color_scheme(self.initial_pix_distrib2, convert_to_float=False)
                gif = assemble_gif([imlist, pix_distrib1, pix_distrib2], num_exp=1, convert_from_float=False)

                npy_to_gif(gif, file_path + '/traj{0}_gr{1}_withpixdistrib'.format(self.i_traj, self.igrp))
            else:
                cPickle.dump(self.initial_pix_distrib, open(file_path + '/initial_pix_distrib.pkl'.format(self.t), 'wb'))
                self.initial_pix_distrib = [im.reshape((1,64,64)) for im in self.initial_pix_distrib]
                pix_distrib = make_color_scheme(self.initial_pix_distrib, convert_to_float=False)
                gif = assemble_gif([imlist, pix_distrib], num_exp=1, convert_from_float=False)
                npy_to_gif(gif, file_path +'/traj{0}_gr{1}_withpixdistrib'.format(self.i_traj, self.igrp))
        else:
            imlist = [np.squeeze(im) for im in imlist]
            npy_to_gif(imlist, imfilename)



if __name__ ==  '__main__':
    Visual_MPC_Server()
