#!/usr/bin/env python
import os

from python_visual_mpc.goaldistancenet.setup_gdn import setup_gdn
import shutil
import socket
import numpy as np
import pdb
from PIL import Image
import pickle
import imp
import copy
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

def plot_warp_err(traj, dir):

    warperrs = []
    tradeoff = []
    for tstep in traj.plan_stat[1:]:
        warperrs.append(tstep['warperrs'])
        tradeoff.append(tstep['tradeoff'])

    tradeoff = np.stack(tradeoff, 0)
    warperrs = np.stack(warperrs, 0)

    pickle.dump({'warperrs':warperrs, 'tradeoff':tradeoff}, open(dir +  '/warperrs_tradeoff.pkl', 'wb'))

    # warperrs shape: tstep, ncam, numtrack
    plt.figure()
    ax = plt.gca()
    ax.plot(warperrs[:,0,0], marker ='d', label='start')
    ax.plot(warperrs[:,0,1], marker='o', label='goal')
    ax.legend()
    plt.savefig(dir + '/warperrors.png')

    plt.figure()
    ax = plt.gca()

    ax.plot(tradeoff[:,0,0], marker='d', label='tradeoff for start')
    ax.plot(tradeoff[:,0,1], marker='d', label='tradeoff for goal')
    ax.legend()
    plt.savefig(dir + '/tradeoff.png')




class Visual_MPC_Server(object):
    def __init__(self, cmd_args=False):

        print('started visual MPC server')
        # pdb.set_trace()

        base_dir = '/'.join(str.split(base_filepath, '/')[:-2])

        cem_exp_dir = base_dir + '/experiments/cem_exp/benchmarks_sawyer'
        self.use_robot = True
        rospy.init_node('visual_mpc_server')
        rospy.loginfo("init visual mpc server")
        if cmd_args:
            parser = argparse.ArgumentParser(description='Run benchmarks')
            parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
            parser.add_argument('--gpu_id', type=int, default=0, help='value to set for cuda visible devices variable')
            parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
            args = parser.parse_args()
            benchmark_name = args.benchmark
            ngpu = args.ngpu
            gpu_id = args.gpu_id
        else:
            benchmark_name = rospy.get_param('~exp')
            gpu_id = rospy.get_param('~gpu_id')
            ngpu = rospy.get_param('~ngpu')

        # load specific agent settings for benchmark:
        bench_dir = cem_exp_dir + '/' + benchmark_name
        print('using configuration: ',benchmark_name)

        if not os.path.exists(bench_dir):
            raise ValueError('benchmark directory does not exist {}'.format(bench_dir))

        bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        if hasattr(bench_conf, 'policy'):
            self.policyparams = bench_conf.policy
        if hasattr(bench_conf, 'agent'):
            self.agentparams = bench_conf.agent

        if 'opencv_tracking' in self.agentparams:
            assert 'predictor_propagation' not in self.policyparams

        print('-------------------------------------------------------------------')
        print('verify planner settings!! ')
        for key in list(self.policyparams.keys()):
            print(key, ': ', self.policyparams[key])
        for key in list(self.agentparams.keys()):
            print(key, ': ', self.agentparams[key])
        print('-------------------------------------------------------------------')

        if self.policyparams['usenet']:
            self.netconf = imp.load_source('params', self.policyparams['netconf']).configuration
            if 'multmachine' in self.policyparams:
                self.predictor = self.netconf['setup_predictor']({}, self.netconf, policyparams=self.policyparams, ngpu=ngpu,redis_address=args.redis)
            else:
                self.predictor = self.netconf['setup_predictor']({}, self.netconf, gpu_id, ngpu)
        else:
            self.netconf = {}
            self.predictor = None

        if 'gdnconf' in self.policyparams:
            self.gdnconf = imp.load_source('params', self.policyparams['gdnconf']).configuration
            self.goal_image_warper = setup_gdn(self.gdnconf, gpu_id)
        else:
            self.gdnconf = {}
            self.goal_image_warper = None

        self.cem_controller = CEM_controller(self.agentparams, self.policyparams, self.predictor, self.goal_image_warper)
        ###########
        self.t = 0


        if self.use_robot:
            self.bridge = CvBridge()

        self.initial_pix_distrib = []


        self.save_subdir = None

        if self.use_robot:
            # initializing the servives:
            rospy.Service('get_action', get_action, self.get_action_handler)
            rospy.Service('init_traj_visualmpc', init_traj_visualmpc, self.init_traj_visualmpc_handler)

            ###
            print('visual mpc server ready for taking requests!')
            rospy.spin()


    def init_traj_visualmpc_handler(self, req):
        self.igrp = req.igrp
        self.i_traj = req.itr

        self.traj = Trajectory(self.agentparams)
        self.traj.i_tr = self.i_traj

        self.t = 0
        goal_main = self.bridge.imgmsg_to_cv2(req.goalmain)
        goal_main = goal_main.astype(np.float32) / 255.
        if 'use_goal_image' in self.policyparams:
            self.goal_image = goal_main
        else:
            self.goal_image = np.zeros_like(goal_main)

        print('init traj{} group{}'.format(self.i_traj, self.igrp))

        self.initial_pix_distrib = []

        self.cem_controller = CEM_controller(self.agentparams, self.policyparams, self.predictor, self.goal_image_warper, save_subdir=req.save_subdir)
        self.save_subdir = req.save_subdir
        return init_traj_visualmpcResponse()

    def get_action_handler(self, req):
        print('handling action')

        self.traj.X_full[self.t, :] = req.state[:self.agentparams['sdim']]
        main_img = self.bridge.imgmsg_to_cv2(req.main)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

        self.traj.images[self.t] = main_img[None]

        self.desig_pos_aux1 = req.desig_pos_aux1
        self.goal_pos_aux1 = req.goal_pos_aux1

        mj_U, plan_stat = self.cem_controller.act(self.traj, self.t,
                                        req.desig_pos_aux1,
                                        req.goal_pos_aux1,
                                        self.goal_image[None])

        self.traj.plan_stat.append(copy.deepcopy(plan_stat))
        self.traj.actions[self.t, :] = mj_U

        if self.t == self.agentparams['T'] -1:
            print('done')

            if 'register_gtruth' in self.policyparams:
                plot_warp_err(self.traj, self.agentparams['record'])

        self.t += 1

        action_resp = np.zeros(self.agentparams['adim'])
        action_resp[:self.agentparams['adim']] = mj_U
        return get_actionResponse(tuple(action_resp))

    def save_video(self):
        file_path = self.netconf['current_dir'] + '/videos'
        if self.save_subdir != None:
            file_path = self.netconf['current_dir'] + "/"+ self.save_subdir +'/videos'

        imlist = np.split(self.traj.images, self.agentparams['T'], axis=0)

        imfilename = file_path + '/traj{0}_gr{1}'.format(self.i_traj, self.igrp)
        pickle.dump(imlist, open(imfilename+ '.pkl', 'wb'))

        if 'predictor_propagation' in self.policyparams:
            if 'ndesig' in self.policyparams:
                pickle.dump(self.initial_pix_distrib1,
                             open(file_path + '/initial_pix_distrib1.pkl'.format(self.t), 'wb'))
                pickle.dump(self.initial_pix_distrib2,
                             open(file_path + '/initial_pix_distrib2.pkl'.format(self.t), 'wb'))
                self.initial_pix_distrib1 = [im.reshape((1, 64, 64)) for im in self.initial_pix_distrib1]
                pix_distrib1 = make_color_scheme(self.initial_pix_distrib1, convert_to_float=False)

                self.initial_pix_distrib2 = [im.reshape((1, 64, 64)) for im in self.initial_pix_distrib2]
                pix_distrib2 = make_color_scheme(self.initial_pix_distrib2, convert_to_float=False)
                gif = assemble_gif([imlist, pix_distrib1, pix_distrib2], num_exp=1, convert_from_float=False)

                npy_to_gif(gif, file_path + '/traj{0}_gr{1}_withpixdistrib'.format(self.i_traj, self.igrp))
            else:
                pickle.dump(self.initial_pix_distrib, open(file_path + '/initial_pix_distrib.pkl'.format(self.t), 'wb'))
                self.initial_pix_distrib = [im.reshape((1,64,64)) for im in self.initial_pix_distrib]
                pix_distrib = make_color_scheme(self.initial_pix_distrib, convert_to_float=False)
                gif = assemble_gif([imlist, pix_distrib], num_exp=1, convert_from_float=False)
                npy_to_gif(gif, file_path +'/traj{0}_gr{1}_withpixdistrib'.format(self.i_traj, self.igrp))
        else:
            imlist = [np.squeeze(im) for im in imlist]
            npy_to_gif(imlist, imfilename)


if __name__ ==  '__main__':
    Visual_MPC_Server(cmd_args=True)
