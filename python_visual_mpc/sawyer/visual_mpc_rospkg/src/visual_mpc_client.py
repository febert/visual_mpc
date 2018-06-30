#!/usr/bin/env python
import time
import argparse
import pickle
import copy
import imp
import os
import pdb
from datetime import datetime

import cv2

import matplotlib.pyplot as plt
print plt.get_backend()



import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Quaternion,
)
from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)

from python_visual_mpc.region_proposal_networks.rpn_tracker import Too_few_objects_found_except

import python_visual_mpc
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils import inverse_kinematics, robot_controller, robot_recorder
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_msgs.msg import Int64
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.checkpoint import write_ckpt, write_timing_file, parse_ckpt
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.copy_from_remote import scp_pix_distrib_files
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.tracking_client import OpenCV_Track_Listener
from visual_mpc_rospkg.srv import get_action, init_traj_visualmpc
from rospy.numpy_msg import numpy_msg
from visual_mpc_rospkg.msg import intarray

from wsg_50_common.msg import Cmd, Status

# from python_visual_mpc.region_proposal_networks.rpn_tracker import RPN_Tracker
from std_msgs.msg import String
class Traj_aborted_except(Exception):
    pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

from python_visual_mpc import __file__ as base_filepath

class Visual_MPC_Client():
    def __init__(self, cmd_args=False):
        print('started visual MPC client')
        # pdb.set_trace()
        self.ctrl = robot_controller.RobotController()
        self.base_dir = '/'.join(str.split(base_filepath, '/')[:-2])

        self.use_save_subdir = False
        if cmd_args:
            parser = argparse.ArgumentParser(description='')
            parser.add_argument('experiment', type=str, help='experiment name')
            parser.add_argument('--robot_name', type=str, default='vestri', help='robot name')
            parser.add_argument('--use_save_subdir', type=str, default='False')
            args = parser.parse_args()
            benchmark_name = args.experiment
            self.use_gui = False
            self.robot_name = args.robot_name

            if args.use_save_subdir == 'True':
                self.use_save_subdir = True
        else:
            self.use_gui = rospy.get_param('~gui')   #experiment name
            benchmark_name = rospy.get_param('~exp')   #experiment name
            self.robot_name = rospy.get_param('~robot')  # experiment name

        cem_exp_dir = self.base_dir + '/experiments/cem_exp/benchmarks_sawyer'
        bench_dir = cem_exp_dir + '/' + benchmark_name

        if not os.path.exists(bench_dir):
            print(bench_dir)
            raise ValueError('benchmark directory does not exist')
        bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        self.policyparams = bench_conf.policy
        self.agentparams = bench_conf.agent

        self.benchname = benchmark_name

        if self.agentparams['adim'] == 5:
            self.enable_rot = True
        else:
            self.enable_rot = False

        if 'netconf' in self.policyparams:
            hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
            self.netconf = hyperparams.configuration
        else:
            self.netconf = {}

        if hasattr(bench_conf, 'dataconf'):
            dataconf = bench_conf.dataconf
        else:
            dataconf_file = self.base_dir + '/'.join(str.split(self.netconf['data_dir'], '/')[:-1]) + '/conf.py'
            dataconf = imp.load_source('hyperparams', dataconf_file).configuration

        self.img_height = self.netconf['img_height']
        self.img_width = self.netconf['img_width']

        self.ndesig = self.agentparams['ndesig']

        self.num_traj = 5000


        self.action_sequence_length = self.agentparams['T'] # number of snapshots that are taken
        self.use_robot = True
        self.robot_move = True #######
        self.reset_active = False # used by the gui to abort trajectories.

        self.save_subdir = ""

        self.use_aux = False

        self.get_action_func = rospy.ServiceProxy('get_action', get_action)
        self.init_traj_visual_func = rospy.ServiceProxy('init_traj_visualmpc', init_traj_visualmpc)

        self.alive_publisher = rospy.Publisher('still_alive', String, queue_size=10)

        if self.use_robot:
            self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
            self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
            self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)
            self.fksrv_name = "ExternalTools/right/PositionKinematicsNode/FKService"
            self.fksrv = rospy.ServiceProxy(self.fksrv_name, SolvePositionFK)

            self.weiss_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
            rospy.Subscriber("/wsg_50_driver/status", Status, self.save_weiss_pos)

        self.use_imp_ctrl = True
        self.interpolate = True
        self.save_active = True
        self.bridge = CvBridge()

        self.action_interval = 1. #Hz
        self.traj_duration = self.action_sequence_length*self.action_interval
        self.control_rate = rospy.Rate(1000)

        self.sdim = self.agentparams['sdim']
        self.adim = self.agentparams['adim']

        if self.adim == 5:
            self.wristrot = True
        else: self.wristrot = False

        # drive to neutral position:
        self.set_neutral_with_impedance()

        self.goal_pos_main = np.zeros([self.ndesig,2])   # the first index is for the ndesig and the second is r,c
        self.desig_pos_main = np.zeros([self.ndesig, 2])
        self.desig_hpos_main = np.zeros([self.ndesig, 2])

        #highres position used when doing tracking
        self.desig_hpos_main = None

        self.tlast_ctrl_alive = rospy.get_time()
        self.tlast_gripper_status = rospy.get_time()

        if 'collect_data' in self.agentparams:
            self.data_collection = True
            save_video = True
            save_actions = True
            assert self.robot_name != ''
            rospy.Subscriber("ctrl_alive", numpy_msg(intarray), self.ctr_alive)
        else:
            save_video = True
            save_actions = False
            self.data_collection = False

        if self.use_save_subdir:
            self.recorder_save_dir = None
            # self.recorder_save_dir = self.base_dir + "/experiments/cem_exp/benchmarks_sawyer/" + self.benchname + \
            #                             '/' + self.save_subdir + "/videos"
        elif self.data_collection:
            self.recorder_save_dir  =self.base_dir + "/experiments/cem_exp/benchmarks_sawyer/" + self.benchname + "/data"
        else:
            self.recorder_save_dir = self.base_dir + "/experiments/cem_exp/benchmarks_sawyer/" + self.benchname + "/videos"

        self.num_pic_perstep = 4
        self.nsave = self.action_sequence_length * self.num_pic_perstep

        self.recorder = robot_recorder.RobotRecorder(agent_params=self.agentparams,
                                                     save_dir=self.recorder_save_dir,
                                                     dataconf=dataconf,
                                                     seq_len=self.nsave,
                                                     use_aux=self.use_aux,
                                                     save_video=save_video,
                                                     save_actions=save_actions,
                                                     save_images=True,
                                                     image_shape=(self.img_height, self.img_width),
                                                     save_lowres=True)

        while self.recorder.ltob.img_cv2 is None:
            print('waiting for images')
            rospy.sleep(0.5)

            if self.data_collection == True:
                self.checkpoint_file = os.path.join(self.recorder.save_dir, 'checkpoint.txt')
                # self.rpn_tracker = RPN_Tracker(self.recorder_save_dir, self.recorder)
                self.rpn_tracker = None
                self.run_data_collection()
            elif self.use_gui:
                rospy.Subscriber('visual_mpc_cmd', numpy_msg(intarray), self.run_visual_mpc_cmd)
                rospy.Subscriber('visual_mpc_reset_cmd', numpy_msg(intarray), self.run_visual_mpc_reset_cmd)
                rospy.spin()
            else:
                self.run_visual_mpc()

    def ctr_alive(self, data):
        self.tlast_ctrl_alive = rospy.get_time()

    def run_visual_mpc_cmd(self, data):
        """"
        data is of shape [2, ndesig, 2]
        data[0] contains designated pixel positions (row, column format)
        data[1] contains goal pixel positions
        """
        points = data.data.reshape((2, self.ndesig, 2))
        self.desig_pos_main = points[0]
        self.desig_pos_main.setflags(write=1)
        self.goal_pos_main = points[1]
        self.goal_pos_main.setflags(write=1)

        self.reset_active = False
        self.run_trajectory(0)

    def run_visual_mpc_reset_cmd(self, data):
        """"
        data is of shape [2, ndesig, 2]
        data[0] contains designated pixel positions (row, column format)
        data[1] contains goal pixel positions
        """
        print('reset activated')
        self.reset_active = True
        self.set_neutral_with_impedance(2.5)

    def mark_goal_desig(self, itr):
        if 'use_goal_image' in self.policyparams:
            print('put object in goal configuration')
            if 'ntask' in self.agentparams:
                ntask = self.agentparams['ntask']
            else: ntask =1
            raw_input()
            imagemain = self.recorder.ltob.img_cropped
            imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
            c_main = Getdesig(imagemain, self.recorder_save_dir, 'goal', n_desig=ntask,
                              im_shape=[self.img_height, self.img_width], clicks_per_desig=1)
            self.goal_pos_main = c_main.desig.astype(np.int64)
            print('goal pos main:', self.goal_pos_main)
            if 'image_medium' in self.agentparams:
                self.goal_image = self.recorder.ltob.img_cropped_medium
            else:
                self.goal_image = self.recorder.ltob.img_cropped
            self.goal_image = cv2.cvtColor(self.goal_image, cv2.COLOR_BGR2RGB)

            print('put object in start configuration')
            raw_input()
            imagemain = self.recorder.ltob.img_cropped
            imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
            c_main = Getdesig(imagemain, self.recorder_save_dir, 'start', n_desig=ntask,
                              im_shape=[self.img_height, self.img_width], clicks_per_desig=1)
            self.desig_pos_main = c_main.desig.astype(np.int64)
            print('desig pos aux1:', self.desig_pos_main)
        else:
            imagemain = self.recorder.ltob.img_cropped
            imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
            c_main = Getdesig(imagemain, self.recorder_save_dir, 'start_traj{}'.format(itr),
                              self.ndesig, im_shape=[self.img_height, self.img_width])
            self.desig_pos_main = c_main.desig.astype(np.int64)
            print('desig pos aux1:', self.desig_pos_main)
            self.goal_pos_main = c_main.goal.astype(np.int64)
            print('goal pos main:', self.goal_pos_main)
            self.goal_image = np.zeros_like(imagemain)


    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

    def set_weiss_gripper(self, des_pos):
        """
        :param des_pos:
        :return:
        """
        cmd = Cmd()
        cmd.speed = 100.
        cmd.pos = des_pos
        # print('command weiss', cmd.pos)
        self.weiss_pub.publish(cmd)

    def save_weiss_pos(self, status):

        self.gripper_pos = status.width
        self.tlast_gripper_status = rospy.get_time()

    def run_visual_mpc(self):
        while True:
            tstart = datetime.now()
            # self.run_trajectory_const_speed(tr)
            done = False
            while not done:
                try:
                    self.run_trajectory(0)
                    done = True
                except Traj_aborted_except:
                    self.recorder.delete_traj(0)

            delta = datetime.now() - tstart
            print('trajectory {0} took {1} seconds'.format(0, delta.total_seconds()))


    def run_data_collection(self):

        # check if there is a checkpoint from which to resume
        if os.path.isfile(self.checkpoint_file):
            last_tr, last_grp = parse_ckpt(self.checkpoint_file)
            start_tr = last_tr + 1
            print('resuming data collection at trajectory {}'.format(start_tr))
            self.recorder.igrp = last_grp
            try:
                self.recorder.delete_traj(start_tr)
            except:
                print('trajectory was not deleted')
        else:
            start_tr = 0

        accum_time = 0
        nfail_traj = 0  #count number of failed trajectories within accum_time
        for tr in range(start_tr, self.num_traj):

            tstart = datetime.now()
            # self.run_trajectory_const_speed(tr)
            done = False
            while not done:
                try:
                    self.run_trajectory(tr)
                    done = True
                except Too_few_objects_found_except:
                    print('too few objects found, redistributing !!')
                    if self.robot_move:
                        self.redistribute_objects()
                except Traj_aborted_except:
                    self.recorder.delete_traj(tr)
                    nfail_traj +=1
                    rospy.sleep(.2)

            if ((tr+1)% 10) == 0:
                self.redistribute_objects()

            delta = datetime.now() - tstart
            print('trajectory {0} took {1} seconds'.format(tr, delta.total_seconds()))
            accum_time += delta.total_seconds()

            avg_nstep = 80
            if ((tr+1)% avg_nstep) == 0:
                average = accum_time/avg_nstep
                write_timing_file(self.recorder.save_dir, average, avg_nstep, nfail_traj)
                accum_time = 0
                nfail_traj = 0

            write_ckpt(self.checkpoint_file, tr, self.recorder.igrp)

            # if ((tr+1) % 3000) == 0:
            #     print 'change objects!'
            #     pdb.set_trace()
            self.alive_publisher.publish('still alive!')


    def get_endeffector_pos(self):
        """
        :param pos_only: only return postion
        :return:
        """

        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = self.ctrl.limb.joint_names()
        joints.position = [self.ctrl.limb.joint_angle(j)
                        for j in joints.name]

        # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
        fkreq.tip_names.append('right_hand')
        try:
            rospy.wait_for_service(self.fksrv_name, 5)
            resp = self.fksrv(fkreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        pos = np.array([resp.pose_stamp[0].pose.position.x,
                         resp.pose_stamp[0].pose.position.y,
                         resp.pose_stamp[0].pose.position.z,
                         ])

        if not self.wristrot:
            return pos
        else:
            quat = np.array([resp.pose_stamp[0].pose.orientation.x,
                             resp.pose_stamp[0].pose.orientation.y,
                             resp.pose_stamp[0].pose.orientation.z,
                             resp.pose_stamp[0].pose.orientation.w
                             ])

            zangle = self.quat_to_zangle(quat)
            return np.concatenate([pos, zangle])

    def quat_to_zangle(self, quat):
        """
        :param quat: quaternion with only
        :return: zangle in rad
        """
        phi = np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2 *(quat[1]**2 + quat[2]**2))
        return np.array([phi])

    def zangle_to_quat(self, zangle):
        quat = Quaternion(  # downward and turn a little
            x=np.cos(zangle / 2),
            y=np.sin(zangle / 2),
            z=0.0,
            w=0.0
        )
        return quat

    def init_visual_mpc_server(self):
        try:
            goal_img_aux1 = np.zeros([self.img_height, self.img_width, 3])
            goal_img_aux1 = self.bridge.cv2_to_imgmsg(goal_img_aux1)
            goal_img_main = self.bridge.cv2_to_imgmsg(self.goal_image)
            rospy.wait_for_service('init_traj_visualmpc', timeout=15.)
            print('waiting for service init_traj_visualmpc')
            self.init_traj_visual_func(0, 0, goal_img_main, goal_img_aux1, self.save_subdir)
        except (rospy.ServiceException, rospy.ROSException) as e:
            raise ValueError("Service call failed: %s" % (e,))

    def run_trajectory(self, i_tr):

        print('setting neutral')
        rospy.sleep(.1)
        # drive to neutral position:
        if self.robot_move:
            self.set_neutral_with_impedance()
        rospy.sleep(.1)

        if self.ctrl.sawyer_gripper:
            self.ctrl.gripper.open()
        else:
            self.set_weiss_gripper(50.)

        self.gripper_closed = False


        if self.use_save_subdir:
            self.save_subdir = raw_input('enter subdir to save data:')

            self.recorder_save_dir = self.base_dir + "/experiments/cem_exp/benchmarks_sawyer/" + self.benchname + \
                                     '/bench/' + self.save_subdir + "/videos"
            self.recorder.image_folder = self.recorder_save_dir
            self.recorder.curr_traj = self.curr_traj = robot_recorder.Trajectory(self.recorder.state_sequence_length)

        if self.data_collection:
            self.recorder.init_traj(i_tr)
        else:
            if not os.path.exists(self.recorder_save_dir):
                os.makedirs(self.recorder_save_dir)

        if self.data_collection:
            rospy.sleep(.1)
            if self.rpn_tracker == None:
                self.desig_pos_main[0] = np.zeros(2)
                self.goal_pos_main[0] = np.zeros(2)
            else:
                im = cv2.cvtColor(self.recorder.ltob.img_cv2, cv2.COLOR_BGR2RGB)
                single_desig_pos, single_goal_pos = self.rpn_tracker.get_task(im,self.recorder.traj_folder)
                self.desig_pos_main[0] = single_desig_pos
                self.goal_pos_main[0] = single_goal_pos
        elif not self.use_gui:
            self.mark_goal_desig(i_tr)

        if self.netconf != {}:
            self.init_visual_mpc_server()

        self.lower_height = 0.22  # using old gripper : 0.16
        self.delta_up = 0.13

        self.xlim = [0.46, 0.83]  # min, max in cartesian Xdirection
        self.ylim = [-0.17, 0.17]  # min, max in cartesian Y-directionn

        if 'random_startpos' in self.policyparams:
            startpos = np.array([np.random.uniform(self.xlim[0], self.xlim[1]), np.random.uniform(self.ylim[0], self.ylim[1])])
        elif 'startpos_basedon_click' in self.agentparams:
            print('setting startpos based on click!')
            assert self.ndesig == 1

            print('desig pos', self.desig_pos_main)
            startpos_x = self.desig_pos_main[0,0] / float(self.img_height) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            startpos_y = self.desig_pos_main[0,1] / float(self.img_width) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]

            x_offset = -0.07
            startpos_x += x_offset
            startpos_x = np.clip(startpos_x, self.xlim[0], self.xlim[1])
            startpos_y = np.clip(startpos_y, self.ylim[0], self.ylim[1])

            startpos = np.array([startpos_x, startpos_y])

        else: startpos = self.get_endeffector_pos()[:2]
        print('startpos', startpos)

        start_angle = np.array([0.])

        go_up_at_start = True

        gripper_state = np.array([0.])  # gripper open
        if go_up_at_start:
            self.des_pos = np.concatenate([startpos, np.array([self.lower_height+self.delta_up]), start_angle], axis=0)
            self.gripper_up = True
        else:
            self.des_pos = np.concatenate([startpos, np.array([self.lower_height]), start_angle], axis=0)
            self.gripper_up = False

        self.topen, self.t_down = 0, 0

        #move to start:
        self.move_to_startpos(self.des_pos)

        if 'opencv_tracking' in self.agentparams:
            self.tracker = OpenCV_Track_Listener(self.agentparams,
                                                 self.recorder,
                                                 self.desig_pos_main)
        rospy.sleep(0.7)

        i_step = 0  # index of current commanded point

        self.ctrl.limb.set_joint_position_speed(.20)
        self.previous_des_pos = copy.deepcopy(self.des_pos)
        start_time = -1

        isave = 0
        t_start = time.time()
        query_times = []

        while isave < self.nsave and not self.reset_active:
            self.curr_delta_time = rospy.get_time() - start_time
            if self.curr_delta_time > self.action_interval and i_step < self.action_sequence_length:
                if 'manual_correction' in self.agentparams:
                    imagemain = self.recorder.ltob.img_cropped
                    imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
                    c_main = Getdesig(imagemain, self.recorder_save_dir, '_t{}'.format(i_step), self.ndesig, only_desig=True)
                    self.desig_pos_main = c_main.desig.astype(np.int64)
                elif 'opencv_tracking' in self.agentparams:
                    self.desig_pos_main, self.desig_hpos_main = self.tracker.get_track()  #tracking only works for 1 desig. pixel!!

                # print 'current position error', self.des_pos - self.get_endeffector_pos(pos_only=True)

                self.previous_des_pos = copy.deepcopy(self.des_pos)
                get_action_start = time.time()
                action_vec = self.query_action(i_step)
                query_times.append(time.time()-get_action_start)

                self.des_pos, going_down = self.apply_act(self.des_pos, action_vec, i_step)
                print('action vec', action_vec)
                start_time = rospy.get_time()

                # print('prev_desired pos in step {0}: {1}'.format(i_step, self.previous_des_pos))
                # print('new desired pos in step {0}: {1}'.format(i_step, self.des_pos))

                if going_down:
                    self.action_interval = 1.5
                else:
                    self.action_interval = 1.

                self.t_prev = start_time
                self.t_next = start_time + self.action_interval
                # print('t_prev', self.t_prev)
                # print('t_next', self.t_next)

                isave_substep  = 0
                tsave = np.linspace(self.t_prev, self.t_next, num=self.num_pic_perstep, dtype=np.float64)
                # print('tsave', tsave)
                print('applying action {}'.format(i_step))
                i_step += 1

            des_joint_angles = self.get_interpolated_joint_angles()

            if self.save_active:
                if isave_substep < len(tsave):
                    if rospy.get_time() > tsave[isave_substep] -.01:
                        if 'opencv_tracking' in self.agentparams:
                            _, self.desig_hpos_main = self.tracker.get_track()
                        self.recorder.save(isave, action_vec, self.get_endeffector_pos(), self.desig_hpos_main, self.desig_pos_main, self.goal_pos_main)
                        isave_substep += 1
                        isave += 1
            try:
                if self.robot_move and not self.reset_active:
                    self.move_with_impedance(des_joint_angles)
                    self.set_weiss_gripper(50.)
                    # print des_joint_angles
            except OSError:
                rospy.logerr('collision detected, stopping trajectory, going to reset robot...')
                rospy.sleep(.5)
                raise Traj_aborted_except('raising Traj_aborted_except')
            # if self.ctrl.limb.has_collided():
            #     rospy.logerr('collision detected!!!')
            #     rospy.sleep(.5)
            #     raise Traj_aborted_except('raising Traj_aborted_except')

            self.control_rate.sleep()

        if self.reset_active:
            return

        print('average iteration took {0} seconds'.format((time.time() - t_start) / self.action_sequence_length))
        print('average action query took {0} seconds'.format(np.mean(np.array(query_times))))


        if not self.data_collection and not self.use_gui:
            self.save_final_image(i_tr)
            self.recorder.save_highres()
            #copy files with pix distributions from remote and make gifs
            # scp_pix_distrib_files(self.policyparams, self.agentparams)
            # v = Visualizer_tkinter(append_masks=False,
            #                        filepath=self.policyparams['current_dir'] + '/verbose',
            #                        numex=5)
            # v.build_figure()

        if not self.reset_active:
            self.goup()
        if self.ctrl.sawyer_gripper:
            self.ctrl.gripper.open()
        else:
            print('delta t gripper status', rospy.get_time() - self.tlast_gripper_status)
            if rospy.get_time() - self.tlast_gripper_status > 10.:
                print('gripper stopped working!')
                pdb.set_trace()
            self.set_weiss_gripper(100.)

        if self.data_collection:
            if rospy.get_time() - self.tlast_ctrl_alive > 10.:
                print('controller failed')
                pdb.set_trace()


    def goup(self):
        print("going up at the end..")
        self.des_pos[2] = self.lower_height + 0.15
        desired_pose = self.get_des_pose(self.des_pos)
        start_joints = self.ctrl.limb.joint_angles()
        try:
            des_joint_angles = inverse_kinematics.get_joint_angles(desired_pose, seed_cmd=start_joints,
                                                                   use_advanced_options=True)
        except ValueError:
            rospy.logerr('no inverse kinematics solution found, '
                         'going to reset robot...')
            current_joints = self.ctrl.limb.joint_angles()
            self.ctrl.limb.set_joint_positions(current_joints)
            raise Traj_aborted_except('raising Traj_aborted_except')
        if self.robot_move:
            self.move_with_impedance_sec(des_joint_angles, duration=1.)

    def get_des_pose(self, des_pos):
        quat = self.zangle_to_quat(des_pos[3])
        desired_pose = inverse_kinematics.get_pose_stamped(des_pos[0],
                                                           des_pos[1],
                                                           des_pos[2],
                                                           quat)
        return desired_pose

    def save_final_image(self, i_tr):
        imagemain = self.recorder.ltob.img_cropped
        cv2.imwrite(self.recorder_save_dir + '/finalimage.png', imagemain, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

    def calc_interpolation(self, previous_goalpoint, next_goalpoint, t_prev, t_next):
        """
        interpolate cartesian positions (x,y,z) between last goalpoint and previous goalpoint at the current time
        :param previous_goalpoint:
        :param next_goalpoint:
        :param goto_point:
        :param tnewpos:
        :return: des_pos
        """
        assert (rospy.get_time() >= t_prev)
        des_pos = previous_goalpoint + (next_goalpoint - previous_goalpoint) * (rospy.get_time()- t_prev)/ (t_next - t_prev)
        if rospy.get_time() - t_next > 2.5:
            des_pos = next_goalpoint
            print('t - tnext > 2.5!!!!')
            pdb.set_trace()

        # print 'current_delta_time: ', self.curr_delta_time
        # print "interpolated pos:", des_pos

        return des_pos

    def get_interpolated_joint_angles(self):
        int_des_pos = self.calc_interpolation(self.previous_des_pos, self.des_pos, self.t_prev, self.t_next)
        # print 'interpolated des_pos: ', int_des_pos

        desired_pose = self.get_des_pose(int_des_pos)
        start_joints = self.ctrl.limb.joint_angles()
        try:
            des_joint_angles = inverse_kinematics.get_joint_angles(desired_pose, seed_cmd=start_joints,
                                                                   use_advanced_options=True)
        except ValueError:
            rospy.logerr('no inverse kinematics solution found, '
                         'going to reset robot...')
            current_joints = self.ctrl.limb.joint_angles()
            self.ctrl.limb.set_joint_positions(current_joints)
            raise Traj_aborted_except('raising Traj_aborted_except')

        return des_joint_angles

    def query_action(self, istep):

        if 'image_medium' in self.agentparams:
            imagemain = self.bridge.cv2_to_imgmsg(self.recorder.ltob.img_cropped_medium)
        else:
            imagemain = self.bridge.cv2_to_imgmsg(self.recorder.ltob.img_cropped)
        imageaux1 = self.bridge.cv2_to_imgmsg(np.zeros([self.img_height, self.img_height, 3]))
        state = self.get_endeffector_pos()

        try:
            rospy.wait_for_service('get_action', timeout=3)
            self.desig_pos_main[:,0] = np.clip(self.desig_pos_main[:,0], 0, self.img_height-1)
            self.desig_pos_main[:, 1] = np.clip(self.desig_pos_main[:, 1], 0, self.img_width - 1)
            self.goal_pos_main[:, 0] = np.clip(self.goal_pos_main[:, 0], 0, self.img_height - 1)
            self.goal_pos_main[:, 1] = np.clip(self.goal_pos_main[:, 1], 0, self.img_width - 1)

            get_action_resp = self.get_action_func(imagemain, imageaux1,
                                              tuple(state.astype(np.float32)),
                                              tuple(self.desig_pos_main.flatten()),
                                              tuple(self.goal_pos_main.flatten()))

            action_vec = get_action_resp.action

        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            raise ValueError('get action service call failed')

        action_vec = action_vec[:self.adim]
        return action_vec


    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self.ctrl.limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)


    def move_with_impedance_sec(self, cmd, duration=2.):
        jointnames = self.ctrl.limb.joint_names()
        prev_joint = [self.ctrl.limb.joint_angle(j) for j in jointnames]
        new_joint = np.array([cmd[j] for j in jointnames])

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        while rospy.get_time() < finish_time:
            int_joints = prev_joint + (rospy.get_time()-start_time)/(finish_time-start_time)*(new_joint-prev_joint)
            # print int_joints
            cmd = dict(list(zip(self.ctrl.limb.joint_names(), list(int_joints))))
            self.move_with_impedance(cmd)
            self.control_rate.sleep()

    def set_neutral_with_impedance(self, duration= 1.5):
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(list(zip(self.ctrl.limb.joint_names(), neutral_jointangles)))
        self.imp_ctrl_release_spring(100)
        self.move_with_impedance_sec(cmd, duration)

    def move_to_startpos(self, pos):
        desired_pose = self.get_des_pose(pos)
        start_joints = self.ctrl.limb.joint_angles()
        try:
            des_joint_angles = inverse_kinematics.get_joint_angles(desired_pose, seed_cmd=start_joints,
                                                                   use_advanced_options=True)
        except ValueError:
            rospy.logerr('no inverse kinematics solution found, '
                         'going to reset robot...')
            current_joints = self.ctrl.limb.joint_angles()
            self.ctrl.limb.set_joint_positions(current_joints)
            raise Traj_aborted_except('raising Traj_aborted_except')
        try:
            if self.robot_move:
                if self.use_imp_ctrl:
                    self.imp_ctrl_release_spring(100)
                    self.move_with_impedance_sec(des_joint_angles)
                else:
                    self.ctrl.limb.move_to_joint_positions(des_joint_angles)
        except OSError:
            rospy.logerr('collision detected, stopping trajectory, going to reset robot...')
            rospy.sleep(.5)
            raise Traj_aborted_except('raising Traj_aborted_except')
        if self.ctrl.limb.has_collided():
            rospy.logerr('collision detected!!!')
            rospy.sleep(.5)
            raise Traj_aborted_except('raising Traj_aborted_except')

    def apply_act(self, des_pos, action_vec, i_act):
        # when rotation is enabled
        posshift = action_vec[:2]
        if self.enable_rot:
            up_cmd = action_vec[2]
            delta_rot = action_vec[3]
            close_cmd = action_vec[4]
        # when rotation is not enabled
        else:
            delta_rot = 0.
            close_cmd = action_vec[2]
            up_cmd = action_vec[3]

        des_pos[3] += delta_rot
        des_pos[:2] += posshift

        des_pos = self.truncate_pos(des_pos)  # make sure not outside defined region

        if close_cmd != 0:
            if self.ctrl.sawyer_gripper:
                self.topen = i_act + close_cmd
                self.ctrl.gripper.close()
                self.gripper_closed = True

        if up_cmd != 0:
            self.t_down = i_act + up_cmd
            des_pos[2] = self.lower_height + self.delta_up
            self.gripper_up = True

        if self.gripper_closed:
            if i_act == self.topen:
                self.ctrl.gripper.open()
                print('opening gripper')
                self.gripper_closed = False

        going_down = False
        if self.gripper_up:
            if i_act == self.t_down:
                des_pos[2] = self.lower_height
                print('going down')
                self.gripper_up = False
                going_down = True

        return des_pos, going_down

    def truncate_pos(self, pos):
        xlim = self.xlim
        ylim = self.ylim

        pos[:2] = np.clip(pos[:2], np.array([xlim[0], ylim[0]]), np.array([xlim[1], ylim[1]]))
        if self.enable_rot:
            alpha_min = -0.78539
            alpha_max = np.pi
            pos[3] = np.clip(pos[3], alpha_min, alpha_max)
        return pos

    def redistribute_objects(self):

        self.set_neutral_with_impedance(duration=1.5)
        print('redistribute...')

        file = '/'.join(str.split(python_visual_mpc.__file__, "/")[
                        :-1]) + '/sawyer/visual_mpc_rospkg/src/utils/pushback_traj_{}.pkl'.format(self.robot_name)
        self.joint_pos = pickle.load(open(file, "rb"))

        self.imp_ctrl_release_spring(100)
        self.imp_ctrl_active.publish(1)

        replay_rate = rospy.Rate(700)
        for t in range(len(self.joint_pos)):
            # print 'step {0} joints: {1}'.format(t, self.joint_pos[t])
            replay_rate.sleep()
            self.move_with_impedance(self.joint_pos[t])


class Getdesig(object):
    def __init__(self, img, basedir, img_namesuffix = '', n_desig=1, only_desig = False,
                 im_shape = None, clicks_per_desig=2):
        import matplotlib.pyplot as plt
        plt.switch_backend('qt5agg')
        self.im_shape = im_shape

        self.only_desig = only_desig
        self.n_desig = n_desig
        self.suf = img_namesuffix
        self.basedir = basedir
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)
        plt.imshow(img)

        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_click = 0

        self.desig = np.zeros((n_desig,2))  #idesig, (r,c)
        if clicks_per_desig == 2:
            self.goal = np.zeros((n_desig, 2))  # idesig, (r,c)
        else: self.goal = None

        self.i_click_max = n_desig * clicks_per_desig
        self.clicks_per_desig = clicks_per_desig

        self.i_desig = 0
        self.i_goal = 0
        self.marker_list = ['o',"D","v","^"]

        plt.show()

    def onclick(self, event):
        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        import matplotlib.pyplot as plt
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)

        print('iclick', self.i_click)

        i_task = self.i_click//self.clicks_per_desig
        print('i_task', i_task)


        if self.i_click == self.i_click_max:
            print('saving desig-goal picture')

            with open(self.basedir +'/desig_goal_pix{}.pkl'.format(self.suf), 'wb') as f:
                dict= {'desig_pix': self.desig,
                       'goal_pix': self.goal}
                pickle.dump(dict, f)

            plt.savefig(self.basedir + '/img_' + self.suf)
            print('closing')
            plt.close()
            return

        rc_coord = np.array([event.ydata, event.xdata])

        if self.i_click % self.clicks_per_desig == 0:
            self.desig[i_task, :] = rc_coord
            color = "r"
        else:
            self.goal[i_task, :] = rc_coord
            color = "g"
        marker = self.marker_list[i_task]
        self.ax.scatter(rc_coord[1], rc_coord[0], s=100, marker=marker, facecolors=color)

        plt.draw()

        self.i_click += 1





if __name__ == '__main__':
    mpc = Visual_MPC_Client(cmd_args=True)
