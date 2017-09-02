#!/usr/bin/env python
import numpy as np
from datetime import datetime
import pdb
import rospy
import matplotlib.pyplot as plt

import socket

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)
import intera_external_devices

import argparse
import imutils
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge, CvBridgeError

from PIL import Image
import inverse_kinematics
import robot_controller
from recorder import robot_recorder
import os
import cPickle
from std_msgs.msg import Float32
from std_msgs.msg import Int64

from berkeley_sawyer.srv import *
import copy
import imp

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
    def __init__(self):

        parser = argparse.ArgumentParser(description='Run benchmarks')
        parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
        parser.add_argument('--goalimage', default='False', help='whether to collect goalimages')
        parser.add_argument('--save_subdir', default='False', type=str, help='')
        parser.add_argument('--canon', default=-1, type=int, help='whether to store canonical example')

        args = parser.parse_args()


        self.base_dir = '/'.join(str.split(base_filepath, '/')[:-2])
        cem_exp_dir = self.base_dir + '/experiments/cem_exp/benchmarks_sawyer'
        benchmark_name = args.benchmark
        bench_dir = cem_exp_dir + '/' + benchmark_name
        if not os.path.exists(bench_dir):
            raise ValueError('benchmark directory does not exist')
        bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        self.policyparams = bench_conf.policy
        self.agentparams = bench_conf.agent

        self.benchname = benchmark_name

        self.args = args
        if 'ndesig' in self.policyparams:
            self.ndesig = self.policyparams['ndesig']
        else: self.ndesig = 1

        if args.canon != -1:
            self.save_canon =True
            self.canon_dir = '/home/guser/catkin_ws/src/lsdc/pushing_data/canonical_singleobject'
            self.canon_ind = args.canon
            pdb.set_trace()
        else:
            self.save_canon = False
            self.canon_dir = ''
            self.canon_ind = None

        self.num_traj = 50

        self.action_sequence_length = self.agentparams['T'] # number of snapshots that are taken
        self.use_robot = True
        self.robot_move = True

        self.save_subdir = ""

        self.use_aux = False
        if self.use_robot:
            self.ctrl = robot_controller.RobotController()

        self.get_action_func = rospy.ServiceProxy('get_action', get_action)
        self.init_traj_visual_func = rospy.ServiceProxy('init_traj_visualmpc', init_traj_visualmpc)

        if self.use_robot:
            self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
            self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
            self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)
            self.name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
            self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)

        self.use_imp_ctrl = True
        self.interpolate = True
        self.save_active = True
        self.bridge = CvBridge()

        self.action_interval = 1 #Hz
        self.traj_duration = self.action_sequence_length*self.action_interval
        self.action_rate = rospy.Rate(self.action_interval)
        self.control_rate = rospy.Rate(1000)

        rospy.sleep(.2)
        # drive to neutral position:
        self.imp_ctrl_active.publish(0)
        self.ctrl.set_neutral()
        self.set_neutral_with_impedance()
        self.imp_ctrl_active.publish(1)
        rospy.sleep(.2)

        self.goal_pos_main = np.zeros([2,2])   # the first index is for the ndesig and the second is r,c
        self.desig_pos_main = np.zeros([2, 2])

        if args.goalimage == "True":
            self.use_goalimage = True
        else: self.use_goalimage = False

        self.run_visual_mpc()

    def mark_goal_desig(self, itr):
        print 'prepare to mark goalpos and designated pixel! press c to continue!'
        imagemain = self.recorder.ltob.img_cropped
        imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
        c_main = Getdesig(imagemain, self.desig_pix_img_dir, '_traj{}'.format(itr), self.ndesig, self.canon_ind, self.canon_dir)
        self.desig_pos_main = c_main.desig.astype(np.int64)
        print 'desig pos aux1:', self.desig_pos_main
        self.goal_pos_main = c_main.goal.astype(np.int64)
        print 'goal pos main:', self.goal_pos_main

    def save_canonical(self):
        imagemain = self.recorder.ltob.img_cropped
        imagemain = np.stack([imagemain, imagemain], axis=0)
        state = self.get_endeffector_pos()
        state = np.stack([state, state], axis=0)
        dict = {}
        dict['desig_pix'] = self.desig_pos_main
        dict['goal_pix'] = self.goal_pos_main
        dict['images'] = imagemain
        dict['endeff'] = state
        ex = self.canon_ind
        cPickle.dump(dict, open(self.canon_dir +'/pkl/example{}.pkl'.format(ex), 'wb'))
        print 'saved canonical example to '+ self.canon_dir +'/pkl/example{}.pkl'.format(ex)

    def collect_goal_image(self, ind=0):
        savedir = self.recording_dir + '/goalimage'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        done = False
        print("Press g to take goalimage!")
        while not done and not rospy.is_shutdown():
            c = intera_external_devices.getch()
            if c:
                # catch Esc or ctrl-c
                if c in ['\x1b', '\x03']:
                    done = True
                    rospy.signal_shutdown("Example finished.")
                if c == 'g':
                    print 'taking goalimage'

                    imagemain = self.recorder.ltob.img_cropped

                    cv2.imwrite( savedir+ "/goal_main{}.png".format(ind),
                                imagemain, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
                    state = self.get_endeffector_pos()
                    with open(savedir + '/goalim{}.pkl'.format(ind), 'wb') as f:
                        cPickle.dump({'main': imagemain, 'state': state}, f)
                    break
                else:
                    print 'wrong key!'

        print 'place object in different location!'
        pdb.set_trace()


    def load_goalimage(self, ind):
        savedir = self.recording_dir + '/goalimage'
        with open(savedir + '/goalim{}.pkl'.format(ind), 'rb') as f:
            dict = cPickle.load(f)
            return dict['main'], dict['state']

    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

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
            print 'trajectory {0} took {1} seconds'.format(0, delta.total_seconds())

    def get_endeffector_pos(self, pos_only=True):
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
            rospy.wait_for_service(self.name_of_service, 5)
            resp = self.fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        pos = np.array([resp.pose_stamp[0].pose.position.x,
                         resp.pose_stamp[0].pose.position.y,
                         resp.pose_stamp[0].pose.position.z])
        return pos

    def init_traj(self):
        try:
            # self.recorder.init_traj(itr)
            if self.use_goalimage:
                goal_img_main, goal_state = self.load_goalimage()
                goal_img_aux1 = np.zeros([64, 64, 3])
            else:
                goal_img_main = np.zeros([64, 64, 3])
                goal_img_aux1 = np.zeros([64, 64, 3])

            goal_img_main = self.bridge.cv2_to_imgmsg(goal_img_main)
            goal_img_aux1 = self.bridge.cv2_to_imgmsg(goal_img_aux1)

            rospy.wait_for_service('init_traj_visualmpc', timeout=1)
            self.init_traj_visual_func(0, 0, goal_img_main, goal_img_aux1, self.save_subdir)

        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            raise ValueError('get_kinectdata service failed')

    def run_trajectory(self, i_tr):

        if self.use_robot:
            print 'setting neutral'
            rospy.sleep(.2)
            # drive to neutral position:
            self.imp_ctrl_active.publish(0)
            self.ctrl.set_neutral()
            self.set_neutral_with_impedance()
            self.imp_ctrl_active.publish(1)
            rospy.sleep(.2)

            self.ctrl.gripper.open()
            self.gripper_closed = False
            self.gripper_up = False

            if self.args.save_subdir == "True":
                self.save_subdir = raw_input('enter subdir to save data:')
                self.desig_pix_img_dir = self.base_dir + "/experiments/cem_exp/benchmarks_sawyer/" + self.benchname + \
                                         '/' + self.save_subdir + "/videos"
            else:
                self.desig_pix_img_dir = self.base_dir + "/experiments/cem_exp/benchmarks_sawyer/" + self.benchname + "/videos"
            if not os.path.exists(self.desig_pix_img_dir):
                os.makedirs(self.desig_pix_img_dir)

            num_pic_perstep = 4
            nsave = self.action_sequence_length*num_pic_perstep

            self.recorder = robot_recorder.RobotRecorder(save_dir=self.desig_pix_img_dir,
                                                         seq_len=nsave,
                                                         use_aux=self.use_aux,
                                                         save_video=True,
                                                         save_actions=False,
                                                         save_images=False
                                                         )

            print 'place object in new location!'
            pdb.set_trace()
            # rospy.sleep(.3)
            if self.use_goalimage:
                self.collect_goal_image(i_tr)
            else:
                self.mark_goal_desig(i_tr)

            self.init_traj()

            self.lower_height = 0.20
            self.xlim = [0.44, 0.83]  # min, max in cartesian X-direction
            self.ylim = [-0.27, 0.18]  # min, max in cartesian Y-direction

            random_start_pos = False
            if random_start_pos:
                startpos = np.array([np.random.uniform(self.xlim[0], self.xlim[1]), np.random.uniform(self.ylim[0], self.ylim[1])])
            else: startpos = self.get_endeffector_pos()[:2]

            self.des_pos = np.concatenate([startpos, np.asarray([self.lower_height])], axis=0)

            self.topen, self.t_down = 0, 0

        #move to start:
        self.move_to_startpos(self.des_pos)

        if self.save_canon:
            self.save_canonical()

        # move to start:
        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + self.traj_duration  # in seconds
        print 'start time', start_time
        print 'finish_time', finish_time

        i_step = 0  # index of current commanded point

        self.ctrl.limb.set_joint_position_speed(.20)
        self.previous_des_pos = copy.deepcopy(self.des_pos)
        start_time = -1

        isave = 0

        while i_step < self.action_sequence_length:

            self.curr_delta_time = rospy.get_time() - start_time
            if self.curr_delta_time > self.action_interval:
                if 'manual_correction' in self.agentparams:
                    imagemain = self.recorder.ltob.img_cropped
                    imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
                    c_main = Getdesig(imagemain, self.desig_pix_img_dir, '_t{}'.format(i_step), self.ndesig,
                                      self.canon_ind, self.canon_dir, only_desig=True)
                    self.desig_pos_main = c_main.desig.astype(np.int64)

                print 'current position error', self.des_pos - self.get_endeffector_pos(pos_only=True)

                self.previous_des_pos = copy.deepcopy(self.des_pos)
                action_vec = self.query_action()
                print 'action vec', action_vec

                self.des_pos = self.apply_act(action_vec, i_step, move=False)
                start_time = rospy.get_time()

                print 'prev_desired pos in step {0}: {1}'.format(i_step, self.previous_des_pos)
                print 'new desired pos in step {0}: {1}'.format(i_step, self.des_pos)

                self.t_prev = start_time
                self.t_next = start_time + self.action_interval
                print 't_prev', self.t_prev
                print 't_next', self.t_next

                isave_substep  = 0
                tsave = np.linspace(self.t_prev, self.t_next, num=num_pic_perstep, dtype=np.float64)
                print 'tsave', tsave
                i_step += 1

                print 'applying action{}'.format(i_step)

            des_joint_angles = self.get_interpolated_joint_angles()

            if self.save_active:
                if isave_substep < len(tsave):
                    if rospy.get_time() > tsave[isave_substep] -.01:
                        print 'saving index{}'.format(isave)
                        print 'isave_substep', isave_substep
                        self.recorder.save(isave, action_vec, self.get_endeffector_pos())
                        isave_substep += 1
                        isave += 1

            try:
                if self.robot_move:
                    self.move_with_impedance(des_joint_angles)
                        # print des_joint_angles
            except OSError:
                rospy.logerr('collision detected, stopping trajectory, going to reset robot...')
                rospy.sleep(.5)
                raise Traj_aborted_except('raising Traj_aborted_except')
            if self.ctrl.limb.has_collided():
                rospy.logerr('collision detected!!!')
                rospy.sleep(.5)
                raise Traj_aborted_except('raising Traj_aborted_except')

            self.control_rate.sleep()

        self.save_final_image(i_tr)
        self.recorder.save_highres()

    def save_final_image(self, i_tr):
        imagemain = self.recorder.ltob.img_cropped
        cv2.imwrite(self.desig_pix_img_dir+'/finalimage{}.png'.format(i_tr), imagemain, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

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
        if rospy.get_time() >= t_next:
            des_pos = next_goalpoint
            print 't > tnext'
        # print 'current_delta_time: ', self.curr_delta_time
        # print "interpolated pos:", des_pos
        return des_pos

    def get_interpolated_joint_angles(self):
        int_des_pos = self.calc_interpolation(self.previous_des_pos, self.des_pos, self.t_prev, self.t_next)
        # print 'interpolated: ', int_des_pos
        desired_pose = inverse_kinematics.get_pose_stamped(int_des_pos[0],
                                                           int_des_pos[1],
                                                           int_des_pos[2],
                                                           inverse_kinematics.EXAMPLE_O)
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

    def query_action(self):

        if self.use_robot:
            if self.use_aux:
                self.recorder.get_aux_img()
                imageaux1 = self.recorder.ltob_aux1.img_msg
            else:
                imageaux1 = np.zeros((64, 64, 3), dtype=np.uint8)
                imageaux1 = self.bridge.cv2_to_imgmsg(imageaux1)

            imagemain = self.bridge.cv2_to_imgmsg(self.recorder.ltob.img_cropped)
            state = self.get_endeffector_pos()
        else:
            imagemain = np.zeros((64,64,3))
            imagemain = self.bridge.cv2_to_imgmsg(imagemain)
            imageaux1 = self.bridge.cv2_to_imgmsg(self.test_img)
            state = np.zeros(3)

        try:
            rospy.wait_for_service('get_action', timeout=240)
            get_action_resp = self.get_action_func(imagemain, imageaux1,
                                              tuple(state),
                                              tuple(self.desig_pos_main.flatten()),
                                              tuple(self.goal_pos_main.flatten()))

            action_vec = get_action_resp.action

        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            raise ValueError('get action service call failed')
        return action_vec


    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self.ctrl.limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)

    def move_with_impedance_sec(self, cmd, tsec = 2.):
        """
        blocking
        """
        tstart = rospy.get_time()
        delta_t = 0
        while delta_t < tsec:
            delta_t = rospy.get_time() - tstart
            self.move_with_impedance(cmd)

    def set_neutral_with_impedance(self):
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(zip(self.ctrl.limb.joint_names(), neutral_jointangles))
        self.imp_ctrl_release_spring(20)
        self.move_with_impedance_sec(cmd)

    def move_to_startpos(self, pos):
        desired_pose = inverse_kinematics.get_pose_stamped(pos[0],
                                                           pos[1],
                                                           pos[2],
                                                           inverse_kinematics.EXAMPLE_O)
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
            if self.use_robot:
                if self.use_imp_ctrl:
                    self.imp_ctrl_release_spring(30)
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

    def apply_act(self, action_vec, i_act, move=True):
        self.des_pos[:2] += action_vec[:2]
        self.des_pos = self.truncate_pos(self.des_pos)  # make sure not outside defined region
        self.imp_ctrl_release_spring(120.)

        close_cmd = action_vec[2]
        if close_cmd != 0:
            self.topen = i_act + close_cmd
            self.ctrl.gripper.close()
            self.gripper_closed = True

        up_cmd = action_vec[3]
        delta_up = .1
        if up_cmd != 0:
            self.t_down = i_act + up_cmd
            self.des_pos[2] = self.lower_height + delta_up
            self.gripper_up = True

        if self.gripper_closed:
            if i_act == self.topen:
                self.ctrl.gripper.open()
                print 'opening gripper'
                self.gripper_closed = False

        if self.gripper_up:
            if i_act == self.t_down:
                self.des_pos[2] = self.lower_height
                print 'going down'
                self.imp_ctrl_release_spring(30.)
                self.gripper_up = False

        if move:
            desired_pose = inverse_kinematics.get_pose_stamped(self.des_pos[0],
                                                               self.des_pos[1],
                                                               self.des_pos[2],
                                                               inverse_kinematics.EXAMPLE_O)
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

            self.move_with_impedance(des_joint_angles)

        return self.des_pos

    def truncate_pos(self, pos):

        xlim = self.xlim
        ylim = self.ylim

        if pos[0] > xlim[1]:
            pos[0] = xlim[1]
        if pos[0] < xlim[0]:
            pos[0] = xlim[0]
        if pos[1] > ylim[1]:
            pos[1] = ylim[1]
        if pos[1] < ylim[0]:
            pos[1] = ylim[0]

        return  pos


    def redistribute_objects(self):
        """
        Loops playback of recorded joint position waypoints until program is
        exited
        """
        with open('/home/guser/catkin_ws/src/berkeley_sawyer/src/waypts.pkl', 'r') as f:
            waypoints = cPickle.load(f)
        rospy.loginfo("Waypoint Playback Started")

        # Set joint position speed ratio for execution
        self.ctrl.limb.set_joint_position_speed(.2)

        # Loop until program is exited
        do_repeat = True
        n_repeat = 0
        while do_repeat and (n_repeat < 2):
            do_repeat = False
            n_repeat += 1
            for i, waypoint in enumerate(waypoints):
                if rospy.is_shutdown():
                    break
                try:
                    print 'going to waypoint ', i
                    self.ctrl.limb.move_to_joint_positions(waypoint, timeout=5.0)
                except:
                    do_repeat = True
                    break


class Getdesig(object):
    def __init__(self,img,basedir,img_namesuffix = '', n_desig=1, canon_ind=None, canon_dir = None, only_desig = False):
        self.only_desig = only_desig
        self.canon_ind = canon_ind
        self.canon_dir = canon_dir
        self.n_desig = n_desig
        self.suf = img_namesuffix
        self.basedir = basedir
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_click = 0

        self.desig = np.zeros((2,2))  #idesig, (r,c)
        self.goal = np.zeros((2, 2))  # idesig, (r,c)

        if self.n_desig == 1:
            self.i_click_max = 2

        if self.n_desig == 2:
            self.i_click_max = 4

        if only_desig:
            self.i_click_max = 1

        self.i_desig = 0
        self.i_goal = 0

        plt.show()

    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)

        print 'iclick', self.i_click

        if self.n_desig == 1:
            if self.i_click == 0:
                self.desig[0,:] = np.array([event.ydata, event.xdata])
                self.ax.scatter(self.desig[self.i_click,1], self.desig[self.i_click,0], s=100, marker="D", facecolors='r', edgecolors='r')
                plt.draw()
            if self.i_click == 1 and not self.only_desig:
                self.goal[0,:] = np.array([event.ydata, event.xdata])
                self.ax.scatter(self.goal[0,1], self.goal[0,0], s=100, facecolors='g', edgecolors='g')
                plt.draw()

        if self.n_desig == 2:
            if self.i_click == 0 or self.i_click == 2:
                if self.i_desig == 0:
                    marker = "D"
                else:
                    marker = "o"
                self.desig[self.i_desig,:] = np.array([event.ydata, event.xdata])
                self.ax.scatter(self.desig[self.i_desig,1], self.desig[self.i_desig,0], s=100, marker=marker, facecolors='r', edgecolors='r')
                self.i_desig +=1
                plt.draw()
            if self.i_click == 1 or self.i_click == 3:
                if self.i_goal == 0:
                    marker = "D"
                else:
                    marker = "o"
                self.goal[self.i_goal] = np.array([event.ydata, event.xdata])
                self.ax.scatter(self.goal[self.i_goal,1], self.goal[self.i_goal,0], s=100, facecolors='g', edgecolors='g', marker=marker)
                self.i_goal +=1
                plt.draw()

        if self.i_click == self.i_click_max:
            print 'saving desig-goal picture'
            plt.savefig(self.basedir +'/startimg_'+self.suf)
            if self.canon_ind != None:
                print 'saving canonical example image to' + self.canon_dir + '/images/img{}'.format(self.canon_ind)
                plt.savefig(self.canon_dir + '/images/img{}'.format(self.canon_ind))

            plt.close()
            with open(self.basedir +'/desig_goal_pix{}.pkl'.format(self.suf), 'wb') as f:
                dict= {'desig_pix': self.desig,
                       'goal_pix': self.goal}
                cPickle.dump(dict, f)

        self.i_click += 1

if __name__ == '__main__':
    mpc = Visual_MPC_Client()
