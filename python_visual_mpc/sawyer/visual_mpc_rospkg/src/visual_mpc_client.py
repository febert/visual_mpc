#!/usr/bin/env python
import numpy as np
from datetime import datetime
import pdb
import rospy
import matplotlib.pyplot as plt

import socket
if socket.gethostname() == 'kullback':
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


class Traj_aborted_except(Exception):
    pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Visual_MPC_Client():
    def __init__(self):

        parser = argparse.ArgumentParser(description='Run benchmarks')
        parser.add_argument('--goalimage', action='store_true', help='whether to collect goalimages')
        parser.add_argument('--steps', default=15, type=int, help='how many steps to execute')
        parser.add_argument('--imgdir', default="", type=str, help='direcotry where to store images of designated pixels')

        args = parser.parse_args()
        self.num_traj = 50

        self.action_sequence_length = args.steps # number of snapshots that are taken
        self.use_robot = True
        # self.base_dir = "/home/frederik/Documents/berkeley_sawyer/src/testbasedir"
        self.recording_dir = "/home/guser/catkin_ws/src/berkeley_sawyer/src/recordings"

        if args.imgdir != "":
            self.desig_pix_img_dir = "/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/" + args.imgdir +"/videos"
        else: self.desig_pix_img_dir = self.recording_dir

        if self.use_robot:
            self.ctrl = robot_controller.RobotController()
            # self.base_dir ="/home/guser/sawyer_data/visual_mpc"
            self.recorder = robot_recorder.RobotRecorder(save_dir=self.recording_dir,
                                                         seq_len=self.action_sequence_length)

        # drive to neutral position:
        ################# self.ctrl.set_neutral()
        self.get_action_func = rospy.ServiceProxy('get_action', get_action)
        self.init_traj_visual_func = rospy.ServiceProxy('init_traj_visualmpc', init_traj_visualmpc)

        if self.use_robot:
            self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
            self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
            self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)
            self.name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
            self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)

        self.use_imp_ctrl = True
        self.interpolate = False
        self.save_active = False
        self.bridge = CvBridge()

        self.max_exec_rate = rospy.Rate(1)

        rospy.sleep(.2)
        # drive to neutral position:
        self.imp_ctrl_active.publish(0)
        self.ctrl.set_neutral()
        self.set_neutral_with_impedance()
        self.imp_ctrl_active.publish(1)
        rospy.sleep(.2)

        self.desig_pos_main = np.zeros(2)
        self.goal_pos_main = np.zeros(2)

        if args.goalimage == "True":
            self.use_goalimage = True
        else: self.use_goalimage = False
        self.run_visual_mpc()

    def mark_goal_desig(self, itr):
        print 'prepare to mark goalpos and designated pixel! press c to continue!'
        pdb.set_trace()
        if not self.use_robot:
            i = 1
            img = Image.open(self.recording_dir + '/{}.png'.format(i))
            img = img.rotate(180)
            self.test_img = np.asarray(img)
            self.test_img = np.concatenate([self.test_img, np.zeros((64,64,3))], axis=0)
            c_main = Getdesig(self.test_img, self.recording_dir, 'b{}'.format(i))
        else:
            imagemain = self.recorder.ltob.img_cropped
            assert imagemain != None
            imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
            c_main = Getdesig(imagemain, self.desig_pix_img_dir, '_traj{}'.format(itr))
            # self.recorder.get_aux_img()
            # imageaux1 = self.bridge.imgmsg_to_cv2(self.recorder.ltob_aux1.img_msg)
            # imageaux1 = cv2.cvtColor(imageaux1, cv2.COLOR_BGR2RGB)
            # # self.bridge.imgmsg_to_cv2(data, "bgr8")  # (1920, 1080)
            # c_aux1 = Getdesig(imageaux1, self.base_dir)

        self.desig_pos_main = c_main.desig.astype(np.int32)
        print 'desig pos aux1:', self.desig_pos_main
        self.goal_pos_main = c_main.goal.astype(np.int32)
        print 'goal pos main:', self.goal_pos_main

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

        # check if there is a checkpoint from which to resume
        start_tr = 0

        for tr in range(start_tr, self.num_traj):

            tstart = datetime.now()
            # self.run_trajectory_const_speed(tr)
            done = False
            while not done:
                try:
                    self.run_trajectory(tr)
                    done = True
                except Traj_aborted_except:
                    self.recorder.delete_traj(tr)

            delta = datetime.now() - tstart
            print 'trajectory {0} took {1} seconds'.format(tr, delta.total_seconds())

            if (tr% 30) == 0 and tr!= 0:
                self.redistribute_objects()

        self.ctrl.set_neutral()


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

    def init_traj(self, itr):
        try:
            self.recorder.init_traj(itr)
            rospy.wait_for_service('init_traj', timeout=1)
            if self.use_goalimage:
                goal_img_main, goal_state = self.load_goalimage(itr)
                goal_img_aux1 = np.zeros([64, 64, 3])
            else:
                goal_img_main = np.zeros([64, 64, 3])
                goal_img_aux1 = np.zeros([64, 64, 3])

            goal_img_main = self.bridge.cv2_to_imgmsg(goal_img_main)
            goal_img_aux1 = self.bridge.cv2_to_imgmsg(goal_img_aux1)
            rospy.wait_for_service('init_traj_visualmpc', timeout=1)
            resp = self.init_traj_visual_func(itr, 0, goal_img_main, goal_img_aux1)

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

            if self.use_goalimage:
                self.collect_goal_image(i_tr)
            else:
                self.mark_goal_desig(i_tr)

            self.init_traj(i_tr)

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



        i_save = 0  # index of current saved step
        for i_act in range(self.action_sequence_length):

            action_vec = self.query_action()
            self.apply_act(action_vec, i_act)

            if self.save_active:
                self.recorder.save(i_save, action_vec, self.get_endeffector_pos())
                i_save += 1

            self.max_exec_rate.sleep()

    def query_action(self):
        if self.use_robot:
            self.recorder.get_aux_img()
            imagemain = self.bridge.cv2_to_imgmsg(self.recorder.ltob.img_cropped)
            imageaux1 = self.recorder.ltob_aux1.img_msg
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
                                              tuple(self.desig_pos_main),
                                              tuple(self.goal_pos_main))

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

    def apply_act(self, action_vec, i_act):
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
    def __init__(self,img,basedir,img_namesuffix = ''):
        self.suf = img_namesuffix
        self.basedir = basedir
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        self.desig = None
        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_click = 0
        plt.show()

    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)

        self.i_click += 1

        if self.i_click == 1:

            self.desig = np.array([event.ydata, event.xdata])
            self.ax.scatter(self.desig[1], self.desig[0], s=60, facecolors='none', edgecolors='b')
            plt.draw()
        elif self.i_click == 2:
            self.goal = np.array([event.ydata, event.xdata])
            self.ax.scatter(self.goal[1], self.goal[0], s=60, facecolors='none', edgecolors='g')
            plt.draw()

        else:
            plt.savefig(self.basedir +'/img_desigpix'+self.suf)
            plt.close()

if __name__ == '__main__':
    mpc = Visual_MPC_Client()
