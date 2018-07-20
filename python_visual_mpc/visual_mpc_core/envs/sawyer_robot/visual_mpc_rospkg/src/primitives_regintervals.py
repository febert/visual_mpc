#!/usr/bin/env python
import pickle
import copy
import os
import pdb
from datetime import datetime

import numpy as np
import rospy
from geometry_msgs.msg import (
    Quaternion,
)
from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils import inverse_kinematics, robot_controller, robot_recorder
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_msgs.msg import Int64
from std_msgs.msg import String
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.checkpoint import write_ckpt, write_timing_file, parse_ckpt
import python_visual_mpc
from python_visual_mpc import __file__ as base_filepath
import imp
import xacro

import argparse

class Traj_aborted_except(Exception):
    pass


from wsg_50_common.msg import Cmd, Status


def quat_to_zangle(quat):
    """
    :param quat: quaternion with only
    :return: zangle in rad
    """
    phi = np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
    return np.array([phi])


def zangle_to_quat(zangle):
    quat = Quaternion(  # downward and turn a little
        x=np.cos(zangle / 2),
        y=np.sin(zangle / 2),
        z=0.0,
        w=0.0
    )

    return quat

class Primitive_Executor(object):
    def __init__(self):

        self.num_traj = 50000

        # must be an uneven number
        self.act_every = 4
        self.duration = 20.#24 #16  # duration of trajectory in seconds

        self.state_sequence_length = 96 # number of snapshots that are taken
        self.ctrl = robot_controller.RobotController()

        action_frequency = float(float(self.state_sequence_length)/float(self.duration)/float(self.act_every))
        print('using action frequency of {}Hz'.format(action_frequency))
        print('time between actions:', 1 / action_frequency)

        # self.robot_name = rospy.get_param('~robot')
        # savedir = rospy.get_param('~savedir')
        # if savedir == '':
        self.robot_name = 'vestri'
        print 'robot_name', self.robot_name
        savedir = "/raid/febert/sawyer_data/newrecording"
        # else:
        #     savedir = savedir
        if not os.path.exists(savedir):
            raise ValueError("{} does not exist".format(savedir))

        self.recorder = robot_recorder.RobotRecorder(agent_params={}, save_dir= savedir,
                                                     seq_len=self.state_sequence_length, use_aux=False)

        self.alive_publisher = rospy.Publisher('still_alive', String, queue_size=10)

        self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)

        self.weiss_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
        rospy.Subscriber("/wsg_50_driver/status", Status, self.save_weiss_pos)

        self.control_rate = rospy.Rate(1000)

        rospy.sleep(.2)
        # drive to neutral position:
        self.set_neutral_with_impedance(duration=3)

        rospy.sleep(.2)

        limb = 'right'
        self.name_fksrv = "ExternalTools/" + limb + "/PositionKinematicsNode/FKService"
        self.fksvc = rospy.ServiceProxy(self.name_fksrv, SolvePositionFK)

        self.topen, self.t_down = None, None

        self.use_imp_ctrl = True
        self.robot_move = True
        self.save_active = True
        self.interpolate = True
        self.enable_rot = False

        self.tlast_gripper_status  = rospy.get_time()

        self.checkpoint_file = os.path.join(self.recorder.save_dir, 'checkpoint.txt')

        print('press c to start data collection..')
        pdb.set_trace()
        self.run_data_collection()


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
                except Traj_aborted_except:
                    self.recorder.delete_traj(tr)
                    nfail_traj +=1
                    rospy.sleep(.2)

            if ((tr+1)% 20) == 0:
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

            self.alive_publisher.publish('still alive!')

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
            rospy.wait_for_service(self.name_fksrv, 5)
            resp = self.fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        pos = np.array([resp.pose_stamp[0].pose.position.x,
                         resp.pose_stamp[0].pose.position.y,
                         resp.pose_stamp[0].pose.position.z,
                         ])

        if pos_only:
            return pos
        else:
            quat = np.array([resp.pose_stamp[0].pose.orientation.x,
                             resp.pose_stamp[0].pose.orientation.y,
                             resp.pose_stamp[0].pose.orientation.z,
                             resp.pose_stamp[0].pose.orientation.w
                             ])

            zangle = quat_to_zangle(quat)
            return np.concatenate([pos, zangle])


    def set_weiss_griper(self, width):
        cmd = Cmd()
        cmd.pos = width
        cmd.speed = 100.
        self.weiss_pub.publish(cmd)

    def save_weiss_pos(self, status):
        self.gripper_pos = status.width
        self.tlast_gripper_status = rospy.get_time()

    def run_trajectory(self, i_tr):

        self.imp_ctrl_release_spring(100.)
        self.set_neutral_with_impedance(duration=1.)


        if self.ctrl.sawyer_gripper:
            self.ctrl.gripper.open()
        else:
            self.set_weiss_griper(50.)

        self.gripper_closed = False
        self.gripper_up = False
        if self.save_active:
            self.recorder.init_traj(i_tr)

        self.lower_height = 0.21  # using old gripper : 0.16
        self.delta_up = 0.13
        self.xlim = [0.46, 0.83]  # min, max in cartesian X-direction
        # self.ylim = [-0.20, 0.18]  # min, max in cartesian Y-direction
        self.ylim = [-0.17, 0.17]  # min, max in cartesian Y-direction

        startpos = np.array(
            [np.random.uniform(self.xlim[0], self.xlim[1]), np.random.uniform(self.ylim[0], self.ylim[1])])
        if self.enable_rot:
            start_angle = np.array([np.random.uniform(0., np.pi * 2)])
        else:
            start_angle = np.array([0.])

        self.des_pos = np.concatenate([startpos, np.asarray([self.lower_height + 0.18]), start_angle], axis=0)

        self.topen, self.t_down = 0, 0

        #move to start:
        self.move_to_startpos()
        self.go_down_atstart()

        i_act = 0  # index of current commanded point
        i_save = 0  # index of current saved step

        self.previous_des_pos = copy.deepcopy(self.des_pos)
        num_act = self.state_sequence_length/self.act_every
        self.t_next = rospy.get_time()
        savecount = 4
        while i_save < self.state_sequence_length:
            if i_act > 0:
                if rospy.get_time() > self.tsave_next[i_save]:
                    if self.save_active:
                        print('saving {} at t{}, realtime{}'.format(i_save, self.tsave_next[i_save], rospy.get_time()))
                        self.recorder.save(i_save, action_vec, self.get_endeffector_pos(pos_only=False))
                    # print 'current position error', self.des_pos[:3] - self.get_endeffector_pos(pos_only=True)
                    i_save += 1
                    savecount +=1

            if rospy.get_time() > self.t_next and savecount == self.act_every:
                self.t_prev = rospy.get_time()
                # print 'current position error', self.des_pos[:3] - self.get_endeffector_pos(pos_only=True)

                self.previous_des_pos = copy.deepcopy(self.des_pos)
                action_vec, godown = self.act_joint(i_act)  # after completing trajectory save final state
                # print 'prev_desired pos in step {0}: {1}'.format(i_act, self.previous_des_pos)
                # print 'new desired pos in step {0}: {1}'.format(i_act, self.des_pos)
                # print 'action vec', action_vec
                if godown:
                    act_interval = float(self.duration/num_act)*2.
                else:
                    act_interval = float(self.duration/num_act)

                self.t_next = rospy.get_time() + act_interval

                if i_act == 0:
                    self.tsave_next  = np.linspace(self.t_prev+ act_interval/self.act_every,self.t_next, self.act_every)
                else:
                    self.tsave_next = np.concatenate([self.tsave_next,
                                    np.linspace(self.t_prev+ act_interval/self.act_every,self.t_next, self.act_every)])

                print('i_act', i_act, 't_act_next', self.t_next)

                assert savecount == self.act_every
                savecount = 0

                i_act += 1

            if self.interpolate:
                des_joint_angles = self.get_interpolated_joint_angles()
            try:
                if self.robot_move:
                    if self.use_imp_ctrl:
                        self.move_with_impedance(des_joint_angles)
                    else:
                        self.ctrl.limb.set_joint_positions(des_joint_angles)
                    # print des_joint_angles
            except OSError:
                rospy.logerr('error detected, stopping trajectory, going to reset robot...')
                rospy.sleep(.5)
                raise Traj_aborted_except('raising Traj_aborted_except')



            # print 'sleep'
            self.control_rate.sleep()

        # #saving the final state:
        # if self.save_active:
        #     self.recorder.save(i_save, action_vec, self.get_endeffector_pos(pos_only=False))

        if i_save != self.state_sequence_length:
            print('trajectory not complete!')
            raise Traj_aborted_except()

        self.goup()
        if self.ctrl.sawyer_gripper:
            self.ctrl.gripper.open()
        else:
            # print 'delta t gripper status', rospy.get_time() - self.tlast_gripper_status
            # if rospy.get_time() - self.tlast_gripper_status > 10.:
            #     print 'gripper stopped working!'
            #     pdb.set_trace()

            self.set_weiss_griper(100.)


    def calc_interpolation(self, previous_goalpoint, next_goalpoint, t_prev, t_next):
        """
        interpolate cartesian positions (x,y,z) between last goalpoint and previous goalpoint at the current time
        :param previous_goalpoint:
        :param next_goalpoint:
        :param goto_point:
        :param tnewpos:
        :return: des_pos
        """
        # assert (rospy.get_time() >= t_prev) and (rospy.get_time() <= t_next)
        des_pos = previous_goalpoint + (next_goalpoint - previous_goalpoint) * (rospy.get_time()- t_prev)/ (t_next - t_prev)
        # print 'current_delta_time: ', self.curr_delta_time
        # print "interpolated pos:", des_pos
        return des_pos

    def get_interpolated_joint_angles(self):
        int_des_pos = self.calc_interpolation(self.previous_des_pos, self.des_pos, self.t_prev, self.t_next)
        # print 'interpolated: ', int_des_pos

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
        self.move_with_impedance_sec(des_joint_angles, duration=1.)

    def go_down_atstart(self):
        print("going down at trajectory start..")
        self.des_pos[2] = self.lower_height
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
        self.move_with_impedance_sec(des_joint_angles, duration=3.)

    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

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

        # self.imp_ctrl_release_spring(50)
        while rospy.get_time() < finish_time:
            int_joints = prev_joint + (rospy.get_time()-start_time)/(finish_time-start_time)*(new_joint-prev_joint)
            # print int_joints
            cmd = dict(list(zip(self.ctrl.limb.joint_names(), list(int_joints))))
            self.move_with_impedance(cmd)
            self.control_rate.sleep()

    def set_neutral_with_impedance(self, duration= 1.5):
        neutral_config = np.array([0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076])
        cmd = dict(list(zip(self.ctrl.limb.joint_names(), list(neutral_config))))
        self.move_with_impedance_sec(cmd, duration=duration)

    def move_to_startpos(self):
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
        try:
            if self.robot_move:
                if self.use_imp_ctrl:
                    self.imp_ctrl_release_spring(100)
                    self.move_with_impedance_sec(des_joint_angles, duration=0.9)
                else:
                    self.ctrl.limb.move_to_joint_positions(des_joint_angles)
        except OSError:
            rospy.logerr('error detected, stopping trajectory, going to reset robot...')
            rospy.sleep(.5)
            raise Traj_aborted_except('raising Traj_aborted_except')


    def act_joint(self, i_act):
        """
        only combined actions
        :param i_act:
        :return:
        """
        maxshift = .1
        if self.enable_rot:
            maxrot = np.pi/4
        else:
            maxrot = 0.
        posshift = np.concatenate((np.random.uniform(-maxshift, maxshift, 2), np.zeros(1),
                                   np.random.uniform(-maxrot, maxrot, 1)),
                                  axis=0)

        self.des_pos += posshift
        self.des_pos = self.truncate_pos(self.des_pos)  # make sure not outside defined region

        if self.ctrl.sawyer_gripper:
            close_cmd = np.random.choice(list(range(5)), p=[0.8, 0.05, 0.05, 0.05, 0.05])
            if close_cmd != 0:
                self.topen = i_act + close_cmd
                self.ctrl.gripper.close()
                self.gripper_closed = True
        else:
            close_cmd = 0

        up_cmd = np.random.choice(list(range(5)), p=[0.9, 0.025, 0.025, 0.025, 0.025])
        if up_cmd != 0:
            self.t_down = i_act + up_cmd
            self.des_pos[2] = self.lower_height + self.delta_up
            self.gripper_up = True
            print('going up')

        if self.gripper_closed:
            if i_act == self.topen:
                if self.ctrl.sawyer_gripper:
                    self.ctrl.gripper.open()
                    print('opening gripper')
                    self.gripper_closed = False

        godown = False
        if self.gripper_up:
            if i_act == self.t_down:
                self.des_pos[2] = self.lower_height
                print('going down')
                self.gripper_up = False

                godown = True


        # self.imp_ctrl_release_spring(80.)
        # self.imp_ctrl_release_spring(200.)
        action_vec = np.concatenate([np.array([posshift[0]]),  # movement in plane
                                     np.array([posshift[1]]),  # movement in plane
                                     np.array([up_cmd]),
                                     np.array([posshift[3]]),
                                     np.array([close_cmd])])
        return action_vec, godown

    def get_des_pose(self, des_pos):
        quat = zangle_to_quat(des_pos[3])
        desired_pose = inverse_kinematics.get_pose_stamped(des_pos[0],
                                                           des_pos[1],
                                                           des_pos[2],
                                                           quat)
        return desired_pose


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

        if self.enable_rot:
            alpha_min = -0.78539
            alpha_max = np.pi
            pos[3] = np.clip(pos[3], alpha_min, alpha_max)

        return  pos

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
            print('step {0} joints: {1}'.format(t, self.joint_pos[t]))
            replay_rate.sleep()
            self.move_with_impedance(self.joint_pos[t])

def main():
    pexec = Primitive_Executor()


if __name__ == '__main__':
    main()
