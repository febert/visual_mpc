#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image as Image_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import shutil
import socket
import thread
import numpy as np
import pdb
from berkeley_sawyer.srv import *
from PIL import Image
import cPickle
import intera_interface
from intera_interface import CHECK_VERSION
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from std_msgs.msg import Float32
from std_msgs.msg import Int64

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

import intera_interface
import argparse

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)

from recorder.demo_recorder import DemoRobotRecorder

class Pushback_Recorder(object):
    N_SAMPLES = 60
    def __init__(self, file = None):
        """
        Records joint data to a file at a specified rate.
        rate: recording frequency in Hertz
        """

        print("Initializing node... ")
        rospy.init_node("pushback_recorder")

        parser = argparse.ArgumentParser(description='select whehter to record or replay')
        parser.add_argument('--record', type=str, default='True', help='')

        args = parser.parse_args()

        self.rs = intera_interface.RobotEnable(CHECK_VERSION)
        self.init_state = self.rs.state().enabled

        self.limb = intera_interface.Limb("right")
        self.joint_names = self.limb.joint_names()
        self.gripper = intera_interface.Gripper("right")

        self.name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
        self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)

        self._navigator = intera_interface.Navigator()
        self.start_callid = self._navigator.register_callback(self.start_recording, 'right_button_ok')
        # Navigator Rethink button press
        self.stop_callid = self._navigator.register_callback(self.stop_recording, 'right_button_show')

        self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)

        self.gripper.set_velocity(self.gripper.MAX_VELOCITY)  # "set 100% velocity"),
        self.gripper.open()
        self.control_rate = rospy.Rate(20)
        self.imp_ctrl_active.publish(0)

        self.collect_active = False
        rospy.on_shutdown(self.clean_shutdown)
        self.joint_pos = []

        self.set_neutral()
        self.recorder = DemoRobotRecorder('/home/sudeep/outputs', self.N_SAMPLES, use_aux=False)


        if args.record == 'False':
            self.playback(file)
        if args.record == 'True':
            print 'ready for recording!'
            rospy.spin()

        raise ValueError('wrong argument!')

    def set_joints(self, command):
        """Move joints to commmand"""
        self.limb.move_to_joint_positions(command)

    def set_neutral(self, speed = .2):
        # using a custom handpicked neutral position
        # starting from j0 to j6:
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(zip(self.joint_names, neutral_jointangles))

        self.limb.set_joint_position_speed(speed)

        done = False
        while not done:
            try:
                self.set_joints(cmd)
            except:
                print 'retrying set neutral...'

            done = True

        # self.limb.move_to_neutral()

    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self.limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)


    def stop_recording(self, data):
        print 'stopped recording'
        self.collect_active = False
        # self.playback()
        self.clean_shutdown()


    def start_recording(self, data):
        if self.joint_pos != []:
            return
        self.recorder.init_traj(0)
        print 'started recording'


        self.collect_active = True
        self.imp_ctrl_active.publish(0)
        self.joint_pos = []

        iter = 0
        while(self.collect_active):
            self.control_rate.sleep()
            self.joint_pos.append(self.limb.joint_angles())
            pose = self.get_endeffector_pos()
            print 'recording ', len(self.joint_pos)
            self.recorder.save(iter, pose)
            iter += 1

            if (iter >= self.N_SAMPLES):
                self.collect_active = False

        # filename = '/home/sudeep/outputs/pushback_traj_.pkl'
        # with open(filename, 'wb') as f:
        #     cPickle.dump(self.joint_pos, f)

        # print 'saved file to ', filename

    def playback(self, file = None):

        pdb.set_trace()
        self.joint_pos = cPickle.load(open(file, "rb"))

        print 'press c to start playback...'
        pdb.set_trace()
        self.imp_ctrl_release_spring(100)
        self.imp_ctrl_active.publish(1)

        replayrate = rospy.Rate(500)
        for t in range(len(self.joint_pos)):
            print 'step {0} joints: {1}'.format(t, self.joint_pos[t])
            replayrate.sleep()
            self.move_with_impedance(self.joint_pos[t])

    def clean_shutdown(self):
        """
       Switches out of joint torque mode to exit cleanly
       """
        print("\nExiting example...")
        self.limb.exit_control_mode()
        if not self.init_state and self.rs.state().enabled:
            print("Disabling robot...")
            self.rs.disable()
    def get_endeffector_pos(self, pos_only=True):
        """
        :param pos_only: only return postion
        :return:
        """

        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = self.limb.joint_names()
        joints.position = [self.limb.joint_angle(j)
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

if __name__ == '__main__':
    P = Pushback_Recorder('/home/sudeep/outputs/pushback_traj_.pkl')  # playback file

