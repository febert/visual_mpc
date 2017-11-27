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
import python_visual_mpc


class Pushback_Recorder(object):
    def __init__(self):
        """
        Records joint data to a file at a specified rate.
        rate: recording frequency in Hertz
        """

        print("Initializing node... ")
        rospy.init_node("pushback_recorder")

        parser = argparse.ArgumentParser(description='select whehter to record or replay')
        parser.add_argument('robot', type=str, help='robot name')
        parser.add_argument('--record', type=str, default='True', help='')


        args = parser.parse_args()
        self.robotname = args.robot

        self.file = '/'.join(str.split(python_visual_mpc.__file__, "/")[
                        :-1]) + '/sawyer/visual_mpc_rospkg/src/utils/pushback_traj_{}.pkl'.format(self.robotname)

        self.rs = intera_interface.RobotEnable(CHECK_VERSION)
        self.init_state = self.rs.state().enabled

        try:
            self.limb = intera_interface.Limb("right")
            self.gripper = intera_interface.Gripper("right")
            self.gripper.set_velocity(self.gripper.MAX_VELOCITY)  # "set 100% velocity"),
            self.gripper.open()
        except:
            rospy.logerr("no gripper found")

        self._navigator = intera_interface.Navigator()
        self.start_callid = self._navigator.register_callback(self.start_recording, 'right_button_ok')
        # Navigator Rethink button press
        self.stop_callid = self._navigator.register_callback(self.stop_recording, 'right_button_show')

        self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)

        self.control_rate = rospy.Rate(1000)
        self.imp_ctrl_active.publish(0)

        self.collect_active = False
        rospy.on_shutdown(self.clean_shutdown)
        self.joint_pos = []

        if args.record == 'False':
            self.playback()
        if args.record == 'True':
            print 'ready for recording!'
            rospy.spin()

        raise ValueError('wrong argument!')

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

        print 'started recording'
        self.collect_active = True
        self.imp_ctrl_active.publish(0)
        self.joint_pos = []
        while(self.collect_active):
            self.control_rate.sleep()
            self.joint_pos.append(self.limb.joint_angles())
            print 'recording ', len(self.joint_pos)

        with open(self.file, 'wb') as f:
            cPickle.dump(self.joint_pos, f)

        print 'saved file to ', file

    def playback(self):
        self.joint_pos = cPickle.load(open(self.file, "rb"))

        print 'press c to start playback...'
        pdb.set_trace()
        self.imp_ctrl_release_spring(100)
        self.imp_ctrl_active.publish(1)

        replayrate = rospy.Rate(700)
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


if __name__ == '__main__':
    P = Pushback_Recorder()  # playback file

