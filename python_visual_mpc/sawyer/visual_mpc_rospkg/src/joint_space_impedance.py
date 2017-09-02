#!/usr/bin/env python

# Copyright (c) 2013-2016, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
Intera SDK Joint Torque Example: joint springs
"""

import argparse
import importlib

import rospy
from dynamic_reconfigure.server import Server
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState

import intera_interface
from intera_interface import CHECK_VERSION
import pdb
from std_msgs.msg import Float32
from std_msgs.msg import Int64

class JointSprings(object):
    """
    Virtual Joint Springs class for torque example.

    @param limb: limb on which to run joint springs example
    @param reconfig_server: dynamic reconfigure server

    JointSprings class contains methods for the joint torque example allowing
    moving the limb to a neutral location, entering torque mode, and attaching
    virtual springs.
    """
    def __init__(self, limb = "right"):

        # control parameters
        self._rate = 1000.0  # Hz
        self._missed_cmds = 20.0  # Missed cycles before triggering timeout

        # create our limb instance
        self._limb = intera_interface.Limb(limb)

        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._des_angles = dict()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + limb + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

        rospy.Subscriber("desired_joint_pos", JointState, self._set_des_pos)
        rospy.Subscriber("release_spring", Float32, self._release)
        rospy.Subscriber("imp_ctrl_active", Int64, self._imp_ctrl_active)

        self.max_stiffness = 20
        self.time_to_maxstiffness = .3  ######### 0.68
        self.t_release = rospy.get_time()

        self._imp_ctrl_is_active = True

        for joint in self._limb.joint_names():
            self._springs[joint] = 30
            self._damping[joint] = 4

    def _imp_ctrl_active(self, inp):
        if inp.data == 1:
            print 'impedance ctrl activated'
            self._imp_ctrl_is_active = True
        if inp.data == 0:
            print 'impedance ctrl deactivated'
            self._imp_ctrl_is_active = False

    def _set_des_pos(self, jointstate):
        self._des_angles = dict(zip(jointstate.name, jointstate.position))

    def _release(self, maxstiff):
        maxstiff = maxstiff.data
        self.max_stiffness = float(maxstiff)

        print "setting maxstiffness to", maxstiff
        self.t_release = rospy.get_time()

    def adjust_springs(self):
        for joint in self._des_angles.keys():
            t_delta = rospy.get_time() - self.t_release
            if t_delta > 0:
                if t_delta < self.time_to_maxstiffness:
                    self._springs[joint] = t_delta/self.time_to_maxstiffness * self.max_stiffness
                else:
                    self._springs[joint] = self.max_stiffness
            else:
                print "warning t_delta smaller than zero!"

    def _update_forces(self):
        """
        Calculates the current angular difference between the start position
        and the current joint positions applying the joint torque spring forces
        as defined on the dynamic reconfigure server.
        """

        # print self._springs
        self.adjust_springs()

        # disable cuff interaction
        if self._imp_ctrl_is_active:
            self._pub_cuff_disable.publish()

        # create our command dict
        cmd = dict()
        # record current angles/velocities
        cur_pos = self._limb.joint_angles()
        cur_vel = self._limb.joint_velocities()
        # calculate current forces

        for joint in self._des_angles.keys():
            # spring portion
            cmd[joint] = self._springs[joint] * (self._des_angles[joint] -
                                                 cur_pos[joint])
            # damping portion
            cmd[joint] -= self._damping[joint] * cur_vel[joint]

        # command new joint torques
        if self._imp_ctrl_is_active:
            self._limb.set_joint_torques(cmd)

    def move_to_neutral(self):
        """
        Moves the limb to neutral location.
        """
        self._limb.move_to_neutral()

    def attach_springs(self):
        """
        Switches to joint torque mode and attached joint springs to current
        joint positions.
        """
        # record initial joint angles
        self._des_angles = self._limb.joint_angles()

        # set control rate
        control_rate = rospy.Rate(self._rate)

        # for safety purposes, set the control rate command timeout.
        # if the specified number of command cycles are missed, the robot
        # will timeout and disable
        self._limb.set_command_timeout((1.0 / self._rate) * self._missed_cmds)

        # loop at specified rate commanding new joint torques
        while not rospy.is_shutdown():
            if not self._rs.state().enabled:
                rospy.logerr("Joint torque example failed to meet "
                             "specified control rate timeout.")
                break
            self._update_forces()
            control_rate.sleep()

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()


def main():
    """RSDK Joint Torque Example: Joint Springs

    Moves the default limb to a neutral location and enters
    torque control mode, attaching virtual springs (Hooke's Law)
    to each joint maintaining the start position.

    Run this example and interact by grabbing, pushing, and rotating
    each joint to feel the torques applied that represent the
    virtual springs attached. You can adjust the spring
    constant and damping coefficient for each joint using
    dynamic_reconfigure.
    """
    # Querying the parameter server to determine Robot model and limb name(s)
    rp = intera_interface.RobotParams()
    valid_limbs = rp.get_limb_names()
    if not valid_limbs:
        rp.log_message(("Cannot detect any limb parameters on this robot. "
                        "Exiting."), "ERROR")
    robot_name = intera_interface.RobotParams().get_robot_name().lower().capitalize()
    # Parsing Input Arguments
    arg_fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument(
        "-l", "--limb", dest="limb", default=valid_limbs[0],
        choices=valid_limbs,
        help='limb on which to attach joint springs'
        )
    args = parser.parse_args(rospy.myargv()[1:])
    # Grabbing Robot-specific parameters for Dynamic Reconfigure
    config_name = ''.join([robot_name,"JointSpringsExampleConfig"])
    config_module = "intera_examples.cfg"
    cfg = importlib.import_module('.'.join([config_module,config_name]))
    # Starting node connection to ROS
    print("Initializing node... ")
    rospy.init_node("sdk_joint_torque_springs_{0}".format(args.limb))
    dynamic_cfg_srv = Server(cfg, lambda config, level: config)
    js = JointSprings(limb=args.limb)
    # register shutdown callback
    rospy.on_shutdown(js.clean_shutdown)
    js.attach_springs()


if __name__ == "__main__":
    main()
