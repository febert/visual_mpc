#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState

import intera_interface
from intera_interface import CHECK_VERSION
from std_msgs.msg import Float32
from std_msgs.msg import Int64
from python_visual_mpc.visual_mpc_core.envs.util.interpolation import QuinticSpline
from intera_core_msgs.msg import JointCommand
from threading import Lock

NEUTRAL_JOINT_ANGLES = np.array([0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076])
max_vel_mag = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
max_accel_mag = np.array([3.5, 2.5, 5, 5, 5, 5, 5])

class ImpedanceController(object):
    """
    BEFORE RUNNING THIS NODE IN A SEPERATE TERMINAL RUN:
    rosrun intera_examples set_interaction_options.py -k [IMPEDANCE STIFFNESS] -r 10
        - run "rosrun intera_examples set_interaction_option.py" -h for details

    """
    def __init__(self, limb = "right"):
        rospy.init_node("custom_impedance_controller")
        rospy.on_shutdown(self.clean_shutdown)

        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        self._rs.enable()

        if not self._rs.state().enabled:
            raise RuntimeError("Robot did not enable...")

        self._limb = intera_interface.Limb(limb)
        print("Robot enabled...")

        # control parameters
        self._rate = rospy.Rate(800)  # Hz
        self._j_names = [n for n in self._limb.joint_names()]
        self._duration = 1.5
        self._desired_joint_pos = NEUTRAL_JOINT_ANGLES.copy()
        self._fit_interp()

        #synchronizes commands
        self._global_lock = Lock()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + limb + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)
        self._cmd_publisher = rospy.Publisher('/robot/limb/{}/joint_command'.format(limb), JointCommand, queue_size=100)

        print("Running. Ctrl-c to quit")

        rospy.Subscriber("desired_joint_pos", JointState, self._set_des_pos)
        rospy.Subscriber("interp_duration", Float32, self._set_duration)
        rospy.Subscriber("imp_ctrl_active", Int64, self._imp_ctrl_active)

        self._ctrl_active = True
        self._node_running = True

        self._rate.sleep()
        self._start_time = None
        while self._node_running:
            if self._ctrl_active:
                self._pub_cuff_disable.publish()
                self._global_lock.acquire()

                if self._start_time is None:
                    self._start_time = rospy.get_time()
                elif self._start_time + self._duration < rospy.get_time():
                    self._fit_interp()
                    self._start_time = rospy.get_time()

                t = min(rospy.get_time() - self._start_time, self._duration)   #T \in [0, self._duration]
                pos, velocity, accel = [x.squeeze() for x in self._interp.get(t)]

                command = JointCommand()
                command.mode = JointCommand.TRAJECTORY_MODE
                command.names = self._j_names
                command.position = pos
                command.velocity = np.clip(velocity, -max_vel_mag, max_vel_mag)
                command.acceleration = np.clip(accel, -max_accel_mag, max_accel_mag)
                self._cmd_publisher.publish(command)
                self._global_lock.release()

            self._rate.sleep()

    def _get_joints(self):
        return np.array([self._limb.joint_angle(n) for n in self._j_names])

    def _imp_ctrl_active(self, inp):
        if inp.data == 1:
            print('impedance ctrl activated')
            self._ctrl_active = True
        if inp.data == 0:
            print('impedance ctrl deactivated')
            self._ctrl_active = False

    def _fit_interp(self):
        self._interp = QuinticSpline(self._get_joints(), self._desired_joint_pos.copy(), self._duration)

    def _set_des_pos(self, jointstate):
        des_angles = dict(list(zip(jointstate.name, jointstate.position)))
        self._desired_joint_pos = np.array(des_angles[n] for n in self._j_names)

        self._global_lock.acquire()
        self._fit_interp()
        self._start_time = None
        self._global_lock.release()

    def _set_duration(self, duration):
        duration = duration.data
        if duration <= 0:
            print("DURATION CANNOT BE NEGATIVE!!")
            return

        print("setting duration to", duration)

        self._global_lock.acquire()
        self._duration = float(duration)
        self._global_lock.release()




    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        self._node_running = False
        print("\nExiting example...")
        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()




if __name__ == "__main__":
    ImpedanceController()
