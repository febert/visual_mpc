#!/usr/bin/env python
import numpy as np
import rospy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_wsg_controller import WSGRobotController
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_dualcam_recorder import RobotDualCamRecorder, Trajectory
CONTROL_RATE = 1000

class RobotEnvironment:
    def __init__(self):
        test_agent_params = {'T' : 15, 'image_height' : 48, 'image_width' : 64, 'data_conf' : {},
                            'adim' : 5, 'sdim' : 5, 'mode_rel' : np.array([True, True, True, True, False]),
                             'targetpos_clip':[[-0.5, -0.5, -0.08, -2 * np.pi, -1], [0.5, 0.5, 0.15, 2 * np.pi, 1]]}
        #initializes node and creates interface with Sawyer
        self._controller = WSGRobotController()
        self._recorder = RobotDualCamRecorder(test_agent_params, self._controller)

        self.control_rate = rospy.Rate(CONTROL_RATE)

        test_traj = Trajectory(test_agent_params)

        for t in range(test_traj.sequence_length):
            if not self._recorder.store_recordings(test_traj, t):
                print('{} is bad!'.format(t))
            self.control_rate.sleep()

        test_traj.save_traj('test_traj')
        print(test_traj.joint_angles)

if __name__ == '__main__':
    env = RobotEnvironment()