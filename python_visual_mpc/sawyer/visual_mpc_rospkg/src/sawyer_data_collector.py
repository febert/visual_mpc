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
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
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

from python_visual_mpc.region_proposal_networks.rpn_tracker import RPN_Tracker
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
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.visual_mpc_client import Visual_MPC_Client

class Sawyer_Data_Collector(Visual_MPC_Client):
    def __init__(self):
        Visual_MPC_Client.__init__(self)
        self.policy = self.policyparams['type'](self.agentparams, self.policyparams)

    def query_action(self, istep):
        return self.policy.act(None, istep)

    def apply_act(self, des_pos, mj_U, i_act):

        if 'discrete_adim' in self.agentparams:
            # when rotation is enabled
            posshift = mj_U[:2]
            if self.enable_rot:
                up_cmd = mj_U[2]
                delta_rot = mj_U[3]
                close_cmd = mj_U[4]
            # when rotation is not enabled
            else:
                delta_rot = 0.
                close_cmd = mj_U[2]
                up_cmd = mj_U[3]

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
        else:
            des_pos = mj_U + des_pos * int(self.agentparams['mode_rel'])
            going_down = False
        return des_pos, going_down


if __name__ == '__main__':
    mpc = Sawyer_Data_Collector()
