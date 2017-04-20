#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image as Image_msg
import os
import shutil
import socket
import thread
import numpy as np
import pdb
from PIL import Image
import cPickle

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller

class Visual_MPC_Server(object):
    def __init__(self):

        # if it is an auxiliary node advertise services
        rospy.init_node('visual_mpc_server')
        rospy.loginfo("init visual mpc server")

        # initializing the server:
        rospy.Service('get_action', get_action, self.get_action_handler)



        self.cem_controller = CEM_controller()
        rospy.spin()


    def get_action_handler(self):





if __name__ ==  '__main__':
    print 'started'
    rec = Visual_MPC_Server()
