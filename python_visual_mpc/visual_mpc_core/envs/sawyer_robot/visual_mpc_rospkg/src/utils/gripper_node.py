#!/usr/bin/env python
import rospy

from wsg_50_common.srv import Move
from wsg_50_common.msg import Cmd, Status
import pdb

class GripperNode(object):
    def __init__(self):
        print("Initializing gripper node... ")

        rospy.init_node("gripper_node")

        # setup_srv()
        self.start_auto_update()
        rospy.spin()

    def start_auto_update(self):
        rospy.Subscriber("/wsg_50_driver/status", Status, self.ctrl_callback)

        self.pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)

    def ctrl_callback(self, status):
        print('status', status)
        cmd = Cmd()
        cmd.pos = 4.
        cmd.speed = 100.
        self.pub.publish(cmd)

def setup_srv():


    rospy.wait_for_service('/wsg_50_driver/Move', timeout=3.)
    set_pos_func = rospy.ServiceProxy('/wsg_50_driver/Move', Move)
    set_pos_func((50., 10.))  #pos speed

    pdb.set_trace()

    set_pos_func((100., 10.))


if __name__ == '__main__':
    GripperNode()