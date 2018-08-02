import cv2
import numpy as np
from visual_mpc_rospkg.srv import set_tracking_target
import rospy
import pdb
from std_msgs.msg import Int32MultiArray

from rospy_tutorials.msg import Floats
from visual_mpc_rospkg.msg import intarray
from rospy.numpy_msg import numpy_msg

class OpenCV_Track_Listener():
    def __init__(self, agentparams, recorder, desig_pos_main):
        """
        :param agentparams:
        :param recorder:
        :param desig_pos_main:

        subscribes to "track_bbox" topic which gives the highres bbox coordinates
        sent by the node in tracking_server.py
        """

        self.recorder = recorder
        self.adim = agentparams['adim']
        self.ndesig = agentparams['ndesig']

        self.box_height = 120
        bbox = np.zeros([self.ndesig, 4])

        # bbox format: col, row of upper left corner, box height, box width
        for p in range(self.ndesig):
            loc = self.recorder.low_res_to_highres(desig_pos_main[p])
            bbox[p] = np.array([int(loc[1] - self.box_height / 2.),
                                int(loc[0] - self.box_height / 2.),
                                self.box_height, self.box_height])

        # rospy.Subscriber("track_bbox", Int32MultiArray, self.store_latest_track)
        rospy.Subscriber("track_bbox", numpy_msg(intarray), self.store_latest_track)

        self.set_tracking_target_func = rospy.ServiceProxy('set_tracking_target', set_tracking_target)
        print('requesting tracking target1: ', bbox)
        rospy.wait_for_service('set_tracking_target', timeout=2)

        self.set_tracking_target_func(tuple(bbox.flatten()))
        self.rec_bbox = None

    def store_latest_track(self, data):
        # print "receiving latest track"
        self.rec_bbox = data.data.reshape(self.ndesig, 4)

    def get_track(self):
        new_desig_pos = np.zeros([self.ndesig,2])
        new_highres_pos = np.zeros([self.ndesig,2])

        for p in range(self.ndesig):
            new_loc = np.array([int(self.rec_bbox[p, 1]), int(self.rec_bbox[p, 0])]) + np.array([self.box_height/2, self.box_height/2])
            new_highres_pos[p] = new_loc
            new_desig_pos[p] = self.recorder.high_res_to_lowres(new_loc)

        return new_desig_pos, new_highres_pos