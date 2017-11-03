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
    def __init__(self, agentparams, policyparams, recorder, desig_pos_main):
        """
        :param agentparams:
        :param recorder:
        :param desig_pos_main:

        subscribes to "track_bbox" topic which gives the highres bbox coordinates
        sent by the node in tracking_server.py
        """

        self.recorder = recorder
        self.adim = agentparams['adim']

        self.box_height = 120
        loc = self.recorder.low_res_to_highres(desig_pos_main[0])
        bbox1 = np.array([int(loc[1] - self.box_height / 2.),
                int(loc[0] - self.box_height / 2.),
                self.box_height, self.box_height])

        if 'ndesig' in policyparams:
            loc = self.recorder.low_res_to_highres(desig_pos_main[1])
            bbox2 = np.array([int(loc[1] - self.box_height / 2.),
                    int(loc[0] - self.box_height / 2.),
                    self.box_height, self.box_height])
        else:
            bbox2 = np.zeros(4, dtype=np.int)

        # rospy.Subscriber("track_bbox", Int32MultiArray, self.store_latest_track)
        rospy.Subscriber("track_bbox", numpy_msg(intarray), self.store_latest_track)

        self.set_tracking_target_func = rospy.ServiceProxy('set_tracking_target', set_tracking_target)
        print 'requesting tracking target1: ', bbox1
        if 'ndesig' in policyparams:
            print 'requesting tracking target2: ', bbox2
        rospy.wait_for_service('set_tracking_target', timeout=2)

        bbox = np.stack([bbox1, bbox2], axis=0).astype(np.int64)
        pdb.set_trace()
        self.set_tracking_target_func(tuple(bbox.flatten()))

        self.bbox = None

    def store_latest_track(self, data):
        # print "receiving latest track"
        self.bbox = data.data

    def get_track(self):
        new_loc = np.array([int(self.bbox[1]), int(self.bbox[0])]) + np.array([float(self.box_height) / 2])
        # Draw bounding box
        return self.recorder.high_res_to_lowres(new_loc), new_loc