import cv2
import numpy as np

import rospy
import pdb
from std_msgs.msg import Int32MultiArray

class OpenCV_Track_Listener():
    def __init__(self, agentparams, recorder, desig_pos_main):
        """
        :param agentparams:
        :param recorder:
        :param desig_pos_main:

        subscribes to "track_bbox" topic which gives the highres bbox coordinates
        sent by the node in tracker.py
        """

        self.recorder = recorder
        self.adim = agentparams['action_dim']

        self.box_height = 120
        loc = self.low_res_to_highres(desig_pos_main[0])
        bbox = (int(loc[1] - self.box_height / 2.),
                int(loc[0] - self.box_height / 2.),
                self.box_height, self.box_height)  # for the small snow-man

        # bbox = cv2.selectROI(frame, False)
        print 'requesting tracking target: ', bbox

        bbox_pub = rospy.Publisher('tracking_target', Int32MultiArray, queue_size=1)
        intlist = Int32MultiArray()
        intlist.data = list(bbox)
        bbox_pub.publish(intlist)

        rospy.Subscriber("track_bbox", Int32MultiArray, self.store_latest_track)

        self.bbox = None

    def store_latest_track(self, data):
        print "getting latest track"
        self.bbox = np.array(data)

    def get_track(self):
        new_loc = np.array([int(self.bbox[1]), int(self.bbox[0])]) + np.array([float(self.box_height) / 2])
        # Draw bounding box
        return self.high_res_to_lowres(new_loc), new_loc

    def low_res_to_highres(self, inp):
        h = self.recorder.crop_highres_params
        l = self.recorder.crop_lowres_params

        if self.adim == 5:
            highres = (inp + np.array([l['startrow'], l['startcol']])).astype(np.float) / l['shrink_before_crop']
        else:
            orig = (inp + np.array([l['startrow'], l['startcol']])).astype(np.float) / l['shrink_before_crop']
            highres = (orig - np.array([h['startrow'], h['startcol']])) * h['shrink_after_crop']

        highres = highres.astype(np.int64)
        return highres

    def high_res_to_lowres(self, inp):
        h = self.recorder.crop_highres_params
        l = self.recorder.crop_lowres_params

        if self.adim == 5:
            lowres = inp.astype(np.float) * l['shrink_before_crop'] - np.array([l['startrow'], l['startcol']])
        else:
            orig = inp.astype(np.float) / h['shrink_after_crop'] + np.array([h['startrow'], h['startcol']])
            lowres = orig.astype(np.float) * l['shrink_before_crop'] - np.array([l['startrow'], l['startcol']])

        lowres = lowres.astype(np.int64)
        return lowres