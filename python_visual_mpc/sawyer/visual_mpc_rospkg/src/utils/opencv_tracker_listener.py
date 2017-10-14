import cv2
import numpy as np

from visual_mpc_rospkg.srv import bbox
import rospy
import pdb


class OpenCV_Track_Listener():
    def __init__(self, agentparams, recorder, desig_pos_main):
        self.recorder = recorder
        self.adim = agentparams['action_dim']

        frame = recorder.ltob.img_cv2
        self.box_height = 120
        loc = self.low_res_to_highres(desig_pos_main[0])
        bbox = (loc[1] - self.box_height / 2., loc[0] - self.box_height / 2., self.box_height, self.box_height)  # for the small snow-man
        # bbox = cv2.selectROI(frame, False)
        print 'bbox', bbox

        rospy.Subscriber("track_bbox", bbox, self.store_latest_track)

    def store_latest_track(self, data):
        pdb.set_trace()
        self.bbox = np.array(data)

    def get_track_open_cv(self):
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