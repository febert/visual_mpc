import cv2
import numpy as np

class OpenCV_Tracker():
    def __init__(self, agentparams, recorder, desig_pos_main):
        self.recorder = recorder
        self.adim = agentparams['action_dim']

        frame = recorder.ltob.img_cv2
        self.box_height = 120
        loc = self.low_res_to_highres(desig_pos_main[0])
        bbox = (loc[1] - self.box_height / 2., loc[0] - self.box_height / 2., self.box_height, self.box_height)  # for the small snow-man
        # bbox = cv2.selectROI(frame, False)
        print 'bbox', bbox
        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(frame, bbox)

    def track_open_cv(self):

        frame = self.recorder.ltob.img_cv2
        ok, bbox = self.tracker.update(frame)

        new_loc = np.array([int(bbox[1]), int(bbox[0])]) + np.array([float(self.box_height) / 2])
        # Draw bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))

        # Display result
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xff
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