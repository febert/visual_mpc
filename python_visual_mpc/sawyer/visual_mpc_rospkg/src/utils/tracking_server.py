#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int64
from sensor_msgs.msg import Image as Image_msg
import imutils
import cv2
import numpy as np

from rospy.numpy_msg import numpy_msg
from visual_mpc_rospkg.msg import intarray
from visual_mpc_rospkg.msg import bbox

from visual_mpc_rospkg.srv import set_tracking_target, set_tracking_targetResponse
import pdb

class Tracker(object):
    def __init__(self):
        print "Initializing node... "
        rospy.init_node("opencv_tracker")
        rospy.Subscriber("main/kinect2/hd/image_color", Image_msg, self.store_latest_im)

        self.tracker_initialized = False
        self.bbox = None

        rospy.Subscriber("main/kinect2/hd/image_color", Image_msg, self.store_latest_im)

        rospy.Service('set_tracking_target', set_tracking_target, self.init_bbx)

        # self.bbox_pub = rospy.Publisher('track_bbox', Int32MultiArray, queue_size=1)
        self.bbox_pub = rospy.Publisher('track_bbox', numpy_msg(intarray), queue_size=10)
        self.bridge = CvBridge()

    def init_bbx(self, req):
        target = req.target
        print "received new tracking target", target
        self.tracker_initialized = False
        self.bbox = target
        resp = set_tracking_targetResponse()
        return resp

    def store_latest_im(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  #(1920, 1080)
        self.lt_img_cv2 = self.crop_highres(cv_image)

        if not self.tracker_initialized and self.bbox is not None:
            self.cv_tracker = cv2.TrackerMIL_create()
            self.cv_tracker.init(self.lt_img_cv2, self.bbox)
            self.tracker_initialized = True
            print 'tracker initialized'

    def crop_highres(self, cv_image):
        #TODO: load the cropping parameters from parameter file
        startcol = 180
        startrow = 0
        endcol = startcol + 1500
        endrow = startrow + 1500
        cv_image = cv_image[startrow:endrow, startcol:endcol]
        shrink_after_crop = .75
        cv_image = cv2.resize(cv_image, (0, 0), fx=shrink_after_crop, fy=shrink_after_crop,
                              interpolation=cv2.INTER_AREA)
        cv_image = imutils.rotate_bound(cv_image, 180)
        return cv_image


    def start(self):
        while not rospy.is_shutdown():
            if self.tracker_initialized:
                ok, bbox = self.cv_tracker.update(self.lt_img_cv2)
                bbox = np.array(bbox).astype(np.int32)
                self.bbox = bbox

                p1 = (bbox[0], bbox[1])
                p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                cv2.rectangle(self.lt_img_cv2, p1, p2, (0, 0, 255))

                # Display result
                cv2.imshow("Tracking", self.lt_img_cv2)
                k = cv2.waitKey(1) & 0xff


                self.bbox_pub.publish(bbox)
                print 'currrent bbox: ',bbox
            else:
                rospy.sleep(0.1)
                print "waiting for tracker to be initialized, bbox=", self.bbox

if __name__ == "__main__":
    r = Tracker()
    r.start()