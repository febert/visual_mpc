import rospy
import numpy as np
import pickle as pkl
import cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg
import copy

import imutils

rospy.init_node("sawyer_custom_controller")
bridge = CvBridge()



def crop_highres(cv_image):
    startcol = 180
    startrow = 0
    endcol = startcol + 1500
    endrow = startrow + 1500
    cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])

    cv_image = imutils.rotate_bound(cv_image, 180)
    cv_image = cv2.resize(cv_image, (0, 0), fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
    return cv_image

fig = None
def plot_im(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")  # (1920, 1080)

    img = crop_highres(cv_image)
    try:
        fig.close()
    except:
        pass
    fig = plt.figure()
    plt.imshow(img[:, :, ::-1])
    plt.scatter(pnts[:, 1], pnts[:, 0])
    plt.show()

d = pkl.load(open('points.pkl', 'rb'))
pnts = d['pnts_kinect']

rospy.Subscriber("/kinect2/hd/image_color", Image_msg, plot_im)
rospy.spin()