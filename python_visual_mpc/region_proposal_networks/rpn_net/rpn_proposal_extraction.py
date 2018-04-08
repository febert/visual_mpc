

import sys

import cv2
import numpy as np
import tensorflow as tf
sys.path.append('./util/faster_rcnn_lib/')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_proposal_tensorflow, im_detect_tensorflow
import fastrcnn_vgg_net
from python_visual_mpc.region_proposal_networks import rpn_net


model_file = '/home/coline/visual_features/detection/tracking/rpn_net/model/fasterrcnn_vgg_coco_net.tfmodel'#'./model/fasterrcnn_vgg_coco_net.tfmodel'
sess_tuple = None

def _init():
    global sess_tuple

    # Construct the computation graph
    input_batch = tf.placeholder(tf.float32, [1, None, None, 3])
    iminfo_batch = tf.placeholder(tf.float32, [1, 3])
    conv5 = fastrcnn_vgg_net.vgg_conv5(input_batch, 'vgg_net')
    rois, rpn_cls_score, rpn_bbox_pred = rpn_net.rpn_net(conv5, iminfo_batch, 'vgg_net',
                                                         anchor_scales=(4, 8, 16, 32), phase='TEST')

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False)))
    saver = tf.train.Saver()
    saver.restore(sess, model_file)
    sess_tuple = (sess, input_batch, iminfo_batch, rois)
    # sess_tuple = (sess, input_batch, iminfo_batch, rois, rpn_cls_score, rpn_bbox_pred)

def draw_box(box, im):
    x1, y1, x2,y2 = box
    im[y1:y2,x1, 2] = 255
    im[y1, x1:x2, 2] = 255
    im[y1:y2, x2-1,2] = 255
    im[y2-1, x1:x2, 2] = 255
    return im
    
def extract_proposal(self, image):
    boxes = im_proposal_tensorflow(sess_tuple, image)
    return boxes
def extract_proposal(im_file):
    im = cv2.imread(im_file)[::-1,:,::-1]#[:, :, ::-1]
    if sess_tuple is None:
        _init()
    boxes = im_proposal_tensorflow(sess_tuple, im)
    for box in boxes:
        im = draw_box(box, im)
    cv2.imwrite('test.jpg', im)
    return boxes

def detect(im_file):
    im = cv2.imread(im_file)[::-1, :, ::-1]
    if sess_tuple is None:
        _init()
    scores, boxes = im_detect_tensorflow(sess_tuple, im)
    for box in boxes:
        im = draw_box(box, im)
    cv2.imwrite('testdetect.jpg', im)
    return boxes


if __name__ == "__main__":
    # The proposed bounding boxes are in the following format
    im_file = sys.argv[1] if len(sys.argv) > 1 else './image/cat.jpg'
    boxes = extract_proposal(im_file)
    # boxes = detect(im_file)
    print("Extracted proposal boxes from", im_file)
    print(len(boxes))
    print("Each row is a proposal in [xmin, ymin, xmax, ymax] format")
