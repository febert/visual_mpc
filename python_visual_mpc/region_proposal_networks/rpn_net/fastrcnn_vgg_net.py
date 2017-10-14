from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# components
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.rpn import ProposalLayer

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def vgg_conv5(input_batch, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        # layer 1
        conv1_1 = conv_relu('conv1_1', input_batch,
                            kernel_size=3, stride=1, output_dim=64)
        conv1_2 = conv_relu('conv1_2', conv1_1,
                            kernel_size=3, stride=1, output_dim=64)
        pool1 = pool('pool1', conv1_2, kernel_size=2, stride=2)
        # layer 2
        conv2_1 = conv_relu('conv2_1', pool1,
                            kernel_size=3, stride=1, output_dim=128)
        conv2_2 = conv_relu('conv2_2', conv2_1,
                            kernel_size=3, stride=1, output_dim=128)
        pool2 = pool('pool2', conv2_2, kernel_size=2, stride=2)
        # layer 3
        conv3_1 = conv_relu('conv3_1', pool2,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_2 = conv_relu('conv3_2', conv3_1,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_3 = conv_relu('conv3_3', conv3_2,
                            kernel_size=3, stride=1, output_dim=256)
        pool3 = pool('pool3', conv3_3, kernel_size=2, stride=2)
        # layer 4
        conv4_1 = conv_relu('conv4_1', pool3,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_2 = conv_relu('conv4_2', conv4_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_3 = conv_relu('conv4_3', conv4_2,
                            kernel_size=3, stride=1, output_dim=512)
        pool4 = pool('pool4', conv4_3, kernel_size=2, stride=2)
        # layer 5
        conv5_1 = conv_relu('conv5_1', pool4,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_2 = conv_relu('conv5_2', conv5_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_3 = conv_relu('conv5_3', conv5_2,
                            kernel_size=3, stride=1, output_dim=512)
    return conv5_3

def rpn_net(conv5, im_info, name, feat_stride=16,
    anchor_scales=(8, 16, 32), phase='TEST'):
    with tf.variable_scope(name):
        # rpn_conv/3x3
        rpn_conv = conv_relu('rpn_conv/3x3', conv5, kernel_size=3, stride=1,
                             output_dim=512)
        # rpn_cls_score
        # Note that we've already subtracted the bg weights from fg weights
        # and do sigmoid instead of softmax (actually sigmoid is not needed
        # for ranking)
        rpn_cls_score = conv('rpn_cls_score', rpn_conv, kernel_size=1, stride=1,
                             output_dim=len(anchor_scales)*3)
        # rpn_bbox_pred
        rpn_bbox_pred = conv('rpn_bbox_pred', rpn_conv, kernel_size=1, stride=1,
                             output_dim=len(anchor_scales)*3*4)

        rois = tf.py_func(ProposalLayer(feat_stride, anchor_scales, phase),
                          [rpn_cls_score, rpn_bbox_pred, im_info],
                          [tf.float32], stateful=False)[0]
        rois.set_shape([None, 5])
        return rois

