
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import sys
import pdb

import os

# curr_dir = sys.path[0]
curr_dir = os.path.dirname(os.path.realpath(__file__))

# orig:
# sys.path = [curr_dir, curr_dir+'/rpn_net/util/faster_rcnn_lib', curr_dir+'/rpn_net'] + sys.path[1:]

#new:
sys.path = [curr_dir,
            curr_dir+'/rpn_net/util/faster_rcnn_lib',
            curr_dir+'/rpn_net/util/faster_rcnn_lib/utils',
            curr_dir+'/rpn_net'] + sys.path[1:]

from fast_rcnn.config import cfg
from fast_rcnn.test import im_proposal_tensorflow, im_detect_tensorflow
from rpn_net import  fastrcnn_vgg_net, rpn_net


# from python_visual_mpc.region_proposal_networks.rpn_net.util.faster_rcnn_lib.fast_rcnn.config import cfg
# from python_visual_mpc.region_proposal_networks.rpn_net.util.faster_rcnn_lib.fast_rcnn.test import im_proposal_tensorflow, im_detect_tensorflow
# from python_visual_mpc.region_proposal_networks.rpn_net import fastrcnn_vgg_net, rpn_net

sess_tuple = None

class BBProposer:
    def __init__(self):
        self.model_file = curr_dir + '/rpn_net/model/fasterrcnn_vgg_coco_net.tfmodel'
        global sess_tuple
        # Construct the computation graph
        input_batch = tf.placeholder(tf.float32, [1, None, None, 3])
        iminfo_batch = tf.placeholder(tf.float32, [1, 3])
        conv5 = fastrcnn_vgg_net.vgg_conv5(input_batch, 'vgg_net')
        rois, rpn_cls_score, rpn_bbox_pred = rpn_net.rpn_net(conv5, iminfo_batch, 'vgg_net',
                                                             anchor_scales=(4, 8, 16, 32), phase='TEST')

        var = 3
        print('sess 0 using CUDA_VISIBLE_DEVICES=',var)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(var)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        saver = tf.train.Saver()
        saver.restore(sess, self.model_file)
        sess_tuple = (sess, input_batch, iminfo_batch, rois)

    def draw_box(self, box, im, c):
        x1, y1, x2,y2 = [int(x) for x in box]
        color = np.zeros(3)
        color[c] = 255
        im[y1:y2,x1, :] = color
        im[y1, x1:x2, :] = color
        im[y1:y2, x2-1,:] = color
        im[y2-1, x1:x2, :] = color
        return im

    def get_crop(self, box, im):
        x1, y1, x2,y2 = box
        x1,x2 = self.fix_crop(x1,x2,im.shape[1])
        y1,y2 = self.fix_crop(y1,y2,im.shape[0])
        return im[y1:y2, x1:x2,:]

    def extract_proposal(self, image):
        boxes = im_proposal_tensorflow(sess_tuple, image)
        return boxes

    def fix_crop(self, x1,x2,maxd):
        x1 = int(x1)
        x2 = int(x2)
        while x2-x1 < 28:
            diff = 28-(x2-x1)
            x1 = int(max(0, x1-diff/2 -1))
            x2 = int(min(maxd, x2 +diff/2 +1))
        return x1,x2

from numpy import *
import os
from pylab import *
import numpy as np

import matplotlib.cbook as cbook
import time
from numpy import random


import tensorflow as tf

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
        kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(axis=3, values=output_groups)
    biased =tf.nn.bias_add(conv, biases)
    return biased



class AlexNetFeaturizer:

    def __init__(self):
        net_data = load(curr_dir + "/bvlc_alexnet.npy").item()
        x = tf.placeholder(tf.float32, shape=(None,None, None,3))
        self.input = x
        print(x)
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)


        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)


        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        init = tf.global_variables_initializer()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # self.sess = tf.Session(config=config)

        var = 3
        print('sess1 using CUDA_VISIBLE_DEVICES=', var)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(var)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # self.sess = tf.Session()
        self.sess.run(init)

        self.num_features = 256
        self.out = conv5

    def getFeatures(self,x):
        if True: #x.shape[0] > 28 and x.shape[1] > 28:
            x = np.expand_dims(x, axis=0)
            x = x.astype(np.float32)
            f = self.sess.run(self.out, feed_dict={self.input: x})
            s = f.shape
            f2 = f.reshape((s[0]*s[1]*s[2], s[3]))
            m = np.mean(f2, axis=0)
            n = np.linalg.norm(m)
            return m/n
        else:
            return None

    def getManyFeatures(self,crops):
        shapes = [c.shape for c in crops]
        biggest_x = max([s[0] for s in shapes])
        biggest_y = max([s[1] for s in shapes])
        newcrops = np.zeros((len(crops), biggest_x, biggest_y, 3))
        for c in range(len(crops)):
            newcrops[c,:shapes[c][0], :shapes[c][1],:] = crops[c]
        x = newcrops.astype(np.float32)
        f = self.sess.run(self.out, feed_dict={self.input: x})
        s = f.shape
        feats = []
        for c in range(len(crops)):
            x = int(shapes[c][0]*s[1]/16)
            y = int(shapes[c][1]*s[2]/16)
            thisf = f[c, :x, :y,:].reshape(-1, 256)
            m = np.mean(thisf, axis=0)
            feats.append(m/ np.linalg.norm(m))
        return feats

    def getFullFeatures(self,x):
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        f = self.sess.run(self.out, feed_dict={self.input: x})
        s = f.shape
        print(s)
        f2 = f.reshape((s[0]*s[1]*s[2], s[3]))
        return f2
