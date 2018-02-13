import tensorflow as tf
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cPickle
import sys
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn
import matplotlib.gridspec as gridspec

from python_visual_mpc.video_prediction.utils_vpred.online_reader import OnlineReader
import tensorflow.contrib.slim as slim

from python_visual_mpc.utils.colorize_tf import colorize
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.utils_vpred.online_reader import read_trajectory

from python_visual_mpc.data_preparation.gather_data import make_traj_name_list

import collections
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet

class Temp_DnC_GDnet(GoalDistanceNet):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True,
                 images = None,
                 iter_num = None,
                 pred_images = None
                 ):
        GoalDistanceNet.__init__(self, conf = None,
                                 build_loss=True,
                                 load_data = True,
                                 images = None,
                                 iter_num = None,
                                 pred_images = None)

        self.build_net()


        if build_loss:
            self.build_loss()
            self.image_summaries = self.build_image_summary([self.I0, self.I1, self.gen_image_I1, length(self.flow_bwd)])


    def build_net(self):
        n_layer = np.log(self.seq_len)

        used = False
        for l in range(n_layer):
            tstep = tf.pow(2,l)
            for t in range(0, self.seq_len, tstep):
                with tf.variable_scope('warpnet', reuse=used):
                    gen_image, warp_pts, flow_field, h6 = self.warp(self.images[:,t], self.images[:,t +tstep])
                used = True

    def build_loss(self):

        train_summaries = []
        val_summaries = []

        if self.conf['norm'] == 'l2':
            norm = mean_square
        elif self.conf['norm'] == 'charbonnier':
            norm = charbonnier_loss
        else: raise ValueError("norm not defined!")

        self.occ_mask_bwd = 1-self.occ_bwd   # 0 at occlusion
        self.occ_mask_fwd = 1-self.occ_fwd

        self.occ_mask_bwd = self.occ_mask_bwd[:, :, :, None]
        self.occ_mask_fwd = self.occ_mask_fwd[:, :, :, None]

        if 'stop_occ_grad' in self.conf:
            print 'stopping occ mask grads'
            self.occ_mask_bwd = tf.stop_gradient(self.occ_mask_bwd)
            self.occ_mask_fwd = tf.stop_gradient(self.occ_mask_fwd)

        self.loss = 0

        I1_recon_cost = norm((self.gen_image_I1 - self.I1), self.occ_mask_bwd)
        train_summaries.append(tf.summary.scalar('train_I1_recon_cost', I1_recon_cost))
        self.loss += I1_recon_cost

        if 'fwd_bwd' in self.conf:
            I0_recon_cost = norm((self.gen_image_I0 - self.I0), self.occ_mask_fwd)
            train_summaries.append(tf.summary.scalar('train_I0_recon_cost', I0_recon_cost))
            self.loss += I0_recon_cost

            fd = self.conf['flow_diff_cost']
            flow_diff_cost = (norm(self.diff_flow_fwd, self.occ_mask_fwd)
                              + norm(self.diff_flow_bwd, self.occ_mask_bwd)) * fd
            train_summaries.append(tf.summary.scalar('train_flow_diff_cost', flow_diff_cost))
            self.loss += flow_diff_cost

            if 'occlusion_handling' in self.conf:
                occ = self.conf['occlusion_handling']
                occ_reg_cost = (tf.reduce_mean(self.occ_fwd) + tf.reduce_mean(self.occ_bwd)) * occ
                train_summaries.append(tf.summary.scalar('train_occlusion_handling', occ_reg_cost))
                self.loss += occ_reg_cost

        if 'smoothcost' in self.conf:
            sc = self.conf['smoothcost']
            smooth_cost_bwd = flow_smooth_cost(self.flow_bwd, norm, self.conf['smoothmode'],
                                               self.occ_mask_bwd) * sc
            train_summaries.append(tf.summary.scalar('train_smooth_bwd', smooth_cost_bwd))
            self.loss += smooth_cost_bwd

            if 'fwd_bwd' in self.conf:
                smooth_cost_fwd = flow_smooth_cost(self.flow_fwd, norm, self.conf['smoothmode'],
                                                   self.occ_mask_fwd) * sc
                train_summaries.append(tf.summary.scalar('train_smooth_fwd', smooth_cost_fwd))
                self.loss += smooth_cost_fwd

        if 'flow_penal' in self.conf:
            flow_penal = (tf.reduce_mean(tf.square(self.flow_bwd)) +
                          tf.reduce_mean(tf.square(self.flow_fwd))) * self.conf['flow_penal']
            train_summaries.append(tf.summary.scalar('flow_penal', flow_penal))
            self.loss += flow_penal


        train_summaries.append(tf.summary.scalar('train_total', self.loss))
        val_summaries.append(tf.summary.scalar('val_total', self.loss))

        self.train_summ_op = tf.summary.merge(train_summaries)
        self.val_summ_op = tf.summary.merge(val_summaries)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


def calc_warpscores(flow_field):
    return np.sum(np.linalg.norm(flow_field, axis=3), axis=[2, 3])


if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/hardthres/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')