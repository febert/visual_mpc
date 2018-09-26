import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet
from python_visual_mpc.goaldistancenet.utils.ops import conv2d, length
from python_visual_mpc.goaldistancenet.gdnet import apply_warp

def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)

class GoalDistanceNetSlowness(GoalDistanceNet):

    def build_net(self):
        if 'fwd_bwd' in self.conf:
            self.build_net_fwdbwd()
        else:
            self.build_net_bwd()

    def build_net_bwd(self):
        with tf.variable_scope('warpnet'):
            self.warped_I0_to_I1, self.warp_pts_bwd, flow_bwd_01, h6_bwd = self.warp(self.I0, self.I1)

        self.gen_I1 = self.warped_I0_to_I1

        if self.build_loss:  # add warping for temporal slowness constraint
            self.add_pair_loss(I1=self.I1, gen_I1=self.gen_I1, flow_bwd=flow_bwd_01)

            with tf.variable_scope('warpnet', reuse=True):
                warped_I0_to_I2, _, flow_bwd_02, _ = self.warp(self.I0, self.I2)


            self.losses['slowness'] = self.conf['slowness_penal'] * tf.reduce_mean(tf.square(flow_bwd_02 - flow_bwd_01))

            self.combine_losses()

            self.add_pair_loss(I1=self.I1, gen_I1=self.gen_I1,  flow_bwd=flow_bwd_01)
            image_summaries = self.build_image_summary([self.I0, self.I1, self.gen_I1, length(flow_bwd_01)])


            self.add_pair_loss(I1=self.I2, gen_I1=warped_I0_to_I2, flow_bwd=flow_bwd_02)
            image_summaries_2 = self.build_image_summary([self.I0,self.I2, warped_I0_to_I2, length(flow_bwd_02)], suf='image2')

            self.image_summaries = tf.summary.merge([image_summaries, image_summaries_2])

    def build_net_fwdbwd(self):
        with tf.variable_scope('warpnet'):
            self.warped_I0_to_I1, self.warp_pts_bwd, flow_bwd_01, h6_bwd = self.warp(self.I0, self.I1)
        with tf.variable_scope('warpnet', reuse=True):
            self.warped_I1_to_I0, self.warp_pts_fwd, flow_fwd_10, h6_fwd = self.warp(self.I1, self.I0)

        self.gen_I1 = self.warped_I0_to_I1
        self.gen_I0 = self.warped_I1_to_I0

        occ_bwd, occ_fwd, diff_flow_bwd, diff_flow_fwd = self.compute_occ(flow_bwd_01, flow_fwd_10)

        if self.build_loss:  # add warping for temporal slowness constraint
            self.add_pair_loss(I1=self.I1, gen_I1=self.gen_I1, occ_bwd=occ_bwd, flow_bwd=flow_bwd_01,
                               diff_flow_bwd=diff_flow_bwd,
                               I0=self.I0, gen_I0=self.gen_I0, occ_fwd=occ_fwd, flow_fwd=flow_fwd_10,
                               diff_flow_fwd=diff_flow_fwd)

            with tf.variable_scope('warpnet', reuse=True):
                warped_I0_to_I2, _, flow_bwd_02, _ = self.warp(self.I0, self.I2)
            with tf.variable_scope('warpnet', reuse=True):
                warped_I2_to_I0, _, flow_fwd_20, _ = self.warp(self.I2, self.I0)

            occ_bwd_02, occ_fwd_20, diff_flow_bwd_02, diff_flow_fwd_20 = self.compute_occ(flow_bwd_02, flow_fwd_20)
            self.add_pair_loss(I1=self.I2, gen_I1=warped_I0_to_I2, occ_bwd=occ_bwd_02, flow_bwd=flow_bwd_02,
                               diff_flow_bwd=diff_flow_bwd_02,
                               I0=self.I0, gen_I0=warped_I2_to_I0, occ_fwd=occ_fwd_20, flow_fwd=flow_fwd_20,
                               diff_flow_fwd=diff_flow_fwd_20, suf='_20')

            self.losses['slowness'] = self.conf['slowness_penal'] * (tf.reduce_mean(tf.square(flow_bwd_02 - flow_bwd_01)) +
                                                                     tf.reduce_mean(tf.square(flow_fwd_20 - flow_fwd_10)))

            self.combine_losses()
            # image_summary:
            image_summaries = self.build_image_summary(
                [self.I0, self.I1, self.gen_I0, self.gen_I1, length(flow_bwd_01), length(flow_fwd_10),
                 1-occ_bwd, 1-occ_fwd])

            image_summaries_2 = self.build_image_summary(
                [self.I0, self.I2, warped_I2_to_I0, warped_I0_to_I2, length(flow_bwd_02), length(flow_fwd_20),
                 1-occ_bwd_02, 1-occ_fwd_20], name='image_2')

            self.image_summaries = tf.summary.merge([image_summaries, image_summaries_2])


    def compute_occ(self, flow_bwd, flow_fwd):
        bwd_flow_warped_fwd, _ = apply_warp(flow_bwd, flow_fwd)
        diff_flow_fwd = flow_fwd + bwd_flow_warped_fwd
        fwd_flow_warped_bwd, _ = apply_warp(flow_fwd, flow_bwd)

        diff_flow_bwd = flow_bwd + fwd_flow_warped_bwd
        mag_sq = length_sq(flow_fwd) + length_sq(flow_bwd)
        occ_thres_mult = 0.01
        occ_thres_offset = 0.5
        occ_thresh = occ_thres_mult * mag_sq + occ_thres_offset
        occ_fwd = tf.cast(length_sq(diff_flow_fwd) > occ_thresh, tf.float32)
        occ_fwd = tf.reshape(occ_fwd, [self.bsize, self.img_height, self.img_width, 1])
        occ_bwd = tf.cast(length_sq(diff_flow_bwd) > occ_thresh, tf.float32)
        occ_bwd = tf.reshape(occ_bwd, [self.bsize, self.img_height, self.img_width, 1])
        return occ_bwd, occ_fwd, diff_flow_bwd, diff_flow_fwd

    def sel_images(self):

        max_deltat = self.conf['max_deltat']
        t_fullrange = 2e4
        delta_t = tf.cast(tf.ceil(max_deltat * (tf.cast(self.iter_num + 1, tf.float32)) / t_fullrange), dtype=tf.int32)
        delta_t = tf.clip_by_value(delta_t, 1, max_deltat - 1)
        self.delta_t = delta_t

        self.tstart = tf.random_uniform([1], 0, self.conf['sequence_length'] - delta_t- 1, dtype=tf.int32)

        if 'deterministic_increase_tdist' in self.conf:
            self.tend = self.tstart + delta_t
        else:
            minval = tf.ones([], dtype=tf.int32)
            self.tend = self.tstart + tf.random_uniform([1], minval, delta_t + 1, dtype=tf.int32)

        begin = tf.stack([0, tf.squeeze(self.tstart), 0, 0, 0], 0)
        I0 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        begin = tf.stack([0, tf.squeeze(self.tend), 0, 0, 0], 0)
        I1 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        begin = tf.stack([0, tf.squeeze(self.tend + 1), 0, 0, 0], 0)
        I2 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        return I0, I1, I2




