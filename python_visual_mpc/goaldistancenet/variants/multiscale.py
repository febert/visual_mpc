import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet
from python_visual_mpc.goaldistancenet.utils.ops import conv2d, length
from python_visual_mpc.goaldistancenet.gdnet import apply_warp


class MultiscaleGoalDistanceNet(GoalDistanceNet):

    def build_net(self):
        self.warped_I0_to_I1, self.warp_pts_bwd, self.flow_bwd, _ = self.warp(self.I0, self.I1)

        self.gen_I1 = self.warped_I0_to_I1
        self.gen_I0, self.flow_fwd = None, None

        self.occ_mask_bwd = 1 - self.occ_bwd  # 0 at occlusion
        self.occ_mask_fwd = 1 - self.occ_fwd

        if self.build_loss:
            self.combine_losses()


    def upconv_intm_flow_block(self, source_image, dest_image, h_m1, h_skip=None, flow_lm1=None,
                               gen_im_m1=None, k=3, tag=None, dest_mult=2, num_feat=None):
        imsize = np.array(h_m1.get_shape().as_list()[1:3])*dest_mult
        source_im = tf.image.resize_images(source_image, imsize, method=tf.image.ResizeMethod.BILINEAR)
        dest_im = tf.image.resize_images(dest_image, imsize, method=tf.image.ResizeMethod.BILINEAR)
        h_m1 = tf.image.resize_images(h_m1, imsize, method=tf.image.ResizeMethod.BILINEAR)

        if flow_lm1 is not None:
            flow_lm1 = tf.image.resize_images(flow_lm1, imsize, method=tf.image.ResizeMethod.BILINEAR)
            gen_im_m1 = tf.image.resize_images(gen_im_m1, imsize, method=tf.image.ResizeMethod.BILINEAR)
            h = tf.concat([h_m1, source_im, dest_im, flow_lm1, gen_im_m1], axis=-1)
        else:
            h = tf.concat([h_m1, source_im, dest_im], axis=-1)

        if h_skip is not None:
            h = tf.concat([h, h_skip], axis=-1)

        for i in range(3):
            h = slim.layers.conv2d(h, num_feat, [k, k], stride=1)

        flow, h_out = h[:,:,:,:num_feat//2], h[:,:,:, num_feat//2:]
        h_out = slim.layers.conv2d(h_out, num_feat//2, [k, k], stride=1)

        flow = slim.layers.conv2d(flow, 2, [k, k], stride=1, activation_fn=None)
        if flow_lm1 is not None:
            flow += tf.image.resize_images(flow_lm1, imsize, method=tf.image.ResizeMethod.BILINEAR)

        gen_im, warp_pts = apply_warp(source_im, flow)
        self.add_pair_loss(dest_im, gen_im, flow_bwd=flow, suf=tag)
        self.gen_I1_multiscale.append(gen_im)
        self.I0_multiscale.append(source_im)
        self.flow_multiscale.append(flow)
        sum = self.build_image_summary([source_im, dest_im, gen_im, length(flow)], suf=tag)
        self.imsum.append(sum)

        return gen_im, flow, h_out, warp_pts


    def warp(self, source_image, dest_image):
        """
        warps I0 onto I1
        :param source_image:
        :param dest_image:
        :return:
        """
        self.gen_I1_multiscale = []
        self.I0_multiscale = []
        self.I1_multiscale = []
        self.flow_multiscale = []
        self.imsum = []

        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
        else: ch_mult = 1

        I0_I1 = tf.concat([source_image, dest_image], axis=3)

        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(I0_I1, out_ch=32*ch_mult, n_layer=3)  #48x64

        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64*ch_mult, n_layer=3)  #24x32

        with tf.variable_scope('h3'):
            gen_im3, flow_h3, h3, _ = self.upconv_intm_flow_block(source_image, dest_image, h2, tag='h3', dest_mult=1, num_feat=64*ch_mult) # 24x32


        with tf.variable_scope('h4'):
            gen_im4, flow_h4, h4, _ = self.upconv_intm_flow_block(source_image, dest_image, h3, h_skip=h1,
                                               gen_im_m1=gen_im3, flow_lm1=flow_h3, tag='h4', dest_mult=2, num_feat=32*ch_mult) # 48x64

        with tf.variable_scope('h5'):
            gen_im5, flow_h5, h5, warp_pts = self.upconv_intm_flow_block(source_image, dest_image, h4,
                                               gen_im_m1=gen_im4, flow_lm1=flow_h4, tag='h5', dest_mult=2, num_feat=32*ch_mult) # 96x128

        self.image_summaries = tf.summary.merge(self.imsum)

        return gen_im5, warp_pts, flow_h5, None


