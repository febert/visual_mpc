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

def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)

        return tf.reduce_sum(error) / normalization


def flow_smooth_cost(flow, norm, mode, mask):
    """
    computes the norms of the derivatives and averages over the image
    :param flow_field:

    :return:
    """
    if mode == '2nd':  # compute 2nd derivative
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

    elif mode == 'sobel':
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # sobel filter
        filter_y = np.transpose(filter_x)
        weight_array = np.ones([3, 3, 1, 2])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y

    weights = tf.constant(weight_array, dtype=tf.float32)

    flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
    delta_u =  tf.nn.conv2d(flow_u, weights, strides=[1, 1, 1, 1], padding='SAME')
    delta_v =  tf.nn.conv2d(flow_v, weights, strides=[1, 1, 1, 1], padding='SAME')

    deltas = tf.concat([delta_u, delta_v], axis=3)

    return norm(deltas, mask)


def get_coords(img_shape):
    """
    returns coordinate grid corresponding to identity appearance flow
    :param img_shape:
    :return:
    """
    y = tf.cast(tf.range(img_shape[1]), tf.float32)
    x = tf.cast(tf.range(img_shape[2]), tf.float32)
    batch_size = img_shape[0]

    X, Y = tf.meshgrid(x, y)
    coords = tf.expand_dims(tf.stack((X, Y), axis=2), axis=0)
    coords = tf.tile(coords, [batch_size, 1, 1, 1])
    return coords

def resample_layer(src_img, warp_pts, name="tgt_img"):
    with tf.variable_scope(name):
        return tf.contrib.resampler.resampler(src_img, warp_pts)

def warp_pts_layer(flow_field, name="warp_pts"):
    with tf.variable_scope(name):
        img_shape = flow_field.get_shape().as_list()
        return flow_field + get_coords(img_shape)

def apply_warp(I0, flow_field):
    warp_pts = warp_pts_layer(flow_field)
    return resample_layer(I0, warp_pts)

class GoalDistanceNet(object):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True,
                 images = None,
                 iter_num = None,
                 pred_images = None
                 ):

        if conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        elif conf['normalization'] == 'None':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % conf['normalization'])

        self.conf = conf

        self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        self.seq_len = self.conf['sequence_length']
        self.bsize = self.conf['batch_size']

        self.img_height = self.conf['orig_size'][0]
        self.img_width = self.conf['orig_size'][1]

        if load_data:
            self.iter_num = tf.placeholder(tf.float32, [], name='iternum')

            train_dict = build_tfrecord_fn(conf, training=True)
            val_dict = build_tfrecord_fn(conf, training=False)
            dict = tf.cond(self.train_cond > 0,
                             # if 1 use trainigbatch else validation batch
                             lambda: train_dict,
                             lambda: val_dict)
            self.images = dict['images']

            if 'vidpred_data' in conf:  # register predicted video to real
                self.pred_images = tf.squeeze(dict['gen_images'])
                self.pred_states = tf.squeeze(dict['gen_states'])

            if 'temp_divide_and_conquer' not in self.conf:
                self.I0, self.I1 = self.sel_images()

        elif images == None:  #feed values at test time
            self.I0 = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.I1 = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))

        else:  # get tensors from videoprediction model
            self.iter_num = iter_num
            self.pred_images = tf.stack(pred_images, axis=1)
            self.images = tf.stack(images[1:], axis=1) # cutting off first image since there is no pred image for it
            self.conf['sequence_length'] = self.conf['sequence_length']-1
            self.I0, self.I1 = self.sel_images()

        self.occ_fwd = tf.zeros([self.bsize,  self.img_height,  self.img_width])
        self.occ_bwd = tf.zeros([self.bsize,  self.img_height,  self.img_width])

        self.losses = []
        self.build_net()

        if build_loss:
            self.add_pair_loss(self.I1, self.gen_I1, self.occ_bwd, self.flow_bwd,
                               self.I0, self.gen_I0, self.occ_fwd, self.flow_fwd)
            self.combine_losses()
            # image_summary:
            if 'fwd_bwd' in self.conf:
                self.image_summaries = self.build_image_summary(
                    [self.I0, self.I1, self.gen_I0, self.gen_I1, length(self.flow_bwd), length(self.flow_fwd), self.occ_mask_bwd, self.occ_mask_fwd])
            else:
                self.image_summaries = self.build_image_summary([self.I0, self.I1, self.gen_I1, length(self.flow_bwd)])

    def build_net(self):
        if 'fwd_bwd' in self.conf:
            with tf.variable_scope('warpnet'):
                self.warped_I0_to_I1, self.warp_pts_bwd, self.flow_bwd, h6_bwd = self.warp(self.I0, self.I1)
            with tf.variable_scope('warpnet', reuse=True):
                self.warped_I1_to_I0, self.warp_pts_fwd, self.flow_fwd, h6_fwd = self.warp(self.I1, self.I0)

            bwd_flow_warped_fwd = apply_warp(self.flow_bwd, self.flow_fwd)
            self.diff_flow_fwd = self.flow_fwd + bwd_flow_warped_fwd

            fwd_flow_warped_bwd = apply_warp(self.flow_fwd, self.flow_bwd)
            self.diff_flow_bwd = self.flow_bwd + fwd_flow_warped_bwd

            if 'hard_occ_thresh' in self.conf:
                print 'doing hard occ thresholding'
                mag_sq = length_sq(self.flow_fwd) + length_sq(self.flow_bwd)

                if 'occ_thres_mult' in self.conf:
                    occ_thres_mult = self.conf['occ_thres_mult']
                    occ_thres_offset = self.conf['occ_thres_offset']
                else:
                    occ_thres_mult = 0.01
                    occ_thres_offset = 0.5

                occ_thresh = occ_thres_mult * mag_sq + occ_thres_offset
                self.occ_fwd = tf.squeeze(tf.cast(length_sq(self.diff_flow_fwd) > occ_thresh, tf.float32))
                self.occ_bwd = tf.squeeze(tf.cast(length_sq(self.diff_flow_bwd) > occ_thresh, tf.float32))
            else:
                bias = self.conf['occlusion_handling_bias']
                scale = self.conf['occlusion_handling_scale']
                diff_flow_fwd_sqlen = tf.reduce_sum(tf.square(self.diff_flow_fwd), axis=3)
                diff_flow_bwd_sqlen = tf.reduce_sum(tf.square(self.diff_flow_bwd), axis=3)
                self.occ_fwd = tf.nn.sigmoid(diff_flow_fwd_sqlen * scale + bias)  # gets 1 if occluded 0 otherwise
                self.occ_bwd = tf.nn.sigmoid(diff_flow_bwd_sqlen * scale + bias)

            self.gen_I1 = self.warped_I0_to_I1
            self.gen_I0 = self.warped_I1_to_I0
        else:
            self.warped_I0_to_I1, self.warp_pts_bwd, self.flow_bwd, _ = self.warp(self.I0, self.I1)
            self.gen_I1 = self.warped_I0_to_I1
            self.gen_I0, self.flow_fwd = None, None

        self.occ_mask_bwd = 1 - self.occ_bwd  # 0 at occlusion
        self.occ_mask_fwd = 1 - self.occ_fwd
        self.occ_mask_bwd = self.occ_mask_bwd[:, :, :, None]
        self.occ_mask_fwd = self.occ_mask_fwd[:, :, :, None]
        if 'stop_occ_grad' in self.conf:
            print 'stopping occ mask grads'
            self.occ_mask_bwd = tf.stop_gradient(self.occ_mask_bwd)
            self.occ_mask_fwd = tf.stop_gradient(self.occ_mask_fwd)


    def sel_images(self):
        sequence_length = self.conf['sequence_length']
        t_fullrange = 2e4
        delta_t = tf.cast(tf.ceil(sequence_length * (tf.cast(self.iter_num + 1, tf.float32)) / t_fullrange), dtype=tf.int32)
        delta_t = tf.clip_by_value(delta_t, 1, sequence_length-1)

        self.tstart = tf.random_uniform([1], 0, sequence_length - delta_t, dtype=tf.int32)
        self.tend = self.tstart + tf.random_uniform([1], tf.ones([], dtype=tf.int32), delta_t + 1, dtype=tf.int32)

        begin = tf.stack([0, tf.squeeze(self.tstart), 0, 0, 0],0)

        if 'vidpred_data' in self.conf:
            I0 = tf.squeeze(tf.slice(self.pred_images, begin, [-1, 1, -1, -1, -1]))
            print 'using pred images'
        else:
            I0 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        begin = tf.stack([0, tf.squeeze(self.tend), 0, 0, 0], 0)
        I1 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        return I0, I1

    def conv_relu_block(self, input, out_ch, k=3, upsmp=False):
        h = slim.layers.conv2d(  # 32x32x64
            input,
            out_ch, [k, k],
            stride=1)

        if upsmp:
            mult = 2
        else: mult = 0.5
        imsize = np.array(h.get_shape().as_list()[1:3])*mult

        h = tf.image.resize_images(h, imsize, method=tf.image.ResizeMethod.BILINEAR)
        return h

    def warp(self, source_image, dest_image):
        """
        warps I0 onto I1
        :param source_image:
        :param dest_image:
        :return:
        """

        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
        else: ch_mult = 1

        if 'late_fusion' in self.conf:
            print 'building late fusion net'
            with tf.variable_scope('pre_proc_source'):
                h3_1 = self.pre_proc_net(source_image, ch_mult)
            with tf.variable_scope('pre_proc_dest'):
                h3_2 = self.pre_proc_net(dest_image, ch_mult)
            h3 = tf.concat([h3_1, h3_2], axis=3)
        else:
            I0_I1 = tf.concat([source_image, dest_image], axis=3)
            with tf.variable_scope('h1'):
                h1 = self.conv_relu_block(I0_I1, out_ch=32*ch_mult)  #24x32x3

            with tf.variable_scope('h2'):
                h2 = self.conv_relu_block(h1, out_ch=64*ch_mult)  #12x16x3

            with tf.variable_scope('h3'):
                h3 = self.conv_relu_block(h2, out_ch=128*ch_mult)  #6x8x3

        with tf.variable_scope('h4'):
            h4 = self.conv_relu_block(h3, out_ch=64*ch_mult, upsmp=True)  #12x16x3

        with tf.variable_scope('h5'):
            h5 = self.conv_relu_block(h4, out_ch=32*ch_mult, upsmp=True)  #24x32x3

        with tf.variable_scope('h6'):
            h6 = self.conv_relu_block(h5, out_ch=16*ch_mult, upsmp=True)  #48x64x3

        with tf.variable_scope('h7'):
            flow_field = slim.layers.conv2d(  # 128x128xdesc_length
                h6,  2, [5, 5], stride=1, activation_fn=None)

        warp_pts = warp_pts_layer(flow_field)
        gen_image = resample_layer(source_image, warp_pts)

        return gen_image, warp_pts, flow_field, h6

    def pre_proc_net(self, input, ch_mult):
        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(input, out_ch=32 * ch_mult)  # 24x32x3
        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult)  # 12x16x3
        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, out_ch=128 * ch_mult)  # 6x8x3
        return h3

    def build_image_summary(self, tensors, numex=16):
        """
        takes numex examples from every tensor and concatentes examples side by side
        and the different tensors from top to bottom
        :param tensors:
        :param numex:
        :return:
        """
        ten_list = []
        for ten in tensors:
            if len(ten.get_shape().as_list()) == 3 or ten.get_shape().as_list()[-1] == 1:
                ten = colorize(ten, tf.reduce_min(ten), tf.reduce_max(ten), 'viridis')
            unstacked = tf.unstack(ten, axis=0)[:numex]
            concated = tf.concat(unstacked, axis=1)
            ten_list.append(concated)
        combined = tf.concat(ten_list, axis=0)
        combined = tf.reshape(combined, [1]+combined.get_shape().as_list())
        return tf.summary.image('Images', combined)

    def add_pair_loss(self, I1, gen_I1, occ_bwd, flow_bwd, diff_flow_fwd=None,
                      I0=None, gen_I0=None, occ_fwd=None, flow_fwd=None, diff_flow_bwd=None):
        occ_mask_bwd = 1 - occ_bwd  # 0 at occlusion
        occ_mask_bwd = occ_mask_bwd[:, :, :, None]
        if occ_fwd is not None:
            occ_mask_fwd = 1 - occ_fwd
            occ_mask_fwd = occ_mask_fwd[:, :, :, None]

        if self.conf['norm'] == 'l2':
            norm = mean_square
        elif self.conf['norm'] == 'charbonnier':
            norm = charbonnier_loss
        else: raise ValueError("norm not defined!")

        self.losses.append((norm((gen_I1 - I1), occ_mask_bwd), 'train_I1_recon_cost'))

        if 'fwd_bwd' in self.conf:
            self.losses.append(norm((gen_I0 - I0), occ_mask_fwd), 'train_I0_recon_cost')

            fd = self.conf['flow_diff_cost']
            self.losses.append(((norm(diff_flow_fwd, occ_mask_fwd)
                                +norm(diff_flow_bwd, occ_mask_bwd)) * fd), 'train_flow_diff_cost')

            if 'occlusion_handling' in self.conf:
                occ = self.conf['occlusion_handling']
                self.losses.append(((tf.reduce_mean(occ_fwd) + tf.reduce_mean(occ_bwd)) * occ,
                                    'train_occlusion_handling'))

        if 'smoothcost' in self.conf:
            sc = self.conf['smoothcost']
            self.losses.append((flow_smooth_cost(flow_bwd, norm, self.conf['smoothmode'],
                                               occ_mask_bwd) * sc, 'train_smooth_bwd'))
            if 'fwd_bwd' in self.conf:
                self.losses.append((flow_smooth_cost(flow_fwd, norm, self.conf['smoothmode'],
                                                   occ_mask_fwd) * sc, 'train_smooth_fwd'))
        if 'flow_penal' in self.conf:
            self.losses.append(((tf.reduce_mean(tf.square(flow_bwd)) +
                                 tf.reduce_mean(tf.square(flow_fwd))) * self.conf['flow_penal'],
                                 'flow_penal'))

    def combine_losses(self):
        train_summaries = []
        val_summaries = []
        self.loss = 0
        for l in self.losses:
            name = l[1]
            single_loss = l[0]
            self.loss += single_loss
            train_summaries.append(tf.summary.scalar(name, single_loss))
        train_summaries.append(tf.summary.scalar('train_total', self.loss))
        val_summaries.append(tf.summary.scalar('val_total', self.loss))
        self.train_summ_op = tf.summary.merge(train_summaries)
        self.val_summ_op = tf.summary.merge(val_summaries)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def visualize(self, sess):
        if 'source_basedirs' in self.conf:  # visualizing single warps from pairs of images
            self.conf['sequence_length'] = 2
            self.conf.pop('vidpred_data', None)
            r = OnlineReader(self.conf, 'test', sess=sess)
            images = r.get_batch_tensors()
            [images] = sess.run([images])

        else:  # when visualizing sequence of warps from video
            videos = build_tfrecord_fn(self.conf)


            if 'vidpred_data' in self.conf:
                images, pred_images = sess.run([videos['images'], videos['gen_images']])
                pred_images = np.squeeze(pred_images)
            else:
                [images] = sess.run([videos['images']])

        num_examples = self.conf['batch_size']
        I1 = images[:, -1]

        gen_images_I0 = []
        gen_images_I1 = []
        I0_t_reals = []
        I0_ts = []
        flow_mags_bwd = []
        flow_mags_fwd = []
        occ_bwd_l = []
        occ_fwd_l = []
        warpscores_bwd = []
        warpscores_fwd = []

        for t in range(self.conf['sequence_length']-1):
            if 'vidpred_data' in self.conf:
                I0_t = pred_images[:, t]
                I0_t_real = images[:, t]

                I0_t_reals.append(I0_t_real)
            else:
                I0_t = images[:, t]

            I0_ts.append(I0_t)

            if 'fwd_bwd' in self.conf:
                [gen_image_I1, bwd_flow, occ_bwd, norm_occ_mask_bwd,
                 gen_image_I0, fwd_flow, occ_fwd, norm_occ_mask_fwd] = sess.run([self.gen_I1,
                                                                                 self.flow_bwd,
                                                                                 self.occ_bwd,
                                                                                 self.occ_mask_bwd,
                                                                                 self.gen_I0,
                                                                                 self.flow_fwd,
                                                                                 self.occ_fwd,
                                                                                 self.occ_mask_fwd,
                                                                                 ], {self.I0_pl: I0_t, self.I1_pl: I1})
                occ_bwd_l.append(occ_bwd)
                occ_fwd_l.append(occ_fwd)

                gen_images_I0.append(gen_image_I0)
            else:
                [gen_image_I1, bwd_flow] = sess.run([self.gen_I1, self.flow_bwd], {self.I0_pl:I0_t, self.I1_pl: I1})

            gen_images_I1.append(gen_image_I1)

            flow_mag_bwd = np.linalg.norm(bwd_flow, axis=3)
            flow_mags_bwd.append(flow_mag_bwd)
            if 'fwd_bwd' in self.conf:
                flow_mag_fwd = np.linalg.norm(fwd_flow, axis=3)
                flow_mags_fwd.append(flow_mag_fwd)
                warpscores_bwd.append(np.mean(np.mean(flow_mag_bwd * np.squeeze(norm_occ_mask_bwd), axis=1), axis=1))
                warpscores_fwd.append(np.mean(np.mean(flow_mag_fwd * np.squeeze(norm_occ_mask_fwd), axis=1), axis=1))
            else:
                warpscores_bwd.append(np.mean(np.mean(flow_mag_bwd, axis=1), axis=1))

            # flow_mags.append(self.color_code(flow_mag, num_examples))

        videos = collections.OrderedDict()
        videos['I0_ts'] = I0_ts
        videos['gen_images_I1'] = gen_images_I1
        videos['flow_mags_bwd'] = (flow_mags_bwd, warpscores_bwd)

        if 'vidpred_data' in self.conf:
            videos['I0_t_real'] = I0_t_reals

        if 'fwd_bwd' in self.conf:
            videos['occ_bwd'] = occ_bwd_l

            videos['gen_images_I0'] = gen_images_I0
            videos['flow_mags_fwd'] = (flow_mags_fwd, warpscores_fwd)
            videos['occ_fwd'] = occ_fwd_l

        name = str.split(self.conf['output_dir'], '/')[-2]
        dict = {'videos':videos, 'name':name, 'I1':I1}

        cPickle.dump(dict, open(self.conf['output_dir'] + '/data.pkl', 'wb'))
        make_plots(self.conf, dict=dict)

    def color_code(self, input, num_examples):
        cmap = plt.cm.get_cmap()

        l = []
        for b in range(num_examples):
            f = input[b] / (np.max(input[b]) + 1e-6)
            f = cmap(f)[:, :, :3]
            l.append(f)

        return np.stack(l, axis=0)


def make_plots(conf, dict=None, filename = None):
    if dict == None:
        dict = cPickle.load(open(filename))

    print 'loaded'
    videos = dict['videos']

    I0_ts = videos['I0_ts']

    # num_exp = I0_t_reals[0].shape[0]
    num_ex = 4
    start_ex = 0

    num_rows = num_ex*len(videos.keys())

    num_cols = len(I0_ts) + 1

    # plt.figure(figsize=(num_rows, num_cols))
    # gs1 = gridspec.GridSpec(num_rows, num_cols)
    # gs1.update(wspace=0.025, hspace=0.05)

    width_per_ex = 2.5

    standard_size = np.array([width_per_ex * num_cols, num_rows * 1.5])  ### 1.5
    figsize = (standard_size).astype(np.int)

    f, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    for col in range(num_cols -1):
        row = 0
        for ex in range(start_ex, start_ex + num_ex, 1):

            for tag in videos.keys():
                print 'doing tag {}'.format(tag)
                if isinstance(videos[tag], tuple):
                    im = videos[tag][0][col]
                    score = videos[tag][1]
                    axarr[row, col].set_title('{:10.3f}'.format(score[col][ex]), fontsize=5)
                else:
                    im = videos[tag][col]

                h = axarr[row, col].imshow(np.squeeze(im[ex]), interpolation='none')

                if len(im.shape) == 3:
                    plt.colorbar(h, ax=axarr[row, col])
                axarr[row, col].axis('off')
                row += 1

    row = 0
    col = num_cols-1
    for ex in range(start_ex, start_ex + num_ex, 1):
        im = dict['I1'][ex]
        h = axarr[row, col].imshow(np.squeeze(im), interpolation='none')
        plt.colorbar(h, ax=axarr[row, col])
        axarr[row, col].axis('off')
        row += len(videos.keys())

    # plt.axis('off')
    f.subplots_adjust(wspace=0, hspace=0.3)

    # f.subplots_adjust(vspace=0.1)
    # plt.show()
    plt.savefig(conf['output_dir']+'/warp_costs_{}.png'.format(dict['name']))


def calc_warpscores(flow_field):
    return np.sum(np.linalg.norm(flow_field, axis=3), axis=[2, 3])

def draw_text(img, float):
    img = (img*255.).astype(np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 5)
    draw.text((0, 0), "{}".format(float), (255, 255, 0), font=font)
    img = np.asarray(img).astype(np.float32)/255.

    return img


if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/hardthres/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')