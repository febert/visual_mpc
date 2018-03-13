import tensorflow as tf
import numpy as np
import os
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
from python_visual_mpc.layers.batchnorm_layer import batchnorm_train, batchnorm_test

def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def charbonnier_loss(x, weights=None, alpha=0.45, beta=1.0, epsilon=0.001, per_int_ex = False):
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
        batch, int, height, width, channels = tf.unstack(x.get_shape().as_list())

        if per_int_ex:
            normalization = tf.cast(height * width * channels, tf.float32)
        else:
            normalization = tf.cast(int * batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)
        if weights is not None:
            error *= tf.reshape(weights, [batch, int, 1,1, 1])

        if per_int_ex:
            return tf.reduce_sum(error, axis=[2,3,4]) / normalization
        else:
            return tf.reduce_sum(error) / normalization


class Dist_Net(object):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True,
                 images = None,
                 inference = False,
                 sess = None,
                 ):

        if conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        if conf['normalization'] == 'bnorm':
            self.normalizer_fn = batchnorm_train
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

        self.train_summaries = []
        self.val_summaries = []

        if 'vgg19' in self.conf:
            self.vgg_dict = np.load(conf['vgg19'], encoding='latin1').item()

        if load_data:
            self.iter_num = tf.placeholder(tf.float32, [], name='iternum')

            if 'source_basedirs' in self.conf:
                tag_images = {'name': 'images',
                              'file': '/images/im{}.png',  # only tindex
                              'shape': [48, 64, 3]}

                tag_actions = {'name': 'actions',
                               'file': '/state_action.pkl',  # only tindex
                               'shape': [3],
                               }
                conf['ngroup'] = 1000
                conf['sourcetags'] = [tag_images, tag_actions]
                r = OnlineReader(conf, 'train', sess=sess)
                train_image_batch, train_action_batch = r.get_batch_tensors()
                r = OnlineReader(conf, 'val', sess=sess)
                val_image_batch, val_action_batch = r.get_batch_tensors()
                self.images, self.actions = tf.cond(self.train_cond > 0,
                                                    # if 1 use trainigbatch else validation batch
                                                    lambda: [train_image_batch, train_action_batch],
                                                    lambda: [val_image_batch, val_action_batch])
            else:
                train_dict = build_tfrecord_fn(conf, training=True)
                val_dict = build_tfrecord_fn(conf, training=False)
                dict = tf.cond(self.train_cond > 0,
                               # if 1 use trainigbatch else validation batch
                               lambda: train_dict,
                               lambda: val_dict)
                self.images = dict['images']
                self.actions = dict['actions']

        elif images == None:  #feed values at test time
            self.Istart = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.Igoal = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
        self.build_loss = build_loss
        self.loss_dict = {}

    def build_net(self):
        self.gen_actions = self.reg_actions(self.images)
        self.create_loss_and_optimizer(self.gen_actions, self.actions)

    def conv_relu_block(self, input, out_ch, k=3, upsmp=False):
        h = slim.layers.conv2d(input, out_ch, [k, k], stride=1)
        h = self.normalizer_fn(h)

        if upsmp:
            mult = 2
        else: mult = 0.5
        imsize = np.array(h.get_shape().as_list()[1:3])*mult

        h = tf.image.resize_images(h, imsize, method=tf.image.ResizeMethod.BILINEAR)
        return h

    def reg_actions(self, inp_images):
        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
            print 'using chmult', ch_mult
        else: ch_mult = 1

        actions = []
        for t in range(self.seq_len):
            if  t== 0:
                reuse = False
            else: reuse = True

            with tf.variable_scope('intm_enc', reuse=reuse):
                with tf.variable_scope('h1'):
                    if 'vgg19' in self.conf:
                        h1 = self.conv_relu_block(self.vgg_layer(inp_images[:, t]), out_ch=32 * ch_mult)  # 24x32x3
                    else:
                        h1 = self.conv_relu_block(inp_images[:, t], out_ch=32 * ch_mult)  # 24x32x3
                with tf.variable_scope('h2'):
                    h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult)  # 12x16x3
                with tf.variable_scope('h3'):
                    h3 = self.conv_relu_block(h2, out_ch=8 * ch_mult)  # 6x8x3

            if 'use_fp' in self.conf:
                _, num_rows, num_cols, num_fp = h3.get_shape().as_list()
                x_map = np.empty([num_rows, num_cols], np.float32)
                y_map = np.empty([num_rows, num_cols], np.float32)

                for i in range(num_rows):
                    for j in range(num_cols):
                        x_map[i, j] = (i - num_rows / 2.0) / num_rows
                        y_map[i, j] = (j - num_cols / 2.0) / num_cols
                x_map = tf.convert_to_tensor(x_map)
                y_map = tf.convert_to_tensor(y_map)

                x_map = tf.reshape(x_map, [num_rows * num_cols])
                y_map = tf.reshape(y_map, [num_rows * num_cols])

                features = tf.reshape(tf.transpose(h3, [0, 3, 1, 2]), [-1, num_rows * num_cols])
                softmax = tf.nn.softmax(features)

                fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
                fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

                f1 = tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])
            else:
                f1 = tf.reshape(h3, [self.bsize, -1])
            f2 = slim.layers.fully_connected(f1, num_outputs=100)
            actions.append(slim.layers.fully_connected(f2, num_outputs=self.conf['adim'], activation_fn=None))

        return tf.stack(actions, axis=1)

    def vgg_layer(self, images):
        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        # print 'images', images
        rgb_scaled = images*255.
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
            ])

        name = "conv1_1"
        with tf.variable_scope(name):
            filt = tf.constant(self.vgg_dict[name][0], name="filter")
            conv = tf.nn.conv2d(bgr, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = tf.constant(self.vgg_dict[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases)
            out = tf.nn.relu(bias)
            out = out/255.
        return out

    def create_loss_and_optimizer(self, gen_actions, actions):
        diff = gen_actions - actions
        mse_loss = tf.reduce_mean(tf.square(diff))
        self.loss_dict['mse_loss'] = (mse_loss, 1.)
        self.combine_losses()

    def combine_losses(self):
        self.loss = 0.
        for k in self.loss_dict.keys():
            single_loss, weight = self.loss_dict[k]
            self.loss += single_loss*weight
            self.train_summaries.append(tf.summary.scalar('train_' + k, single_loss))
            self.val_summaries.append(tf.summary.scalar('val_' + k, single_loss))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.train_summaries.append(tf.summary.scalar('train_total', self.loss))
        self.train_summ_op = tf.summary.merge(self.train_summaries)

        self.val_summaries.append(tf.summary.scalar('val_total', self.loss))
        self.val_summ_op = tf.summary.merge(self.val_summaries)

    def color_code(self, input, num_examples):
        cmap = plt.cm.get_cmap()

        l = []
        for b in range(num_examples):
            f = input[b] / (np.max(input[b]) + 1e-6)
            f = cmap(f)[:, :, :3]
            l.append(f)
        return np.stack(l, axis=0)

if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/tdac_cons0_cartgripper/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')
