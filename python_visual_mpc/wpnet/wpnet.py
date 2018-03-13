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


class WaypointNet(object):
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
        self.nz = self.conf['lt_dim']

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
                conf['ngroup'] = 1000
                conf['sourcetags'] = [tag_images]

                if not inference:
                    r = OnlineReader(conf, 'train', sess=sess)
                    train_image_batch = r.get_batch_tensors()

                    r = OnlineReader(conf, 'val', sess=sess)
                    val_image_batch = r.get_batch_tensors()
                    self.images = tf.cond(self.train_cond > 0,
                                          # if 1 use trainigbatch else validation batch
                                          lambda: train_image_batch,
                                          lambda: val_image_batch)
                else:
                    r = OnlineReader(conf, 'test', sess=sess)
                    self.images = r.get_batch_tensors()
            else:
                train_dict = build_tfrecord_fn(conf, training=True)
                val_dict = build_tfrecord_fn(conf, training=False)
                dict = tf.cond(self.train_cond > 0,
                               # if 1 use trainigbatch else validation batch
                               lambda: train_dict,
                               lambda: val_dict)
                self.images = dict['images']

            self.Istart = self.images[:,0]
            self.I_intm = self.images[:, 1:self.conf['sequence_length'] - 1]
            self.Igoal = self.images[:, -1]

        elif images == None:  #feed values at test time
            self.Istart = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.Igoal = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
        self.build_loss = build_loss
        self.loss_dict = {}

    def build_net(self, traintime = True):
        self.build_vae(traintime)

    def build_vae(self, traintime):
        self.ctxt_encoding = self.ctxt_enc(self.Istart, self.Igoal)

        self.z = []
        if traintime:
            self.z_mean, self.z_log_sigma_sq = self.intm_enc(self.I_intm)
            for i in range(self.seq_len -2):
                eps = tf.random_normal([self.bsize, self.nz], 0.0, 1.0, dtype=tf.float32)
                self.z.append(tf.add(self.z_mean[:,i], tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq[:, i])), eps)))
            self.z = tf.stack(self.z, axis=1)
        else:
            self.z_mean, self.z_log_sigma_sq = None, None
            for i in range(self.seq_len -2):
                eps = tf.random_normal([self.bsize, self.nz], 0.0, 1.0, dtype=tf.float32)
                self.z.append(eps)
            self.z = tf.stack(self.z, axis=1)

        # Get the reconstructed mean from the decoder
        self.x_reconstr_mean = self.decode(self.ctxt_encoding, self.z)
        self.z_summary = tf.summary.histogram("z", self.z)

        intm = tf.unstack(self.I_intm, axis=1)
        x_reconstr_mean = tf.unstack(self.x_reconstr_mean, axis=1)
        self.image_summaries = self.build_image_summary(side_by_side=[[self.Istart, self.Igoal] + intm,
                                                                      [self.Istart, self.Igoal] + x_reconstr_mean])

        self.create_loss_and_optimizer(self.x_reconstr_mean, self.I_intm, self.z_log_sigma_sq, self.z_mean, traintime)

    def sel_images(self):
        sequence_length = self.conf['sequence_length']
        t_fullrange = 2e4
        delta_t = tf.cast(tf.ceil(sequence_length * (tf.cast(self.iter_num + 1, tf.float32)) / t_fullrange), dtype=tf.int32)
        delta_t = tf.clip_by_value(delta_t, 1, sequence_length-1)

        self.tstart = tf.random_uniform([1], 0, sequence_length - delta_t, dtype=tf.int32)
        self.tend = self.tstart + tf.random_uniform([1], tf.ones([], dtype=tf.int32), delta_t + 1, dtype=tf.int32)

        begin = tf.stack([0, tf.squeeze(self.tstart), 0, 0, 0],0)

        I0 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        begin = tf.stack([0, tf.squeeze(self.tend), 0, 0, 0], 0)
        I1 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        return I0, I1

    def conv_relu_block(self, input, out_ch, k=3, upsmp=False):
        h = slim.layers.conv2d(input, out_ch, [k, k], stride=1)
        h = self.normalizer_fn(h)

        if upsmp:
            mult = 2
        else: mult = 0.5
        imsize = np.array(h.get_shape().as_list()[1:3])*mult

        h = tf.image.resize_images(h, imsize, method=tf.image.ResizeMethod.BILINEAR)
        return h

    def intm_enc(self, intm_images):
        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
            print 'using chmult', ch_mult
        else: ch_mult = 1

        log_sigma_diag_l = []
        mu_l = []

        for t in range(self.seq_len -2):
            if  t== 0:
                reuse = False
            else: reuse = True

            with tf.variable_scope('intm_enc', reuse=reuse):
                with tf.variable_scope('h1'):
                    if 'vgg19' in self.conf:
                        h1 = self.conv_relu_block(self.vgg_layer(intm_images[:, t]), out_ch=32 * ch_mult)  # 24x32x3
                    else:
                        h1 = self.conv_relu_block(intm_images[:, t], out_ch=32 * ch_mult)  # 24x32x3
                with tf.variable_scope('h2'):
                    h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult)  # 12x16x3
                with tf.variable_scope('h3'):
                    h3 = self.conv_relu_block(h2, out_ch=8 * ch_mult)  # 6x8x3

            h3 = tf.reshape(h3, [self.bsize, -1])

            mu = slim.layers.fully_connected(h3, num_outputs=self.nz, activation_fn=None)
            mu_l.append(mu)

            log_sigma_diag = slim.layers.fully_connected(h3, num_outputs=self.nz, activation_fn=None)
            log_sigma_diag_l.append(log_sigma_diag)

        return tf.stack(mu_l,axis=1), tf.stack(log_sigma_diag_l,axis=1)

    def ctxt_enc(self, start_im, goal_im):
        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
        else: ch_mult = 1

        if 'vgg19' in self.conf:
            vgg_enc0 = self.vgg_layer(start_im)
            vgg_enc1 = self.vgg_layer(goal_im)
            I0_I1 = tf.concat([vgg_enc0, vgg_enc1], axis=3)
        else:
            I0_I1 = tf.concat([start_im, goal_im], axis=3)

        self.enc1, self.enc2, self.enc3 = [], [], []
        with tf.variable_scope('ctxt_enc'):
            with tf.variable_scope('h1'):
                h1 = self.conv_relu_block(I0_I1, out_ch=32*ch_mult)  #24x32x3
                self.enc1.append(h1)
            with tf.variable_scope('h2'):
                h2 = self.conv_relu_block(h1, out_ch=64*ch_mult)  #12x16x3
                self.enc2.append(h2)
            with tf.variable_scope('h3'):
                h3 = self.conv_relu_block(h2, out_ch=8*ch_mult)  #6x8x3
                self.enc3.append(h3)

        ctxt_enc = slim.layers.fully_connected(tf.reshape(h3, [self.bsize, -1]), num_outputs=self.nz, activation_fn=None)
        return ctxt_enc

    def decode(self, ctxt_enc, z):
        """
        warps I0 onto I1
        :param start_im:
        :param goal_im:
        :return:
        """
        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
        else:
            ch_mult = 1

        gen_images = []

        for t in range(self.seq_len -2):
            if  t== 0:
                reuse = False
            else: reuse = True

            enc = tf.concat([ctxt_enc, z[:, t]], axis=1)
            enc = tf.tile(tf.reshape(enc,[self.bsize, 1,1, enc.get_shape().as_list()[1]]), [1,6,8,1])

            with tf.variable_scope('dec', reuse=reuse):
                with tf.variable_scope('h1'):
                    if 'skipcon' in self.conf:
                        enc = tf.concat([enc, self.enc3[t]], axis=3)
                    h1 = self.conv_relu_block(enc, out_ch=32 * ch_mult, upsmp=True)  # 12, 16
                with tf.variable_scope('h2'):
                    if 'skipcon' in self.conf:
                        h1 = tf.concat([h1, self.enc2[t]], axis=3)
                    h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult, upsmp=True)   # 24, 32
                with tf.variable_scope('h3'):
                    if 'skipcon' in self.conf:
                        h2 = tf.concat([h2, self.enc1[t]], axis=3)
                    h3 = self.conv_relu_block(h2, out_ch=3, upsmp=True)  # 48, 64
                with tf.variable_scope('h4'):
                    gen_images.append(slim.layers.conv2d(h3, 3, kernel_size=[3, 3], stride=1, activation_fn=tf.nn.sigmoid))

        return tf.stack(gen_images, axis=1)

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

    def build_image_summary(self, tensors=None, side_by_side=None, numex=8, name=None):
        """
        takes numex examples from every tensor and concatentes examples side by side
        and the different tensors from top to bottom
        :param tensors:
        :param numex:
        :return:
        """

        ten_list = []
        if side_by_side is not None:
            l_side, r_side = side_by_side

            for l, r in zip(l_side, r_side):
                if len(l.get_shape().as_list()) == 3 or l.get_shape().as_list()[-1] == 1:
                    l = colorize(l, tf.reduce_min(l), tf.reduce_max(l), 'viridis')
                l = tf.unstack(l, axis=0)[:numex]
                if len(r.get_shape().as_list()) == 3 or r.get_shape().as_list()[-1] == 1:
                    r = colorize(r, tf.reduce_min(r), tf.reduce_max(r), 'viridis')
                r = tf.unstack(r, axis=0)[:numex]
                lr = [tf.concat([l_, r_], axis=1) for l_, r_  in zip(l,r)]
                concated = tf.concat(lr, axis=1)
                ten_list.append(concated)
        else:
            for ten in tensors:
                if len(ten.get_shape().as_list()) == 3 or ten.get_shape().as_list()[-1] == 1:
                    ten = colorize(ten, tf.reduce_min(ten), tf.reduce_max(ten), 'viridis')
                unstacked = tf.unstack(ten, axis=0)[:numex]
                concated = tf.concat(unstacked, axis=1)
                ten_list.append(concated)

        combined = tf.concat(ten_list, axis=0)
        combined = tf.reshape(combined, [1]+combined.get_shape().as_list())

        if name ==None:
            name = 'Images'
        return tf.summary.image(name, combined)


    def create_loss_and_optimizer(self, reconstr_mean, I_intm, z_log_sigma_sq, z_mean, traintime):

        diff = reconstr_mean - I_intm
        if 'MSE' in self.conf:
            # unweighted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            reconstr_loss = tf.reduce_mean(tf.square(diff))
        else:
            reconstr_errs = charbonnier_loss(diff, per_int_ex=True)
            self.weights = tf.nn.softmax(1/(reconstr_errs +1e-6), -1)
            reconstr_loss = charbonnier_loss(diff, weights=self.weights)

            dist = tf.square(tf.cast(tf.range(1, self.seq_len - 1), tf.float32) - self.seq_len/2)
            weights_reg = dist[None]*self.weights
            self.loss_dict['tweights_reg'] = (tf.reduce_mean(weights_reg), self.conf['tweights_reg'])

        self.loss_dict['rec'] = (reconstr_loss, 1.)

        if traintime:
            latent_loss = 0.
            for i in range(self.seq_len -2):
                latent_loss += -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq[:,i] - tf.square(z_mean[:,i])
                                                    - tf.exp(z_log_sigma_sq[:,i]))
            if 'sched_lt_cost' in self.conf:
                mx_it = self.conf['sched_lt_cost'][1]
                min_it = self.conf['sched_lt_cost'][0]
                self.sched_lt_cost = tf.clip_by_value((self.iter_num-min_it)/(mx_it - min_it), 0., 1.)
                self.train_summaries.append(tf.summary.scalar('sched_lt_cost', self.sched_lt_cost))
            else:
                self.sched_lt_cost = 1.

            self.loss_dict['lt_loss'] = (latent_loss, self.conf['lt_cost_factor']*self.sched_lt_cost)

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

    print 'num_rows', num_rows
    print 'num_cols', num_cols

    width_per_ex = 2.5

    standard_size = np.array([width_per_ex * num_cols, num_rows * 1.5])  ### 1.5
    figsize = (standard_size).astype(np.int)

    f, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    print 'start'
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

    if 'I1' in dict:
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


if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/tdac_cons0_cartgripper/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')