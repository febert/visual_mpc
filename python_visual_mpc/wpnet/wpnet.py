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

def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def charbonnier_loss(x, weights=None, alpha=0.45, beta=1.0, epsilon=0.001):
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
        if weights is not None:
            error *= weights.reshape([batch, 1,1, channels])

        return tf.reduce_sum(error) / (normalization + 1e-6)


class WaypointNet(object):
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
        self.nz = 8

        self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        self.seq_len = self.conf['sequence_length']
        self.bsize = self.conf['batch_size']

        self.img_height = self.conf['orig_size'][0]
        self.img_width = self.conf['orig_size'][1]

        if load_data:
            self.iter_num = tf.placeholder(tf.float32, [], name='iternum')

            tag_images = {'name': 'images',
                          'file': '/images/im{}.png',  # only tindex
                          'shape': [48, 64, 3]}
            conf['sequence_length'] = 30,
            conf['ngroup'] = 1000
            conf['sourcetags'] = [tag_images]
            sess = tf.InteractiveSession()

            r = OnlineReader(conf, 'train', sess=sess)
            train_image_batch = r.get_batch_tensors()

            r = OnlineReader(conf, 'val', sess=sess)
            val_image_batch = r.get_batch_tensors()

            r = OnlineReader(conf, 'test', sess=sess)
            self.test_images = r.get_batch_tensors()

            self.images = tf.cond(self.train_cond > 0,
                             # if 1 use trainigbatch else validation batch
                             lambda: train_image_batch,
                             lambda: val_image_batch)

            self.Istart = self.images[:,0]
            self.I_intm = self.images[1:self.conf['sequence_lenght'] - 1]
            self.Igoal = self.images[:, -1]

        elif images == None:  #feed values at test time
            self.Istart = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.Igoal = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
        self.build_loss = build_loss
        self.train_losses = {}

    def build_net(self):
        # Encode our data into z, generate images, using train or val data
        with tf.variable_scope("train_model"):
            self.build_vae(traintime=True)

        # sample from prior, no encoder
        with tf.variable_scope("train_model"):  # Encode our data into z and return the mean and covariance
            self.build_vae(traintime=False)

    def build_vae(self, traintime):
        self.ctxt_encoding = self.ctxt_enc(self.Istart, self.Igoal)
        if traintime:
            self.z_mean, self.z_log_sigma_sq, self.tweights = self.intm_enc(self.I_intm)
            self.z = []
            for i in range(self.seq_len -2):
                eps = tf.random_normal([self.bsize, self.nz], 0.0, 1.0, dtype=tf.float32)
                self.z.append(tf.add(self.z_mean[:,i], tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq[:, i])), eps)))
            self.z = tf.stack(self.z, axis=1)
        else:
            for i in range(self.seq_len -2):
                eps = tf.random_normal([self.bsize, self.nz], 0.0, 1.0, dtype=tf.float32)
                self.z.append(eps)
            self.z = tf.stack(self.z, axis=1)

        # Get the reconstructed mean from the decoder
        self.x_reconstr_mean = self.decode(self.ctxt_encoding, self.z)
        self.z_summary = tf.summary.histogram("z", self.z)
        self.tweights_summary = tf.summary.histogram("tweights", self.tweights)

        self.create_loss_and_optimizer(self.x_reconstr_mean, self.I_intm,
                                       self.tweights, self.z_log_sigma_sq, self.z_mean)

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
        else: ch_mult = 1

        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(intm_images, out_ch=32*ch_mult)  #24x32x3
        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64*ch_mult)  #12x16x3
        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, out_ch=128*ch_mult)  #6x8x3

        mu = slim.layers.fully_connected(h3.reshape[self.bsize, -1], num_outputs=self.nz*(self.seq_len-2))
        mu = tf.reshape(mu, [self.bsize, self.seq_len, self.nz])
        log_sigma_diag = slim.layers.fully_connected(h3.reshape[self.bsize, -1], num_outputs=self.nz*(self.seq_len-2))
        log_sigma_diag = tf.reshape(log_sigma_diag, [self.bsize, self.seq_len, self.nz])
        timeweights = slim.layers.fully_connected(h3.reshape[self.bsize, -1], num_outputs=self.seq_len-2)

        return mu, log_sigma_diag, timeweights

    def ctxt_enc(self, start_im, goal_im):
        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
        else: ch_mult = 1

        I0_I1 = tf.concat([start_im, goal_im], axis=3)
        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(I0_I1, out_ch=32*ch_mult)  #24x32x3
        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64*ch_mult)  #12x16x3
        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, out_ch=128*ch_mult)  #6x8x3

        ctxt_enc = slim.layers.fully_connected(h3.reshape[self.bsize, -1], num_outputs=self.nz)
        return ctxt_enc

    def decode(self, ctxt_enc, z):
        """
        warps I0 onto I1
        :param start_im:
        :param goal_im:
        :return:
        """
        enc = tf.concat([ctxt_enc, z], axis=1)
        if 'ch_mult' in self.conf:
            ch_mult = self.conf['ch_mult']
        else:
            ch_mult = 1
        enc = tf.tile(tf.reshape(enc,[self.bsize, 1,1,self.nz]), [1,6,8,1])
        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(enc, out_ch=32 * ch_mult, upsmp=True)  # 12, 16
        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult, upsmp=True)   # 24, 32
        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, out_ch=(self.seq_len - 2)*3, upsmp=True)  # 48, 64
        return tf.split(h3, self.seq_len - 1, axis=-1)

    def build_image_summary(self, tensors, numex=16, name=None):
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

        if name ==None:
            name = 'Images'
        return tf.summary.image(name, combined)


    def create_loss_and_optimizer(self, reconstr_mean, I_intm, timeweights, z_log_sigma_sq, z_mean):
        self.train_summaries = {}
        self.val_summaries = {}

        reconstr_loss = charbonnier_loss(reconstr_mean - I_intm, weights=timeweights)
        self.train_summaries['rec'] = reconstr_loss
        self.val_summaries['rec'] = reconstr_loss

        #regularization for time weights:
        expected_t = tf.reduce_sum(tf.range(1,self.seq_len -2)[None, :]*timeweights)
        tweights_reg = tf.square((self.seq_len - 2)/2 - expected_t)*self.conf['tweights_reg']
        self.train_summaries['tweights_reg'] = tweights_reg
        self.val_summaries['tweights_reg'] = tweights_reg

        # TODO: verify why is there no product for the determinant?
        latent_loss = 0
        for i in range(self.seq_len -2):
            latent_loss += -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq[:,i] - tf.square(z_mean[:,i])
                                                - tf.exp(z_log_sigma_sq[:,i]), 1)
        self.train_summaries['lt_loss'] = latent_loss
        self.val_summaries['lt_loss'] = latent_loss
        self.combine_losses()


    def combine_losses(self):

        train_summaries = []
        val_summaries = []
        self.train_loss = 0
        for k in self.train_losses.keys():
            single_loss = self.train_losses[k]
            self.train_loss += single_loss
            train_summaries.append(tf.summary.scalar(k, single_loss))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.train_loss)
        train_summaries.append(tf.summary.scalar('train_total', self.train_loss))
        self.train_summ_op = tf.summary.merge(train_summaries)

        val_loss = 0
        for k in self.val_losses.keys():
            single_loss = self.val_losses[k]
            val_loss += single_loss
            train_summaries.append(tf.summary.scalar(k, single_loss))

        val_summaries.append(tf.summary.scalar('val_total', val_loss))
        self.val_summ_op = tf.summary.merge(val_summaries)


    def visualize(self, sess):

        videos = collections.OrderedDict()
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