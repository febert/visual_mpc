import tensorflow as tf

import copy
from python_visual_mpc.wpnet.visualize import make_plots
import pickle
import collections
import numpy as np
from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn

from .metric_basecls import Metric_Basecls

from python_visual_mpc.video_prediction.utils_vpred.online_reader import OnlineReader
import tensorflow.contrib.slim as slim

from python_visual_mpc.utils.colorize_tf import colorize

from python_visual_mpc.layers.batchnorm_layer import batchnorm_train, batchnorm_test
from python_visual_mpc.wpnet.operations import lrelu

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


class WaypointNet(Metric_Basecls):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True,
                 images = None,
                 inference = False,
                 sess = None,
                 pref = '',
                 ):

        self._hp = self._default_hparams().override_from_dict(conf)
        
        if self._hp.normalization == 'in':
            self.normalizer_fn = instance_norm
        if self._hp.normalization == 'bnorm':
            self.normalizer_fn = batchnorm_train
        elif self._hp.normalization == 'None':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % conf['normalization'])

        self.pref = pref
        self.conf = conf
        self.nz = self._hp.lt_dim

        self.lr = tf.placeholder_with_default(self._hp.learning_rate, ())
        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        self.seq_len = self._hp.sequence_length//self._hp.skip_frame
        self.n_intm = self.seq_len - 2
        self.bsize = self._hp.batch_size

        self.img_height = self._hp.orig_size[0]
        self.img_width = self._hp.orig_size[1]

        self.train_summaries = []
        self.val_summaries = []

        self.masks = []

        if 'vgg19' in self.conf:
            self.vgg_dict = np.load(conf['vgg19'], encoding='latin1').item()

        if load_data:
            self.iter_num = tf.placeholder(tf.float32, [], name='iternum')
            modconf = copy.deepcopy(conf)
            modconf['data_dir'] = conf['data_dir']['var_scence']
            self.images = self.get_data(modconf)
            modconf['data_dir'] = conf['data_dir']['stat_scence']
            self.images_statscene = self.get_data(modconf)

        elif images == None:  #feed values at test time
            self.Istart = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.Igoal = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
        self.build_loss = build_loss
        self.loss_dict = {}

    def _default_hparams(self):
        default_dict = {



        }

        parent_params = super()._default_hparams()
        parent_params.set_hparam('ncam', 2)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def get_data(self, conf):
        train_dict = build_tfrecord_fn(conf, mode='train')
        val_dict = build_tfrecord_fn(conf, mode='val')
        dict = tf.cond(self.train_cond > 0,
                       # if 1 use trainigbatch else validation batch
                       lambda: train_dict,
                       lambda: val_dict)
        return dict['images']

    def build_net(self, traintime = True, reuse = False):
        with tf.variable_scope('model', reuse=reuse):
            self.build_vae(traintime)


    def build_vae(self, traintime):
        self.z = []

        if 'enc_avg_pool' in self.conf:
            eps = tf.random_normal([self.bsize, self.n_intm*self.nz*self._hp.enc_avg_pool[0]*self._hp.enc_avg_pool[1]], 0.0, 1.0, dtype=tf.float32)
            eps = tf.reshape(eps, [self.bsize, self.n_intm] + self._hp.enc_avg_pool + [self.nz])
        else:
            eps = tf.random_normal([self.bsize, self.nz], 0.0, 1.0, dtype=tf.float32)

        if traintime:
            self.z_mean, self.z_log_sigma_sq = self.encode(self.I_intm)
            if 'deterministic' in self.conf:
                self.z = self.z_mean
                print('latent not sampled!! deterministic latent')
            elif 'three_stage_training' in self.conf:
                self.z = tf.cond(self.iter_num < self._hp.three_stage_training,
                                  lambda: self.z_mean,
                                  lambda: self.z_mean + tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
            else:
                self.z = self.z_mean + tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps)
        else:
            self.z_mean, self.z_log_sigma_sq = None, None
            self.z = eps

        # Get the reconstructed mean from the decoder
        self.x_reconstr_mean = self.decode(self.z)

        intm = tf.unstack(self.I_intm, axis=1)
        x_reconstr_mean = tf.unstack(self.x_reconstr_mean, axis=1)
        self.image_summaries = self.build_image_summary(side_by_side=[[self.Istart] + intm + [self.Igoal],
                                                                      [self.Istart] + x_reconstr_mean + [self.Igoal]])

        self.image_summaries = tf.summary.merge([self.image_summaries])
        self.make_histo({'z_mean':self.z_mean, 'z_log_sigma':self.z_log_sigma_sq, 'z_samples':self.z})

        self.create_loss_and_optimizer(self.x_reconstr_mean, self.I_intm, self.z_log_sigma_sq, self.z_mean, traintime)

    def sel_images(self):
        sequence_length = self._hp.sequence_length
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
        h = slim.layers.conv2d(input, out_ch, [k, k], stride=1, activation_fn=lambda x: lrelu(x, 0.1))
        h = self.normalizer_fn(h)

        if upsmp:
            mult = 2
        else: mult = 0.5
        imsize = np.array(h.get_shape().as_list()[1:3])*mult

        h = tf.image.resize_images(h, imsize, method=tf.image.ResizeMethod.BILINEAR)
        return h

    def encode(self, intm_images):
        if 'ch_mult' in self.conf:
            ch_mult = self._hp.ch_mult
            print('using chmult', ch_mult)
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
                        if 'inference_use_cont' in self.conf:
                            inp = tf.concat([intm_images[:, t], self.Istart, self.Igoal], axis=3)
                        else:
                            inp = intm_images[:, t]
                        h1 = self.conv_relu_block(inp, out_ch=32 * ch_mult)  # 24x32x3
                with tf.variable_scope('h2'):
                    h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult)  # 12x16x3

                with tf.variable_scope('h3'):
                    h3 = self.conv_relu_block(h2, out_ch=64 * ch_mult)  # 6x8x3
                with tf.variable_scope('h4'):
                    h4_mu = slim.layers.conv2d(h3, self.nz, [3, 3], stride=1)
                    h4_sigma = slim.layers.conv2d(h3, self.nz, [3, 3], stride=1)
                mu_l.append(tf.image.resize_images(h4_mu, self._hp.enc_avg_pool, method=tf.image.ResizeMethod.BILINEAR))
                log_sigma_diag_l.append(tf.image.resize_images(h4_sigma, self._hp.enc_avg_pool, method=tf.image.ResizeMethod.BILINEAR))

        return tf.stack(mu_l,axis=1), tf.stack(log_sigma_diag_l,axis=1)

    def decode(self, z):
        """
        warps I0 onto I1
        :param start_im:
        :param goal_im:
        :return:
        """
        if 'ch_mult' in self.conf:
            ch_mult = self._hp.ch_mult
        else:
            ch_mult = 1
        gen_images = []

        for t in range(self.seq_len -2):
            if  t== 0:
                reuse = False
            else: reuse = True
            with tf.variable_scope('dec', reuse=reuse):
                enc = z[:,t]
                if self._hp.enc_avg_pool == [1,1]:
                    enc = tf.tile(enc, [1,3,4,1])

                if self._hp.enc_avg_pool != [6,8]:
                    with tf.variable_scope('h0'):
                        enc = self.conv_relu_block(enc, out_ch=32 * ch_mult, upsmp=True)  # 6, 8

                with tf.variable_scope('h1'):
                    if 'skipcon' in self.conf:
                        enc = tf.concat([enc, self.enc3], axis=3)
                    h1 = self.conv_relu_block(enc, out_ch=32 * ch_mult, upsmp=True)  # 12, 16
                with tf.variable_scope('h2'):
                    if 'skipcon' in self.conf:
                        h1 = tf.concat([h1, self.enc2], axis=3)
                    h2 = self.conv_relu_block(h1, out_ch=64 * ch_mult, upsmp=True)   # 24, 32
                with tf.variable_scope('h3'):
                    if 'skipcon' in self.conf:
                        h2 = tf.concat([h2, self.enc1], axis=3)
                    h3 = self.conv_relu_block(h2, out_ch=32* ch_mult, upsmp=True)  # 48, 64
                with tf.variable_scope('h4'):
                    gen_image = slim.layers.conv2d(h3, 3, kernel_size=[3, 3], stride=1, activation_fn=tf.nn.sigmoid)

                if 'condcpy' in self.conf:
                    with tf.variable_scope('masks'):
                        h3 = self.conv_relu_block(h2, out_ch=32* ch_mult, upsmp=True)  # 48, 64
                        h4_m = slim.layers.conv2d(h3, 3, [3,3], stride=1)
                        masks = tf.nn.softmax(h4_m)
                        self.masks.append(masks)
                        masks = tf.split(masks, 3, axis=-1)
                    gen_images.append(masks[0]*self.Istart + masks[1]*gen_image + masks[2]*self.Igoal)
                else:
                    gen_images.append(gen_image)
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

    def make_histo(self, hist_dict):
        self.train_hist_sum = []
        self.val_hist_sum = []
        for k in list(hist_dict.keys()):
            if hist_dict[k] is not None:
                self.train_hist_sum.append(tf.summary.histogram(self.pref + '/train_' + k, hist_dict[k]))
                self.val_hist_sum.append(tf.summary.histogram(self.pref + '/val_' + k, hist_dict[k]))


    def build_image_summary(self, tensors=None, side_by_side=None, numex=8, name=None):
        """
        takes numex examples from every tensor and concatentes examples side by side
        and the different tensors from top to bottom

        side_by_side: list of list of timesteps of size [batchsize, w, h, ch]
        :param tensors:
        :param numex:
        :return:
        """
        ten_list = []
        if side_by_side is not None:

            for t in range(len(side_by_side[0])):
                t_lists = []
                for list_el in side_by_side:
                    list_el_t = list_el[t]
                    if len(list_el_t.get_shape().as_list()) == 3 or list_el_t.get_shape().as_list()[-1] == 1:
                        list_el_t = colorize(list_el_t, tf.reduce_min(list_el_t), tf.reduce_max(list_el_t), 'viridis')
                    t_lists.append(tf.unstack(list_el_t[:numex], axis=0))
                row = tf.concat([tf.concat(l, axis=1) for l in zip(*t_lists)], axis=1)
                ten_list.append(row)
        else:
            for ten in tensors:
                if len(ten.get_shape().as_list()) == 3 or ten.get_shape().as_list()[-1] == 1:
                    ten = colorize(ten, tf.reduce_min(ten), tf.reduce_max(ten), 'viridis')
                unstacked = tf.unstack(ten, axis=0)[:numex]
                concated = tf.concat(unstacked, axis=1)
                ten_list.append(concated)

        combined = tf.concat(ten_list, axis=0)
        combined = tf.reshape(combined, [1]+combined.get_shape().as_list())

        if name == None:
            name = 'Images'
        return tf.summary.image(self.pref + '/' + name, combined)


    def create_loss_and_optimizer(self, reconstr_mean, I_intm, z_log_sigma_sq, z_mean, traintime):
        diff = reconstr_mean - I_intm
        if 'MSE' in self.conf:
            if 'min_loss' in self.conf:
                reconstr_loss = tf.reduce_mean(tf.square(diff), [2,3,4])
                self.loss_dict['rec'] = (tf.reduce_mean(reconstr_loss), 0.)
                self.weights = tf.nn.softmax(1/(reconstr_loss + 1e-6), -1)
                reconstr_loss = tf.reduce_mean(reconstr_loss*self.weights)
                self.loss_dict['rec_weighted'] = (reconstr_loss, 1.)
            else:
                reconstr_loss = tf.reduce_mean(tf.square(diff))
                self.loss_dict['rec'] = (reconstr_loss, 1.)
        else:
            # reconstr_errs = charbonnier_loss(diff, per_int_ex=True)
            # self.weights = tf.nn.softmax(1/(reconstr_errs +1e-6), -1)
            # reconstr_loss = charbonnier_loss(diff, weights=self.weights)
            #
            # dist = tf.square(tf.cast(tf.range(1, self.seq_len - 1), tf.float32) - self.seq_len/2)
            # weights_reg = dist[None]*self.weights
            # self.loss_dict['tweights_reg'] = (tf.reduce_mean(weights_reg), self._hp.tweights_reg'])
            reconstr_loss = charbonnier_loss(diff)
            self.loss_dict['rec'] = (reconstr_loss, 1.)


        if traintime:
            latent_loss = 0.
            for i in range(self.seq_len -2):
                latent_loss += -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + z_log_sigma_sq[:,i] - tf.square(z_mean[:,i])
                                                    - tf.exp(z_log_sigma_sq[:,i]), axis=[1,2,3]))
            if 'sched_lt_cost' in self.conf:
                mx_it = self._hp.sched_lt_cost[1]
                min_it = self._hp.sched_lt_cost[0]
                if 'lt_cost_factor_start' in self.conf:
                    clip_low = self._hp.lt_cost_factor_start
                else:
                    clip_low = 0.

                self.sched_lt_cost = tf.clip_by_value((self.iter_num-min_it)/(mx_it - min_it), clip_low, 1.)
                self.train_summaries.append(tf.summary.scalar('sched_lt_cost', self.sched_lt_cost))
            else:
                self.sched_lt_cost = 1.

            self.loss_dict['lt_loss'] = (latent_loss, self._hp.lt_cost_factor*self.sched_lt_cost)

        self.combine_losses()

    def combine_losses(self):
        self.loss = 0.
        for k in list(self.loss_dict.keys()):
            single_loss, weight = self.loss_dict[k]
            self.loss += single_loss*weight
            self.train_summaries.append(tf.summary.scalar(self.pref + 'train_' + k, single_loss))
            self.val_summaries.append(tf.summary.scalar(self.pref + 'val_' + k, single_loss))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.train_summaries.append(tf.summary.scalar('train_total', self.loss))
        self.train_summ_op = tf.summary.merge(self.train_summaries + [self.train_hist_sum])

        self.val_summaries.append(tf.summary.scalar('val_total', self.loss))
        self.val_summ_op = tf.summary.merge(self.val_summaries + [self.train_hist_sum])

    def color_code(self, input, num_examples):
        cmap = plt.cm.get_cmap()

        l = []
        for b in range(num_examples):
            f = input[b] / (np.max(input[b]) + 1e-6)
            f = cmap(f)[:, :, :3]
            l.append(f)
        return np.stack(l, axis=0)

    def visualize(self, sess):

        # Run through validation set.
        feed_dict = {self.iter_num: np.float32(100000),
                     self.train_cond: 0}

        [images, gen_images] = sess.run([self.images, self.x_reconstr_mean, self.weights], feed_dict)

        dict = collections.OrderedDict()
        dict['images'] = images
        dict['gen_images'] = gen_images

        pickle.dump(dict, open(self._hp.output_dir'] + '/data.pkl', 'wb'))
        make_plots(self.conf, dict=dict)
