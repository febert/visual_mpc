import tensorflow as tf
import numpy as np
import Image
import sys
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers

def coords(h, w, batch_size):
    y = tf.cast(tf.range(h), tf.float32)
    x = tf.cast(tf.range(w), tf.float32)

    X,Y = tf.meshgrid(x,y)
    partial_tile_shape = tf.constant([1,1,1])
    tile_shape = tf.concat([tf.reshape(batch_size, [-1]), partial_tile_shape], 0)
    coords = tf.tile(tf.expand_dims(tf.stack((Y,X), axis=2), axis=0), tile_shape)
    return coords

def resample_layer(src_img, warp_pts, name="tgt_img"):
    with tf.variable_scope(name):
        return tf.contrib.resampler.resampler(src_img, warp_pts)

def warp_pts_layer(flow_field, name="warp_pts"):
    with tf.variable_scope(name):
        img_shape = tf.shape(flow_field)
        return flow_field + coords(img_shape[1], img_shape[2], img_shape[0])

def mean_squared_error(true, pred):
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))

class GoalDistanceNet(object):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True
                 ):

        self.layer_normalization = conf['normalization']
        if conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        elif conf['normalization'] == 'None':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % self.layer_normalization)

        self.conf = conf
        self.iter_num = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())
        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        if load_data:
            train_images, train_actions, train_states = build_tfrecord_fn(conf, training=True)
            val_images, val_actions, val_states = build_tfrecord_fn(conf, training=False)



            self.images, self.actions, self.states = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                              lambda: [train_images, train_actions, train_states],
                                              lambda: [val_images, val_actions, val_states])

            self.I0, self.I1 = self.sel_images()

        else:
            self.img_height = conf['row_end'] - conf['row_start']
            self.img_width = 64
            self.I0 = self.I0_pl= tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], self.img_height, self.img_width, 3))
            self.I1 = self.I1_pl= tf.placeholder(tf.float32, name='images',
                                     shape=(conf['batch_size'], self.img_height, self.img_width, 3))
        self.build()

        if build_loss:
            self.build_loss()


    def sel_images(self):
        sequence_length = self.conf['sequence_length']
        delta_t = tf.cast(tf.ceil(sequence_length * (self.iter_num + 1) / 100), dtype=tf.int32)  #################
        delta_t = tf.clip_by_value(delta_t, 1, sequence_length-1)

        self.tstart = tf.random_uniform([1], 0, sequence_length - delta_t, dtype=tf.int32)

        self.tend = self.tstart + tf.random_uniform([1], tf.ones([], dtype=tf.int32), delta_t + 1, dtype=tf.int32)

        begin = tf.stack([0, tf.squeeze(self.tstart), 0, 0, 0],0)
        I0 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        begin = tf.stack([0, tf.squeeze(self.tend), 0, 0, 0], 0)
        I1 = tf.squeeze(tf.slice(self.images, begin, [-1, 1, -1, -1, -1]))

        return I0, I1


    # def conv_relu_block(self, input, channel_mult, k=3, strides=2, upsmp=False):
    #     if not upsmp:
    #         h = conv_pool2d(input, self.conf['basedim'] * channel_mult, kernel_size=(k, k),
    #                         strides=(strides, strides))  # 20x32x3
    #     else:
    #         h = upsample_conv2d(input, self.conf['basedim'] * channel_mult, kernel_size=(k, k),
    #                         strides=(strides, strides))  # 20x32x3
    #
    #     h = self.normalizer_fn(h)
    #     h = tf.nn.relu(h)
    #     return h

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


    def build(self):
        I0_I1 = tf.concat([self.I0, self.I1], axis=3)

        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(I0_I1, out_ch=32)  #24x32x3

        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, out_ch=64)  #12x16x3

        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, out_ch=128)  #6x8x3

        with tf.variable_scope('h4'):
            h4 = self.conv_relu_block(h3, out_ch=64, upsmp=True)  #12x16x3

        with tf.variable_scope('h5'):
            h5 = self.conv_relu_block(h4, out_ch=32, upsmp=True)  #24x32x3

        with tf.variable_scope('h6'):
            h6 = self.conv_relu_block(h5, out_ch=16, upsmp=True)  #48x64x3

        with tf.variable_scope('h7'):
            self.flow_field = slim.layers.conv2d(  # 128x128xdesc_length
                h6,  2, [5, 5], stride=1, activation_fn=None)

        self.warp_pts = warp_pts_layer(self.flow_field)
        self.gen_image = resample_layer(self.I0, self.warp_pts)

    def build_loss(self):

        summaries = []
        self.loss = mean_squared_error(self.gen_image, self.I1)
        summaries.append(tf.summary.scalar('train_recon_cost', self.loss))
        self.train_summ_op = tf.summary.merge(summaries)

        summaries.append(tf.summary.scalar('val_recon_cost', self.loss))
        self.val_summ_op = tf.summary.merge(summaries)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def visualize(self, sess):

        train_images, train_actions, train_states = build_tfrecord_fn(self.conf)

        images = sess.run(train_images)

        warped_images = []

        num_examples = self.conf['batch_size']
        I1 = images[:, -1]


        for t in range(14):
            I0_t = images[:, t]

            [output, flow] = sess.run([self.gen_image, self.flow_field], {self.I0_pl:I0_t, self.I1_pl: I1})

            flow_mag = np.linalg.norm(flow, axis=3)
            flow_mag = np.squeeze(np.split(flow_mag, num_examples, axis=0))
            cmap = plt.cm.get_cmap('jet')
            flow_mag_ = []

            for b in range(num_examples):
                f = flow_mag[b]/(np.max(flow_mag[b]) + 1e-6)
                f = cmap(f)[:, :, :3]
                flow_mag_.append(f)
            flow_mag = flow_mag_
            plt.imshow(flow_mag[0])

            im_height = output.shape[1]
            im_width = output.shape[2]
            warped_column = np.squeeze(np.split(output, num_examples, axis=0))

            input_column = I0_t
            input_column = np.squeeze(np.split(input_column, num_examples, axis=0))

            warped_column = [np.concatenate([inp, warp, fmag], axis=0) for inp, warp, fmag in
                             zip(input_column, warped_column, flow_mag)]

            warped_column = np.concatenate(warped_column, axis=0)

            warped_images.append(warped_column)

        warped_images = np.concatenate(warped_images, axis=1)

        dir = self.conf['output_dir']
        Image.fromarray((warped_images*255.).astype(np.uint8)).save(dir + '/warped.png')

        I1 = I1.reshape(num_examples*im_height, im_width,3)
        Image.fromarray((I1 * 255.).astype(np.uint8)).save(dir + '/finalimage.png')

        sys.exit('complete!')
