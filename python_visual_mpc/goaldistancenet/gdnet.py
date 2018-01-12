import tensorflow as tf
import numpy as np
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm


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
                 images=None,
                 build_loss=True
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

        self.images = images

        if images ==None:
            from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn

            train_images, train_actions, train_states = build_tfrecord_fn(conf, training=True)
            val_images, val_actions, val_states = build_tfrecord_fn(conf, training=False)



            self.images, self.actions, self.states = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                              lambda: [train_images, train_actions, train_states],
                                              lambda: [val_images, val_actions, val_states])
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


    def conv_relu_block(self, input, channel_mult, k=3, strides=2, upsmp=False):
        if not upsmp:
            h = conv_pool2d(input, self.conf['basedim'] * channel_mult, kernel_size=(k, k),
                            strides=(strides, strides))  # 20x32x3
        else:
            h = upsample_conv2d(input, self.conf['basedim'] * channel_mult, kernel_size=(k, k),
                            strides=(strides, strides))  # 20x32x3

        h = self.normalizer_fn(h)
        h = tf.nn.relu(h)
        return h


    def build(self):
        self.I0, self.I1 = self.sel_images()

        I0_I1 = tf.concat([self.I0, self.I1], axis=3)

        with tf.variable_scope('h1'):
            h1 = self.conv_relu_block(I0_I1, channel_mult=1, strides=2)  #24x32x3

        with tf.variable_scope('h2'):
            h2 = self.conv_relu_block(h1, channel_mult=2, strides=2)  #12x16x3

        with tf.variable_scope('h3'):
            h3 = self.conv_relu_block(h2, channel_mult=3, strides=2)  #6x8x3

        with tf.variable_scope('h4'):
            h4 = self.conv_relu_block(h3, channel_mult=1, strides=2, upsmp=True)  #12x16x3

        with tf.variable_scope('h5'):
            h5 = self.conv_relu_block(h4, channel_mult=1, strides=2, upsmp=True)  #24x32x3

        with tf.variable_scope('h6'):
            h6 = self.conv_relu_block(h5, channel_mult=1, strides=2, upsmp=True)  #48x64x3

        with tf.variable_scope('h7'):
            self.flow_field = conv2d(h6, 2, [5,5])

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