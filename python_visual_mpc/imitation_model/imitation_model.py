import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.contrib.slim as slim
import numpy as np
import os



NUMERICAL_EPS = 1e-7

class ImitationBaseModel:
    def __init__(self, conf, images, actions, end_effector):
        self.conf = conf
        assert ('adim' in self.conf), 'must specify action dimension in conf wiht key adim'
        assert (self.conf['adim'] == actions.get_shape()[2]), 'conf adim does not match input actions'
        assert ('sdim' in self.conf), 'must specify state dimension in conf with key sdim'
        assert (self.conf['sdim'] == end_effector.get_shape()[2]), 'conf sdim does not match input end_effector'
        self.sdim, self.adim = self.conf['sdim'], self.conf['adim']

        vgg19_path = './'
        if 'vgg19_path' in conf:
            vgg19_path = conf['vgg19_path']

        # for vgg layer
        self.vgg_dict = np.load(os.path.join(vgg19_path, "vgg19.npy"), encoding='latin1').item()

        #input images
        self.images = images

        raw_input_splits = tf.split(actions, self.adim, axis=-1)
        raw_input_splits[-1] = raw_input_splits[-1] / 100
        self.gtruth_actions = tf.concat(raw_input_splits, axis=-1)

        # ground truth ep
        self.gtruth_endeffector_pos = end_effector

    def _build_conv_layers(self, input_images):
        layer1 = tf_layers.layer_norm(self.vgg_layer(input_images), scope='conv1_norm')
        layer2 = tf_layers.layer_norm(
            slim.layers.conv2d(layer1, 32, [3, 3], stride=2, scope='conv2'), scope='conv2_norm')

        layer3 = tf_layers.layer_norm(
            slim.layers.conv2d(layer2, 32, [3, 3], stride=2, scope='conv3'), scope='conv3_norm')
        layer4 = tf_layers.layer_norm(
            slim.layers.conv2d(layer3, 48, [3, 3], stride=1, scope='conv4'), scope='conv4_norm')
        layer5 = tf_layers.layer_norm(
            slim.layers.conv2d(layer4, 64, [3, 3], stride=1, scope='conv5'), scope='conv5_norm')

        batch_size, num_rows, num_cols, num_fp = layer5.get_shape()
        # print 'shape', layer3.get_shape
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]

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

        features = tf.reshape(tf.transpose(layer5, [0, 3, 1, 2]), [-1, num_rows * num_cols])
        softmax = tf.nn.softmax(features)

        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
        return tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])

    def _build_loss(self, last_fc, input_action, T):
        _, input_dim = input_action.get_shape()
        input_dim = int(input_dim)

        self.debug_input = input_action

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            mixture_activations = slim.layers.fully_connected(last_fc, (input_dim + 2) * num_mix,
                                                              scope='predicted_mixtures', activation_fn=None)
            mixture_activations = tf.reshape(mixture_activations, shape=(-1, num_mix, input_dim + 2))
            self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
            self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
            self.means = mixture_activations[:, :, 2:]

            if 'round_down' in self.conf:
                thresh_mask = tf.abs(self.means) > self.conf['round_down']
                self.means = tf.cast(thresh_mask, self.means.dtype) * self.means

            gtruth_mean_sub = tf.reduce_sum(
                tf.square(self.means - tf.reshape(input_action, shape=(-1, 1, input_dim))), axis=-1)

            self.likelihoods = tf.exp(-0.5 * gtruth_mean_sub / tf.square(self.std_dev)) / self.std_dev / np.power(
                2 * np.pi, input_dim / 2.) * self.mixing_parameters
            self.lg_likelihoods = tf.log(tf.reduce_sum(self.likelihoods, axis=-1) + NUMERICAL_EPS)
            self.MDN_log_l = tf.reduce_sum(tf.log(tf.reduce_sum(self.likelihoods, axis=-1) + NUMERICAL_EPS)) \
                            / self.conf['batch_size'] / float(int(T))
            self.loss = - self.MDN_log_l

            mix_mean = tf.reduce_sum(self.means * tf.reshape(self.mixing_parameters, shape=(-1, num_mix, 1)), axis=1)

            self.diagnostic_l2loss = tf.reduce_sum(tf.square(input_action - mix_mean)) / self.conf['batch_size']

        else:
            self.predicted_actions = slim.layers.fully_connected(last_fc, input_dim, scope='predicted_actions',
                                                                 activation_fn=None)
            if 'round_down' in self.conf:
                thresh_mask = tf.abs(self.predicted_actions) > self.conf['round_down']
                self.predicted_actions = tf.cast(thresh_mask, self.predicted_actions.dtype) * self.predicted_actions

            total_loss = tf.reduce_sum(tf.square(input_action - self.predicted_actions)) \
                         + 0.5 * tf.reduce_sum(tf.abs(input_action - self.predicted_actions))
            self.action_loss = total_loss / float(self.conf['batch_size'] * int(T))
            self.loss = self.action_loss


    def build(self):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, tf_layers.layer_norm]):
            in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()

            input_images = tf.reshape(self.images, shape = (in_batch * in_time, in_rows, in_cols, 3))
            input_action = tf.reshape(self.gtruth_actions, shape = (in_batch * in_time, self.adim))
            input_end_effector = tf.reshape(self.gtruth_endeffector_pos, shape=(in_batch * in_time, self.sdim))

            fp_flat = self._build_conv_layers(input_images)

            conv_out = tf.concat([fp_flat,
                                  input_end_effector],
                                 1)

            last_fc = slim.layers.fully_connected(conv_out, 100, scope='fc1')

            self._build_loss(last_fc, input_action, in_time)


    # Source: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
    def vgg_layer(self, images):
        bgr_scaled = tf.to_float(images)

        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        blue, green, red = tf.split(axis=-1, num_or_size_splits=3, value=bgr_scaled)


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

        return out


class SimpleModel(ImitationBaseModel):
    def build(self):
        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        # get first images
        start_image = self.images[:, 0, :, :, :]
        #get final end effector
        #start_xy = self.gtruth_endeffector_pos[:, 0, :2]
        target = self.gtruth_endeffector_pos[:, 0, :4]
        #target = tf.concat([start_xy, final_conf], axis=1)

        fp_flat = self._build_conv_layers(start_image)
        layer_1 = slim.layers.fully_connected(fp_flat, 100)
        layer_2 = slim.layers.fully_connected(layer_1, 50)
        self._build_loss(layer_2, target, 1)
        self.final_frame_aux_loss = self.loss

        if 'MDN_loss' in self.conf:
            self.loss += self.diagnostic_l2loss


class ImitationLSTMModel(ImitationBaseModel):
    def build(self):

        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        #images are flattened for convolution
        input_images = tf.reshape(self.images, shape=(in_batch * in_time, in_rows, in_cols, 3))
        #actions are flattened for loss
        input_action = tf.reshape(self.gtruth_actions, shape = (in_batch * in_time, self.adim))
        #but configs are not (fed into lstm)
        input_end_effector = self.gtruth_endeffector_pos

        fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))

        #see if model can predict task from first frame
        first_frame_features = fp_flat[:, 0, :]
        self.final_frame_state_pred = slim.layers.fully_connected(first_frame_features, self.sdim,
                                    scope='predicted_final', activation_fn=None)
        raw_final_loss = tf.reduce_sum(tf.square(self.final_frame_state_pred - input_end_effector[:, -1, :self.sdim])) \
                         + 0.5 * tf.reduce_sum(tf.abs(self.final_frame_state_pred - input_end_effector[:, -1, :self.sdim]))
        self.final_frame_aux_loss = raw_final_loss / float(self.conf['batch_size'])

        final_pred_broadcast = tf.tile(tf.reshape(self.final_frame_state_pred,
                                                  shape = (in_batch, 1, self.sdim)),
                                       [1,in_time, 1])

        lstm_in = tf.concat([fp_flat, input_end_effector[:, :, :self.sdim], final_pred_broadcast],-1)

        lstm_layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(l) for l in self.conf['lstm_layers']])
        last_fc, states = tf.nn.dynamic_rnn(cell = lstm_layers, inputs = lstm_in,
                                            dtype = tf.float32, parallel_iterations=int(in_batch))
        last_fc = tf.reshape(last_fc, shape=(in_batch * in_time, -1))

        self._build_loss(last_fc, input_action, in_time)

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
            self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
            self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.adim))
        else:
            self.predicted_actions = tf.reshape(self.predicted_actions, shape=(in_batch, in_time, self.adim))

        self.loss += self.final_frame_aux_loss

class ImitationLSTMModelState(ImitationBaseModel):
    def build(self):

        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1
        #images are flattened for convolution
        input_images = tf.reshape(self.images[:, :-1,:,:,:], shape=(in_batch * in_time, in_rows, in_cols, 3))
        #actions are flattened for loss
        #but configs are not (fed into lstm)
        output_end_effector = tf.reshape(self.gtruth_endeffector_pos[:, 1:], shape = (in_batch * in_time, self.sdim))

        fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))

        #see if model can predict task from first frame
        first_initial = slim.layers.fully_connected(self.gtruth_endeffector_pos[:, 0, :], self.conf['lstm_layers'][0],
                                    scope='first_hidden', activation_fn = tf.tanh)
        lstm_in = fp_flat

        lstm_layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(l)) for l in self.conf['lstm_layers']])

        all_initial = tuple([tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, self.conf['lstm_layers'][0])), first_initial)]
                            + [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, l)), tf.zeros((in_batch, l)))
                               for l in self.conf['lstm_layers'][1:]])

        last_fc, states = tf.nn.dynamic_rnn(cell = lstm_layers, inputs = lstm_in, initial_state=all_initial,
                                            dtype = tf.float32, parallel_iterations=int(in_batch))
        last_fc = tf.reshape(last_fc, shape=(in_batch * in_time, -1))

        self._build_loss(last_fc, output_end_effector, in_time)

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
            self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
            self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim))
        else:
            self.predicted_actions = tf.reshape(self.predicted_actions, shape=(in_batch, in_time, self.sdim))


        #self.loss += 0.1 * self.diagnostic_l2loss
