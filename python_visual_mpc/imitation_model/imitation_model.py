import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.contrib.slim as slim
import numpy as np
import os



NUMERICAL_EPS = 1e-7

class ImitationBaseModel:
    def __init__(self, conf, images, actions, end_effector):
        self.conf = conf

        #input images
        self.images = images
        #ground truth
        self.gtruth_actions = actions
        self.gtruth_endeffector_pos = end_effector

        assert ('adim' in self.conf), 'must specify action dimension in conf wiht key adim'
        assert (self.conf['adim'] == self.gtruth_actions.get_shape()[2]), 'conf adim does not match tf record'
        assert ('sdim' in self.conf), 'must specify state dimension in conf with key sdim'
        assert (self.conf['sdim'] == self.gtruth_endeffector_pos.get_shape()[2]), 'conf sdim does not match tf record'

        self.sdim, self.adim = self.conf['sdim'], self.conf['adim']

        vgg19_path = './'
        if 'vgg19_path' in conf:
            vgg19_path = conf['vgg19_path']

        #for vgg layer
        self.vgg_dict = np.load(os.path.join(vgg19_path, "vgg19.npy"), encoding='latin1').item()

    def build(self):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, tf_layers.layer_norm]):
            in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()

            input_images = tf.reshape(self.images, shape = (in_batch * in_time, in_rows, in_cols, 3))
            raw_input_action = tf.reshape(self.gtruth_actions, shape = (in_batch * in_time, self.adim))
            raw_input_splits = tf.split(raw_input_action, self.adim, axis = -1)
            raw_input_splits[-1] = raw_input_splits[-1] / 100
            input_action = tf.concat(raw_input_splits, axis = -1)

            input_end_effector = tf.reshape(self.gtruth_endeffector_pos, shape=(in_batch * in_time, self.sdim))

            layer1 = tf_layers.layer_norm(self.vgg_layer(input_images), scope='conv1_norm')
            layer2 = tf_layers.layer_norm(
                slim.layers.conv2d(layer1, 32, [3, 3], stride=2, scope='conv2'), scope='conv2_norm')

            layer3 = tf_layers.layer_norm(
                slim.layers.conv2d(layer2, 32, [3, 3], stride=2, scope='conv3'), scope='conv3_norm')

            batch_size, num_rows, num_cols, num_fp = layer3.get_shape()
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

            features = tf.reshape(tf.transpose(layer3, [0, 3, 1, 2]), [-1, num_rows * num_cols])
            softmax = tf.nn.softmax(features)

            fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
            fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
            fp_flat = tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])


            # self.predicted_eeps = slim.layers.fully_connected(fp_flat, self.sdim, scope='predicted_state',
            #                                                   activation_fn=None)

            conv_out = tf.concat([fp_flat,
                                  input_end_effector],
                                 1)

            last_fc = slim.layers.fully_connected(conv_out, 100, scope='fc1')


            if 'MDN_loss' in self.conf:
                num_mix = self.conf['MDN_loss']
                mixture_activations = slim.layers.fully_connected(last_fc, (self.adim + 2) * num_mix,
                                                                  scope = 'predicted_mixtures',activation_fn=None)
                mixture_activations = tf.reshape(mixture_activations, shape = (-1, num_mix, self.adim + 2))
                self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
                self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
                self.means = mixture_activations[:, :, 2:]

                gtruth_mean_sub = tf.reduce_sum(
                            tf.square(self.means - tf.reshape(input_action, shape = (-1, 1, self.adim))), axis = -1)

                self.likelihoods = tf.exp(-0.5 * gtruth_mean_sub / tf.square(self.std_dev)) / self.std_dev / np.power(2 * np.pi, self.adim / 2) * self.mixing_parameters
                self.loss = - tf.reduce_sum(tf.log(tf.reduce_sum(self.likelihoods, axis=-1) + NUMERICAL_EPS)) / self.conf['batch_size']

                mix_mean = tf.reduce_sum(self.means* tf.reshape(self.mixing_parameters, shape=(-1, num_mix, 1)), axis = 1)

                self.diagnostic_l2loss = tf.reduce_sum(tf.square(input_action - mix_mean)) / self.conf['batch_size']

            else:
                self.predicted_actions = slim.layers.fully_connected(last_fc, self.adim, scope='predicted_actions',
                                                                     activation_fn=None)
                total_loss = tf.reduce_sum(tf.square(input_action - self.predicted_actions)) \
                            + 0.5 * tf.reduce_sum(tf.abs(input_action - self.predicted_actions))
                self.loss = total_loss / float(self.conf['batch_size'] * int(in_time))




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


