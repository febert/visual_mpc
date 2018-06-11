from python_visual_mpc.imitation_model.attention_models.base_model import BaseAttentionModel
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.seq2seq as seq2seq
class AttentionGoalImages(BaseAttentionModel):
    def __init__(self, conf, data_dict):
        input_images = data_dict['images'][:, :-2]
        goal_images = data_dict['images'][:, -2:]
        actions = data_dict['actions'][:, :-2]
        endeffector_pos = data_dict['endeffector_pos'][:, :-2]


        super().__init__(conf, input_images, actions, endeffector_pos, goal_images)

    # Source: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
    def _vgg_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _vgg_conv(self, bottom, name):
        with tf.variable_scope(name):
            filt = tf.constant(self.vgg_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(self.vgg_dict[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu
    def vgg_layer(self, images):
        if images.dtype is tf.uint8:
            print('read uint8 images')
            bgr_scaled = tf.to_float(images)
        else:
            print('read float images')
            bgr_scaled = images * 255.

        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        blue, green, red = tf.split(axis=-1, num_or_size_splits=3, value=bgr_scaled)

        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        conv1_1 = self._vgg_conv(bgr, "conv1_1")
        conv1_2 = self._vgg_conv(conv1_1, "conv1_2")
        conv1_out = self._vgg_pool(conv1_2, "pool1")

        conv2_1 = self._vgg_conv(conv1_out, "conv2_1")
        conv2_2 = self._vgg_conv(conv2_1, "conv2_2")
        conv2_out = self._vgg_pool(conv2_2, "pool2")

        conv3_1 = self._vgg_conv(conv2_out, "conv3_1")
        conv3_2 = self._vgg_conv(conv3_1, "conv3_2")
        conv3_3 = self._vgg_conv(conv3_2, "conv3_3")
        out = conv3_4 = self._vgg_conv(conv3_3, "conv3_4")

        return out

    def _build_conv_layers(self, input_images):
        vgg_layers = tf_layers.layer_norm(self.vgg_layer(input_images[:, :, :, ::-1]), scope='vgg_norm')

        layer2 = tf_layers.layer_norm(
            slim.layers.conv2d(vgg_layers, self.num_feats, [3, 3], stride=1, scope='conv2'), scope='conv2_norm')

        batch_size, num_rows, num_cols, num_fp = layer2.get_shape()
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

        features = tf.reshape(tf.transpose(layer2, [0, 3, 1, 2]), [-1, num_rows * num_cols])
        softmax = tf.nn.softmax(features)

        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
        return tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])

    def build(self, is_Train=True):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        in_batch, in_time, in_cam, in_rows, in_cols, _ = self.images.get_shape()

        first_flat = tf.reshape(self.images[:, 0], shape=(in_batch * in_cam, in_rows, in_cols, 3))
        goal_flat = tf.reshape(self.goal_image, shape=(in_batch * in_cam * 2, in_rows, in_cols, 3))
        conv_in = tf.concat((first_flat, goal_flat), 0)
        conv_features = self._build_conv_layers(conv_in)

        fp_first = tf.reshape(conv_features[:in_batch * in_cam], shape = (in_batch, 1, self.num_feats * 2 * self.conf['ncam']))
        fp_goal = tf.reshape(conv_features[in_batch * in_cam:], shape=(in_batch, 2, self.num_feats * 2 * self.conf['ncam']))

        num_units = self.num_feats * 2 * self.conf['ncam']

        fc_in = tf.reshape(tf.concat((fp_goal, fp_first), 1), shape=(in_batch, -1))
        fc_outs = []
        for i in range(in_time // 3):
            fc_out = slim.layers.fully_connected(fc_in, 20,
                                        scope='fc_{}'.format(i), activation_fn=tf.nn.relu)
            fc_outs.append(fc_out)

        fc_out = tf.concat(fc_outs, 0)

        input_action = []
        for i in range(in_time // 3):
            input_action.append(tf.reshape(self.gtruth_actions[:, i * 3], shape=(in_batch, 1, -1)))
        input_action = tf.reshape(tf.concat(input_action, 1), shape=(-1, self.adim))

        self._build_loss(fc_out, input_action, in_time)

        num_mix = self.conf['MDN_loss']
        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time // 3, num_mix))
        self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time // 3, num_mix))
        self.means = tf.reshape(self.means, shape=(in_batch, in_time // 3, num_mix, self.adim))


