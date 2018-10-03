import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import numpy as np


class BaseGoalClassifier:
    def __init__(self, conf, datasets=None):
        self._train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self._hp = self._default_hparams().override_from_dict(conf)

        if datasets is not None:
            self._goal_images = tf.cast(self._get_trainval_batch('goal_image', datasets), tf.float32)
            self._input_im = tf.cast(self._get_trainval_batch('final_frame', datasets), tf.float32)
            self._label = self._get_trainval_batch('label', datasets)
        else:
            raise NotImplementedError("Test functionality not implemented")

    def _get_trainval_batch(self, key, datasets):
        train_data = tf.concat([d[key] for d in datasets], 0)
        val_data = tf.concat([d[key, 'val'] for d in datasets], 0)
        return tf.cond(self._train_cond > 0, lambda: train_data, lambda: val_data)

    def _default_hparams(self):
        params = {
            'pretrained_path': '{}/weights'.format(os.environ['VMPC_DATA_DIR']),
            'max_steps': 80000
        }
        return HParams(**params)

    def build(self):
        print(self._vgg_layer(self._goal_images))

    def base_feed_dict(self, is_train):
        train_val = 0
        if is_train:
            train_val = 1
        return {self._train_cond: train_val}

    def _vgg_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _vgg_conv(self, vgg_dict, bottom, name):
        with tf.variable_scope(name):
            filt = tf.constant(vgg_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(vgg_dict[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _vgg_layer(self, images):
        """
        :param images: float32 RGB in range 0 - 255
        :return:
        """
        vgg_dict = np.load(os.path.join(self._hp.pretrained_path, "vgg19.npy"), encoding='latin1').item()
        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        red, green, blue = tf.split(axis=-1, num_or_size_splits=3, value=images)

        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        conv1_1 = self._vgg_conv(vgg_dict, bgr, "conv1_1")
        conv1_2 = self._vgg_conv(vgg_dict, conv1_1, "conv1_2")
        conv1_out = self._vgg_pool(conv1_2, "pool1")

        conv2_1 = self._vgg_conv(vgg_dict, conv1_out, "conv2_1")
        conv2_2 = self._vgg_conv(vgg_dict, conv2_1, "conv2_2")
        conv2_out = self._vgg_pool(conv2_2, "pool2")

        conv3_1 = self._vgg_conv(vgg_dict, conv2_out, "conv3_1")
        conv3_2 = self._vgg_conv(vgg_dict, conv3_1, "conv3_2")
        conv3_3 = self._vgg_conv(vgg_dict, conv3_2, "conv3_3")
        out = self._vgg_conv(vgg_dict, conv3_3, "conv3_4")

        return out

    @property
    def max_steps(self):
        return self._hp.max_steps
