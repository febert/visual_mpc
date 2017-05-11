# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA, DNA, and STP."""

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
import pdb

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12



def construct_model(conf,
                    input_images,
                    is_training = True,
                    ):
    if 'batch_norm' in conf:
        print 'using batchnorm'
        with slim.arg_scope(
                [slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                normalizer_params= {"is_training": is_training}, normalizer_fn= tf_layers.batch_norm
                ):
            recimage = feature_point_autoenc(conf, input_images)
    else:
        recimage = feature_point_autoenc(conf, input_images)
    return recimage

def feature_point_autoenc(conf, input_images):

    enc0 = slim.layers.conv2d(input_images, 32, [3, 3], stride=1, scope='conv0')  # 64x64x32
    enc1 = slim.layers.conv2d(enc0, 64, [3, 3], stride=1, scope='conv1')  # 64x64x64
    enc2 = slim.layers.conv2d(enc1, conf['num_fp'], [3, 3], stride=1, scope='conv2')  # 64x64x numfp

    _, num_rows, num_cols, num_fp = enc2.get_shape()
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

    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(enc2, [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)

    fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)

    fl0 = slim.layers.fully_connected(
        tf.reshape(tf.concat(1, [fp_x, fp_y]), [conf['batch_size'], -1]),
        3 * 64 ** 2,  # im_channels* width* height
        activation_fn=None,
        scope='state_enc3')

    fp_x = tf.reshape(fp_x, [conf['batch_size'], conf['num_fp'], 1])
    fp_y = tf.reshape(fp_y, [conf['batch_size'], conf['num_fp'], 1])
    fp_out = tf.concat(2, [fp_x, fp_y])

    output_images = tf.reshape(fl0, [conf['batch_size'], 64,64,3])

    return output_images, fp_out