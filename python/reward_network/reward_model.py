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
                    images_0,
                    images_1 = None,
                    is_training = True,
                    ):

    if 'batch_norm' in conf:
        print 'using batchnorm'
        with slim.arg_scope(
                [slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                normalizer_params= {"is_training": is_training}, normalizer_fn= tf_layers.batch_norm
                ):
            logits, fp1, fp2 = build_model(conf, images_0, images_1)
    else:
        logits, fp1, fp2 = build_model(conf, images_0, images_1)

    return logits, fp1, fp2


def build_model(conf, images_0, images_1):
    fp1, fp2 = None, None
    with tf.variable_scope('emb0'):
        if 'num_fp' in conf:
            print 'using feature points'
            fp0 = get_fp(conf, images_0)
            emb0 = tf.reshape(fp0, [conf['batch_size'], -1])
        else:
            emb0 = gen_embedding(conf, images_0)
    with tf.variable_scope('emb1'):
        if 'num_fp' in conf:
            print 'using feature points'
            fp1 = get_fp(conf, images_1)
            emb1 = tf.reshape(fp1, [conf['batch_size'], -1])
        else:
            emb1 = gen_embedding(conf, images_1)

    joint_embedding = tf.concat(1, [emb0, emb1])

    fl1 = slim.layers.fully_connected(
        joint_embedding,
        400,
        scope='state_enc1')

    if 'dropout' in conf:
        fl1 = tf.nn.dropout(fl1, conf['dropout'])
        print 'using dropout with p = ', conf['dropout']

    fl2 = slim.layers.fully_connected(
        fl1,
        200,
        scope='state_enc2')

    if 'dropout' in conf:
        fl2 = tf.nn.dropout(fl2, conf['dropout'])

    fl3 = slim.layers.fully_connected(
        fl2,
        conf['sequence_length'] - 1,
        activation_fn=None,
        scope='state_enc3')

    return fl3, fp1, fp2


def gen_embedding(conf, input):
    enc0 = slim.layers.conv2d(  # 32x32x32
        input,
        32, [5, 5],
        stride=2,
        scope='conv0',
    )

    if 'maxpool' in conf:
        enc0_maxp = slim.layers.max_pool2d(enc0, [2, 2], stride=[2, 2])
        enc1 = slim.layers.conv2d(enc0_maxp, 64, [3, 3], stride=1, scope='conv1')  #16x16x64
    else:
        enc1 = slim.layers.conv2d(enc0, 64, [3, 3], stride=2, scope='conv1')

    enc2 = slim.layers.conv2d(  # 16x16x64
        enc1,
        64, [3, 3],
        stride=1,
        scope='conv2')

    if 'maxpool' in conf:
        enc2_maxp = slim.layers.max_pool2d(enc2, [2,2], stride=[2,2])
        enc3 = slim.layers.conv2d(enc2_maxp, 128, [3, 3], stride=1, scope='conv3')  #8x8x128
    else:
        enc3 = slim.layers.conv2d(enc2, 128, [3, 3], stride=2, scope='conv3')  # 8x8x128

    enc4 = slim.layers.conv2d(  # 8x8x64
        enc3,
        64, [3, 3],
        stride=1,
        scope='conv4',
    )

    emb = tf.reshape(enc4, [int(enc4.get_shape()[0]), -1])
    return emb

def get_fp(conf, input_images):

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

    fp_x = tf.reshape(fp_x, [conf['batch_size'], conf['num_fp'], 1])
    fp_y = tf.reshape(fp_y, [conf['batch_size'], conf['num_fp'], 1])
    fp_out = tf.concat(2, [fp_x, fp_y])

    return fp_out
