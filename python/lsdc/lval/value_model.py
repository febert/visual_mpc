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
import pdb

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


def construct_model(conf, images, goalpos, desig_pos,
                    pix_distrib_input=None):

    """
    Build network for estimating value function
    """
    batch_size = conf['batch_size']
    img_height = 64
    img_width = 64
    color_channels = 3
    images = tf.reshape(images, [batch_size, img_height, img_width, color_channels])

    batch_size = int(batch_size)
    if pix_distrib_input != None:
        num_objects = pix_distrib_input.get_shape()[1]


    # images, size 64x64

    enc0 = slim.layers.conv2d(    # enc0 32x32
        images,
        32, [5, 5],
        stride=2,
        scope='conv0',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm1'}
    )

    enc1 = slim.layers.conv2d(   # enc1 16x16
        enc0,
        64, [5, 5],
        stride=2,
        scope='conv1',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm2'}
    )

    enc2 = slim.layers.conv2d(    #enc2 8x8
        enc1,
        128, [5, 5],
        stride=2,
        scope='conv2',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm2'}
    )

    # Pass in low-dimensional inputs:
    low_dim = tf.concat(1, [goalpos, desig_pos])
    smear = tf.reshape(low_dim, [batch_size, 1, 1, int(low_dim.get_shape()[1])])
    smear = tf.tile(smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1], name='tile')

    enc2_concat = tf.concat(3, [enc2, smear], name='concat')

    enc3  = slim.layers.conv2d(   #enc2 8x8
        enc2_concat,
        128, [5, 5],
        stride=1,
        scope='conv3',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm2'}
    )

    enc3_flat = tf.reshape(enc3, shape=[batch_size, -1], name='reshape')

    enc4 = slim.layers.fully_connected(
        enc3_flat,
        100,
        scope='fc1'
    )

    enc5 = slim.layers.fully_connected(
        enc4,
        100,
        scope='fc2'
    )

    value = slim.layers.fully_connected(
        enc5,
        1,
        scope='out'
    )

    return value

