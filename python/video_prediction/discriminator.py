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
from lstm_ops import basic_conv_lstm_cell

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA


def construct_discriminator(images, conf = None):


    enc0 = slim.layers.conv2d(
        images,
        32, [5, 5],
        stride=2,
        scope='conv0',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm1'}
    )

    enc1 = slim.layers.conv2d(
        enc0,
        64, [5, 5],
        stride=2,
        scope='conv1',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm2'}
    )

    enc2 = slim.layers.conv2d_transpose(
        enc1,
        32, [5, 5],
        stride=2,
        scope='t_conv1',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm2'}
    )

    enc3 = slim.layers.conv2d_transpose(
        enc2,
        32, [5, 5],
        stride=2,
        scope='t_conv2',
        # normalizer_fn=tf_layers.batch_norm,
        # normalizer_params={'scope': 'batch_norm2'}
    )

    enc4 = slim.layers.fully_connected(
        enc3,
        100,
        scope='state_pred',
        activation_fn=None)

    enc5 = slim.layers.fully_connected(
        enc4,
        2,
        scope='state_pred',
        activation_fn=None)

