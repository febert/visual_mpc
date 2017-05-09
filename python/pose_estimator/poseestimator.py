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
                    images,
                    is_training
                    ):

    if 'batch_norm' in conf:
        print 'using batchnorm'
        with slim.arg_scope(
                [slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                normalizer_params= {"is_training": is_training}, normalizer_fn= tf_layers.batch_norm
                ):
            logits = build_model(conf, images)
    else:
        logits = build_model(conf, images)

    return logits


def build_model(conf, images_0):
    emb = gen_embedding(conf, images_0)


    fl1 = slim.layers.fully_connected(
        emb,
        400,
        scope='state_enc1')

    fl2 = slim.layers.fully_connected(
        fl1,
        200,
        scope='state_enc2')

    fl3 = slim.layers.fully_connected(
        fl2,
        3,
        activation_fn=None,
        scope='state_enc3')

    return fl3


def gen_embedding(conf, input):
    enc0 = slim.layers.conv2d(  # 32x32x32
        input,
        32, [5, 5],
        stride=2,
        scope='conv0',
    )

    enc1 = slim.layers.conv2d(enc0,
        64, [3, 3],
        stride=2,
        scope='conv1')

    enc2 = slim.layers.conv2d(  # 16x16x64
        enc1,
        64, [3, 3],
        stride=1,
        scope='conv2')

    enc3 = slim.layers.conv2d(enc2,
      128,
      [3, 3],
      stride=2,
      scope='conv3')  # 8x8x128

    enc4 = slim.layers.conv2d(  # 8x8x64
        enc3,
        64, [3, 3],
        stride=1,
        scope='conv4',
    )

    emb = tf.reshape(enc4, [int(enc4.get_shape()[0]), -1])
    return emb