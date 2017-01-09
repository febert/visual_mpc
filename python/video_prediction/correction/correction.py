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


def construct_correction(images,
                         num_masks=1,
                         stp=False,
                         cdna=False,
                         dna=True,
                         pix_distrib_input=None):
    """Build network for predicting optical flow

    """

    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')
    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    if pix_distrib_input != None:
        num_objects = pix_distrib_input.shape[1]

    concat_img = tf.concat(3, [images[0], images[1]])

    enc0 = slim.layers.conv2d(
        concat_img,
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

    if dna:
        # Using largest hidden state for predicting untied conv kernels.
        enc4 = slim.layers.conv2d_transpose(
            enc3, DNA_KERN_SIZE ** 2, 1, stride=1, scope='convt4'
        )

    prop_distrib = []
    summaries = []

    if dna:
        # Only one mask is supported (more should be unnecessary).
        if num_masks != 1:
            raise ValueError('Only one mask is supported for DNA model.')
        transformed = [dna_transformation(images[0], enc4)]
    else:
        raise ValueError

    masks = slim.layers.conv2d_transpose(
        enc3, num_masks + 1, 1, stride=1, scope='convt7')
    masks = tf.reshape(
        tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
        [int(batch_size), int(img_height), int(img_width), num_masks + 1])
    mask_list = tf.split(3, num_masks + 1, masks)
    output = mask_list[0] * images[0]
    for layer, mask in zip(transformed, mask_list[1:]):
        output += layer * mask
    gen_images= output
    gen_masks= mask_list

    if pix_distrib_input != None:

        for ob in range(num_objects):

            pix_distrib_input_ob = tf.slice(pix_distrib_input,
                                            begin=[0, ob, 0, 0], size=[-1, 1, -1, -1])
            if dna:
                transf_distrib = [dna_transformation(pix_distrib_input_ob, enc4)]
                pdb.set_trace()
            else:
                raise ValueError

            pix_distrib_output = mask_list[0] * pix_distrib_input_ob
            mult_list = []
            for i in range(num_masks):
                mult_list.append(transf_distrib[i] * mask_list[i+1])
                pix_distrib_output += mult_list[i]

            prop_distrib.append(pix_distrib_output)

        return gen_images, gen_masks, prop_distrib
    else:
        return gen_images, gen_masks, None



def dna_transformation(prev_image, dna_input):
    """Apply dynamic neural advection to previous image.

    Args:
      prev_image: previous image to be transformed.
      dna_input: hidden lyaer to be used for computing DNA transformation.
    Returns:
      List of images transformed by the predicted CDNA kernels.
    """
    # Construct translated images.
    prev_image_pad = tf.pad(prev_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
    image_height = int(prev_image.get_shape()[1])
    image_width = int(prev_image.get_shape()[2])

    inputs = []
    for xkern in range(DNA_KERN_SIZE):
        for ykern in range(DNA_KERN_SIZE):
            inputs.append(
                tf.expand_dims(
                    tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                             [-1, image_height, image_width, -1]), [3]))
    inputs = tf.concat(3, inputs)

    # Normalize channels to 1.
    kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
    kernel = tf.expand_dims(
        kernel / tf.reduce_sum(
            kernel, [3], keep_dims=True), [4])

    return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)
