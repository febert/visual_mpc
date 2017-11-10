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



class Trafo_Flow(object):
    def __init__(self,
                 conf,
                 images,
                 pix_distrib_input=None,
                 reuse=None):
        """Build network for predicting optical flow
        """

        num_masks = conf['num_masks']

        batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
        if pix_distrib_input != None:
            num_objects = pix_distrib_input.get_shape()[1]

        concat_img = tf.concat([images[0], images[1]], axis=3)  #64x64x3

        if 'separable_filters' in conf:
            print 'applying separable filters'
            from python_visual_mpc.video_prediction.basecls.utils.transformations import\
                sep_dna_transformation as dna_transformation
            from python_visual_mpc.video_prediction.basecls.utils.transformations import\
                sep_cdna_transformation as cdna_transformation
        else:
            from python_visual_mpc.video_prediction.basecls.utils.transformations import \
                dna_transformation as dna_transformation
            from python_visual_mpc.video_prediction.basecls.utils.transformations import \
                cdna_transformation as cdna_transformation

        with slim.arg_scope(
                [slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if 'large_core' in conf:
                final_layer, middle_layer = build_large_core(concat_img)
                print 'using large core'
            else:
                final_layer, middle_layer = build_core(concat_img)

            if conf['model'] == 'DNA':

                # Using largest hidden state for predicting untied conv kernels.
                if 'separable_filters' in conf:
                    num_filters = conf['kern_size'] * 2
                else: num_filters = conf['kern_size'] ** 2
                enc4 = slim.layers.conv2d_transpose(final_layer, num_filters, 1, stride=1, scope='convt4')

                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed, self.kernels  = dna_transformation(conf, images[0], enc4)

            elif conf['model'] == 'CDNA':
                cdna_input = tf.reshape(middle_layer, [int(batch_size), -1])
                transformed, self.kernels = cdna_transformation(conf, images[0], cdna_input, scope='track_cdna', reuse_sc= reuse)

            masks = slim.layers.conv2d_transpose(
                final_layer, num_masks + 1, 1, stride=1, scope='convt7')
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
                [int(batch_size), int(img_height), int(img_width), num_masks + 1])
            mask_list = tf.split(masks,num_masks + 1, axis=3)
            output = mask_list[0] * images[0]
            for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask
            self.gen_images= output
            self.gen_masks= mask_list

            if 'visual_flowvec' in conf:
                if conf['model'] == 'DNA':
                    motion_vecs = compute_motion_vector_dna(conf, self.kernels)
                elif conf['model'] == 'CDNA':
                    motion_vecs = compute_motion_vector_cdna(conf, self.kernels)

                output = tf.zeros([conf['batch_size'], 64, 64, 2])
                for vec, mask in zip(motion_vecs, mask_list[1:]):
                    if conf['model'] == 'CDNA':
                        vec = tf.reshape(vec, [conf['batch_size'], 1, 1, 2])
                        vec = tf.tile(vec, [1, 64, 64, 1])
                    output += vec * mask
                self.flow_vectors01 = output
            else:
                self.flow_vectors01 = None

            if pix_distrib_input != None:
                if conf['model'] == 'DNA':
                    transf_distrib, self.kernels = dna_transformation(conf, pix_distrib_input, enc4)
                else:
                    transf_distrib, self.kernels = cdna_transformation(conf,
                                                                pix_distrib_input,
                                                                cdna_input,
                                                                scope = 'track_cdna',
                                                                reuse_sc=True)

                self.gen_distrib_output = mask_list[0] * pix_distrib_input
                for i in range(num_masks):
                    self.gen_distrib_output += transf_distrib[i] * mask_list[i + 1]
            else:
                self.gen_distrib_output = None

def build_core(concat_img):

    enc0 = slim.layers.conv2d(   #32x32x32
                concat_img,
                32, [5, 5],
                stride=2,
                scope='conv0',
            )

    enc1 = slim.layers.conv2d( #16x16x64
        enc0,
        64, [5, 5],
        stride=2,
        scope='conv1',
    )

    enc2 = slim.layers.conv2d_transpose(  #32x32x32
        enc1,
        32, [5, 5],
        stride=2,
        scope='t_conv1',
    )

    enc3 = slim.layers.conv2d_transpose( # 64x64x32
        enc2,
        32, [5, 5],
        stride=2,
        scope='t_conv2',
    )

    return enc3, enc1


def build_large_core(concat_img):
    enc0 = slim.layers.conv2d(  # 32x32x32
        concat_img,
        32, [5, 5],
        stride=2,
        scope='conv0',
    )

    enc1 = slim.layers.conv2d(  # 32x32x32
        enc0,
        32, [5, 5],
        stride=1,
        scope='conv1',
    )

    enc2 = slim.layers.conv2d(  # 16x16x64
        enc1,
        64, [5, 5],
        stride=2,
        scope='conv2',
    )

    enc3 = slim.layers.conv2d_transpose(  # 32x32x32
        enc2,
        32, [5, 5],
        stride=2,
        scope='t_conv1',
    )

    enc4 = slim.layers.conv2d_transpose(  # 32x32x32
        enc3,
        32, [5, 5],
        stride=2,
        scope='t_conv2',
    )

    enc5 = slim.layers.conv2d_transpose(  # 64x64x32
        enc4,
        32, [5, 5],
        stride=2,
        scope='t_conv3',
    )

    return enc5, enc2

def compute_motion_vector_cdna(conf, cdna_kerns):

    range = conf['kern_size'] / 2
    dc = np.linspace(-range, range, num= conf['kern_size'])
    dc = np.expand_dims(dc, axis=0)
    dc = np.repeat(dc, conf['kern_size'], axis=0)
    dr = np.transpose(dc)
    dr = tf.constant(dr, dtype=tf.float32)
    dc = tf.constant(dc, dtype=tf.float32)

    cdna_kerns = tf.transpose(cdna_kerns, [2, 3, 0, 1])
    cdna_kerns = tf.split(cdna_kerns, conf['num_masks'], axis=1)
    cdna_kerns = [tf.reshape(k, [conf['batch_size'], conf['kern_size'], conf['kern_size']]) for k in cdna_kerns]

    vecs = []
    for kern in cdna_kerns:
        vec_r = tf.multiply(dr, kern)
        vec_r = tf.reduce_sum(vec_r, axis=[1,2])
        vec_c = tf.multiply(dc, kern)
        vec_c = tf.reduce_sum(vec_c, axis=[1, 2])

        vecs.append(tf.stack([vec_r,vec_c], axis=1))
    return vecs

def compute_motion_vector_dna(conf, dna_kerns):

    range = conf['kern_size'] / 2
    dc = np.linspace(-range, range, num= conf['kern_size'])
    dc = np.expand_dims(dc, axis=0)
    dc = np.repeat(dc, conf['kern_size'], axis=0)

    dc = dc.reshape([1,1,conf['kern_size'],conf['kern_size']])
    dc = np.repeat(dc, 64, axis=0)
    dc = np.repeat(dc, 64, axis=1)

    dr = np.transpose(dc, [0,1,3,2])

    dr = tf.constant(dr, dtype=tf.float32)
    dc = tf.constant(dc, dtype=tf.float32)

    dna_kerns = tf.reshape(dna_kerns, [conf['batch_size'], 64,64,conf['kern_size'],conf['kern_size']])

    dr = tf.expand_dims(dr, axis=0)
    dc = tf.expand_dims(dc, axis=0)

    vec_r = tf.multiply(dr, dna_kerns)
    vec_r = tf.reduce_sum(vec_r, axis=[3,4])
    vec_c = tf.multiply(dc, dna_kerns)
    vec_c = tf.reduce_sum(vec_c, axis=[3,4])

    vec_c = tf.expand_dims(vec_c, axis=-1)
    vec_r = tf.expand_dims(vec_r, axis=-1)

    flow = tf.concat([vec_r, vec_c], axis=-1)  # size: [conf['batch_size'], 64, 64, 2]

    return [flow]