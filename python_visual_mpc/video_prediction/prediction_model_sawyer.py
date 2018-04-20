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
from python_visual_mpc.video_prediction.lstm_ops12 import basic_conv_lstm_cell
from python_visual_mpc.misc.zip_equal import zip_equal

from python_visual_mpc.video_prediction.basecls.utils.transformations import dna_transformation, sep_dna_transformation
from python_visual_mpc.video_prediction.basecls.utils.transformations import cdna_transformation, sep_cdna_transformation

import pdb

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


class Prediction_Model(object):

    def __init__(self,
                images,
                actions=None,
                states=None,
                iter_num=-1.0,
                pix_distributions1=None,
                pix_distributions2=None,
                conf = None):

        self.pix_distributions1 = pix_distributions1
        self.pix_distributions2 = pix_distributions2
        self.actions = actions
        self.iter_num = iter_num
        self.conf = conf
        self.images = images

        self.cdna, self.stp, self.dna = False, False, False
        if self.conf['model'] == 'CDNA':
            self.cdna = True
        elif self.conf['model'] == 'DNA':
            self.dna = True
        elif self.conf['model'] == 'STP':
            self.stp = True
        if self.stp + self.cdna + self.dna != 1:
            raise ValueError("More than one option selected!")

        self.k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_masks = conf['num_masks']
        self.context_frames = conf['context_frames']

        self.batch_size, self.img_height, self.img_width, self.color_channels = [int(i) for i in
                                                                                 images[0].get_shape()[0:4]]
        self.lstm_func = basic_conv_lstm_cell

        # Generated robot states and images.
        self.gen_states = []
        self.gen_images = []
        self.gen_masks = []

        self.moved_images = []
        self.moved_bckgd = []

        self.moved_pix_distrib1 = []
        self.moved_pix_distrib2 = []

        self.states = states
        self.gen_distrib1 = []
        self.gen_distrib2 = []

        self.trafos = []
        self.movd_parts_list = []

        self.flow_vectors = []

        if 'separable_filters' in self.conf:
            print('applying separable filters')
            self.dna_transformation = sep_dna_transformation
            self.cdna_transformation = sep_cdna_transformation
        else:
            self.dna_transformation = dna_transformation
            self.cdna_transformation = cdna_transformation


    def build(self):

        if 'kern_size' in list(self.conf.keys()):
            KERN_SIZE = self.conf['kern_size']
        else:
            KERN_SIZE = 5

        batch_size, img_height, img_width, color_channels = self.images[0].get_shape()[0:4]
        lstm_func = basic_conv_lstm_cell


        if self.states != None:
            current_state = self.states[0]
        else:
            current_state = None

        if self.actions == None:
            self.actions = [None for _ in self.images]

        if self.k == -1:
            feedself = True
        else:
            # Scheduled sampling:
            # Calculate number of ground-truth frames to pass in.
            num_ground_truth = tf.to_int32(
                tf.round(tf.to_float(batch_size) * (self.k / (self.k + tf.exp(self.iter_num / self.k)))))
            feedself = False

        # LSTM state sizes and states.

        if 'lstm_size' in self.conf:
            lstm_size = self.conf['lstm_size']
            print('using lstm size', lstm_size)
        else:
            lstm_size = np.int32(np.array([16, 32, 64, 32, 16]))


        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None

        t = -1
        for image, action in zip(self.images[:-1], self.actions[:-1]):
            t +=1
            print(t)
            # Reuse variables after the first timestep.
            reuse = bool(self.gen_images)

            done_warm_start = len(self.gen_images) > self.context_frames - 1
            with slim.arg_scope(
                    [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                     tf_layers.layer_norm, slim.layers.conv2d_transpose],
                    reuse=reuse):

                if feedself and done_warm_start:
                    # Feed in generated image.
                    prev_image = self.gen_images[-1]             # 64x64x6
                    if self.pix_distributions1 != None:
                        prev_pix_distrib1 = self.gen_distrib1[-1]
                        if 'ndesig' in self.conf:
                            prev_pix_distrib2 = self.gen_distrib2[-1]
                elif done_warm_start:
                    # Scheduled sampling
                    prev_image = scheduled_sample(image, self.gen_images[-1], batch_size,
                                                  num_ground_truth)
                else:
                    # Always feed in ground_truth
                    prev_image = image
                    if self.pix_distributions1 != None:
                        prev_pix_distrib1 = self.pix_distributions1[t]
                        if 'ndesig' in self.conf:
                            prev_pix_distrib2 = self.pix_distributions2[t]
                        if len(prev_pix_distrib1.get_shape()) == 3:
                            prev_pix_distrib1 = tf.expand_dims(prev_pix_distrib1, -1)
                            if 'ndesig' in self.conf:
                                prev_pix_distrib2 = tf.expand_dims(prev_pix_distrib2, -1)

                if 'refeed_firstimage' in self.conf:
                    assert self.conf['model']=='STP'
                    if t > 1:
                        input_image = self.images[1]
                        print('refeed with image 1')
                    else:
                        input_image = prev_image
                else:
                    input_image = prev_image

                # Predicted state is always fed back in
                if not 'ignore_state_action' in self.conf:
                    state_action = tf.concat(axis=1, values=[action, current_state])

                enc0 = slim.layers.conv2d(    #32x32x32
                    input_image,
                    32, [5, 5],
                    stride=2,
                    scope='scale1_conv1',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm1'})

                hidden1, lstm_state1 = lstm_func(       # 32x32x16
                    enc0, lstm_state1, lstm_size[0], scope='state1')
                hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')

                enc1 = slim.layers.conv2d(     # 16x16x16
                    hidden1, hidden1.get_shape()[3], [3, 3], stride=2, scope='conv2')

                hidden3, lstm_state3 = lstm_func(   #16x16x32
                    enc1, lstm_state3, lstm_size[1], scope='state3')
                hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')

                enc2 = slim.layers.conv2d(  # 8x8x32
                    hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv3')

                if not 'ignore_state_action' in self.conf:
                    # Pass in state and action.
                    if 'ignore_state' in self.conf:
                        lowdim = action
                        print('ignoring state')
                    else:
                        lowdim = state_action

                    smear = tf.reshape(
                        lowdim,
                        [int(batch_size), 1, 1, int(lowdim.get_shape()[1])])
                    smear = tf.tile(
                        smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])

                    enc2 = tf.concat(axis=3, values=[enc2, smear])
                else:
                    print('ignoring states and actions')

                enc3 = slim.layers.conv2d(   #8x8x32
                    enc2, hidden3.get_shape()[3], [1, 1], stride=1, scope='conv4')

                hidden5, lstm_state5 = lstm_func(  #8x8x64
                    enc3, lstm_state5, lstm_size[2], scope='state5')
                hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
                enc4 = slim.layers.conv2d_transpose(  #16x16x64
                    hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

                hidden6, lstm_state6 = lstm_func(  #16x16x32
                    enc4, lstm_state6, lstm_size[3], scope='state6')
                hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')

                if 'noskip' not in self.conf:
                    # Skip connection.
                    hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16

                enc5 = slim.layers.conv2d_transpose(  #32x32x32
                    hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
                hidden7, lstm_state7 = lstm_func( # 32x32x16
                    enc5, lstm_state7, lstm_size[4], scope='state7')
                hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

                if not 'noskip' in self.conf:
                    # Skip connection.
                    hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

                enc6 = slim.layers.conv2d_transpose(   # 64x64x16
                    hidden7,
                    hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm9'})

                if 'transform_from_firstimage' in self.conf:
                    prev_image = self.images[1]
                    if self.pix_distributions1 != None:
                        prev_pix_distrib1 = self.pix_distributions1[1]
                        prev_pix_distrib1 = tf.expand_dims(prev_pix_distrib1, -1)
                    print('transform from image 1')

                if self.conf['model']=='DNA':
                    # Using largest hidden state for predicting untied conv kernels.

                    if 'separable_filters' in self.conf:
                        num_filters = KERN_SIZE*2
                    else:
                        num_filters = KERN_SIZE**2

                    trafo_input = slim.layers.conv2d_transpose(
                        enc6,num_filters, 1, stride=1, scope='convt4_cam2')

                    transformed_l, _ = self.dna_transformation(self.conf, prev_image, trafo_input)
                    if self.pix_distributions1 != None:
                        transf_distrib_ndesig1, _ = self.dna_transformation(self.conf, prev_pix_distrib1, trafo_input)
                        if 'ndesig' in self.conf:
                            transf_distrib_ndesig2, _ = self.dna_transformation(self.conf, prev_pix_distrib2, trafo_input)
                    extra_masks = 1

                if self.conf['model'] == 'CDNA':
                    if 'gen_pix' in self.conf:
                        enc7 = slim.layers.conv2d_transpose(
                            enc6, color_channels, 1, stride=1, scope='convt4')
                        transformed_l = [tf.nn.sigmoid(enc7)]
                        extra_masks = 2
                    else:
                        transformed_l = []
                        extra_masks = 1

                    if 'mov_bckgd' in self.conf:
                        extra_masks = self.num_masks

                    cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
                    new_transformed, cdna_kerns = self.cdna_transformation(self.conf, prev_image,
                                                            cdna_input,
                                                            reuse_sc=reuse)
                    transformed_l += new_transformed
                    self.moved_images.append(transformed_l)

                    ## move the background is chosen:
                    if 'mov_bckgd' in self.conf:
                        cdna_input = tf.reshape(hidden5, [int(self.batch_size), -1])
                        bckgd_transformed, _ = self.cdna_transformation(self.conf, self.images[0],
                                                                        cdna_input,
                                                                        reuse_sc=reuse,
                                                                        scope='bckgd_trafo')
                        self.moved_bckgd.append(bckgd_transformed)

                    if self.pix_distributions1 != None:
                        transf_distrib_ndesig1, _ = self.cdna_transformation(self.conf, prev_pix_distrib1,
                                                                                   cdna_input,
                                                                                     reuse_sc=True)
                        self.moved_pix_distrib1.append(transf_distrib_ndesig1)

                        if 'mov_bckgd' in self.conf:
                            bcgkd_distrib = tf.reshape(self.pix_distributions1[0], (self.batch_size,64,64,1))
                            transf_distrib_bckgd, _ = self.cdna_transformation(self.conf, bcgkd_distrib,
                                                                                 cdna_input,
                                                                                 reuse_sc=True,
                                                                                 scope='bckgd_trafo')
                        if 'ndesig' in self.conf:
                            transf_distrib_ndesig2, _ = self.cdna_transformation(self.conf,
                                                                                   prev_pix_distrib2,
                                                                                   cdna_input,
                                                                                   reuse_sc=True)

                            self.moved_pix_distrib2.append(transf_distrib_ndesig2)

                if '1stimg_bckgd' in self.conf:
                    background = self.images[0]
                    print('using background from first image..')
                else: background = prev_image

                if 'mov_bckgd' in self.conf:
                    output, mask_list, moved_parts = self.fuse_trafos_movbckgd(
                                                         enc6,
                                                         bckgd_transformed,
                                                         transformed_l,
                                                         scope='convt7_cam2',
                                                         extra_masks=extra_masks,
                                                         reuse=reuse)
                    self.movd_parts_list.append(moved_parts)
                else:
                    output, mask_list = self.fuse_trafos(enc6, background,
                                                         transformed_l,
                                                         scope='convt7_cam2',
                                                         extra_masks= extra_masks)
                self.gen_images.append(output)
                self.gen_masks.append(mask_list)

                if self.pix_distributions1!=None:
                    if 'mov_bckgd' in self.conf:
                        pix_distrib_output = self.fuse_pix_movebckgd(mask_list,
                                                                    transf_distrib_ndesig1,
                                                                    transf_distrib_bckgd)
                    else:
                        pix_distrib_output = self.fuse_pix_distrib(extra_masks,
                                                                    mask_list,
                                                                    self.pix_distributions1,
                                                                    prev_pix_distrib1,
                                                                    transf_distrib_ndesig1)

                    self.gen_distrib1.append(pix_distrib_output)

                    if 'ndesig' in self.conf:
                        pix_distrib_output = self.fuse_pix_distrib(extra_masks,
                                                                    mask_list,
                                                                    self.pix_distributions2,
                                                                    prev_pix_distrib2,
                                                                    transf_distrib_ndesig2)

                        self.gen_distrib2.append(pix_distrib_output)

                if 'visual_flowvec' in self.conf:
                    motion_vecs = self.compute_motion_vector(cdna_kerns)
                    output = tf.zeros([self.conf['batch_size'],64,64,2])
                    for vec, mask in zip(motion_vecs, mask_list[1:]):
                        vec = tf.reshape(vec, [self.conf['batch_size'], 1, 1, 2])
                        vec = tf.tile(vec, [1, 64,64, 1])
                        output += vec * mask

                    self.flow_vectors.append(output)

                if current_state != None:
                    current_state = slim.layers.fully_connected(
                        state_action,
                        int(current_state.get_shape()[1]),
                        scope='state_pred',
                        activation_fn=None)

                self.gen_states.append(current_state)


    def fuse_trafos(self, enc6, background_image, transformed, scope, extra_masks):
        masks = slim.layers.conv2d_transpose(
            enc6, (self.conf['num_masks']+ extra_masks), 1, stride=1, scope=scope)

        img_height = 64
        img_width = 64
        num_masks = self.conf['num_masks']

        if self.conf['model']=='DNA':
            if num_masks != 1:
                raise ValueError('Only one mask is supported for DNA model.')

        # the total number of masks is num_masks +extra_masks because of background and generated pixels!
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, num_masks +extra_masks])),
            [int(self.batch_size), int(img_height), int(img_width), num_masks +extra_masks])
        mask_list = tf.split(axis=3, num_or_size_splits=num_masks +extra_masks, value=masks)

        output = mask_list[0] * background_image
        for layer, mask in zip_equal(transformed, mask_list[1:]):
            output += layer * mask

        return output, mask_list

    def fuse_trafos_movbckgd(self, enc6, moved_background_image, transformed, scope, extra_masks, reuse):
        print('moving backgd')
        num_masks = self.conf['num_masks']
        img_height = 64
        img_width = 64

        ## moving the background

        masks = slim.layers.conv2d_transpose(
            enc6, (self.conf['num_masks']+ extra_masks), 1, stride=1, scope=scope)
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, num_masks +extra_masks])),
            [int(self.batch_size), int(img_height), int(img_width), num_masks +extra_masks])
        mask_list = tf.split(axis=3, num_or_size_splits=num_masks +extra_masks, value=masks)

        complete_transformed = moved_background_image + transformed

        output = 0
        moved_parts = []
        for layer, mask in zip_equal(complete_transformed, mask_list):
            moved_parts.append(layer * mask)
            output += layer * mask

        return output, mask_list, moved_parts

    def fuse_pix_distrib(self, extra_masks, mask_list, pix_distributions, prev_pix_distrib,
                         transf_distrib):

        if '1stimg_bckgd' in self.conf:
            background_pix = pix_distributions[0]
            background_pix = tf.expand_dims(background_pix, -1)
            print('using pix_distrib-background from first image..')
        else:
            background_pix = prev_pix_distrib
        pix_distrib_output = mask_list[0] * background_pix
        for i in range(self.num_masks):
            pix_distrib_output += transf_distrib[i] * mask_list[i + extra_masks]
        return pix_distrib_output

    def fuse_pix_movebckgd(self, mask_list, transf_pix, transf_backgd_pix):
        pix_distrib = transf_backgd_pix + transf_pix

        pix_distrib_output = 0
        for pix, mask in zip_equal(pix_distrib, mask_list):
            pix_distrib_output += pix* mask
        return pix_distrib_output

    def compute_motion_vector(self, cdna_kerns):

        range = self.conf['kern_size'] / 2
        dc = np.linspace(-range, range, num= self.conf['kern_size'])
        dc = np.expand_dims(dc, axis=0)
        dc = np.repeat(dc, self.conf['kern_size'], axis=0)
        dr = np.transpose(dc)
        dr = tf.constant(dr, dtype=tf.float32)
        dc = tf.constant(dc, dtype=tf.float32)

        cdna_kerns = tf.transpose(cdna_kerns, [2, 3, 0, 1])
        cdna_kerns = tf.split(cdna_kerns, self.conf['num_masks'], axis=1)
        cdna_kerns = [tf.squeeze(k) for k in cdna_kerns]

        vecs = []
        for kern in cdna_kerns:
            vec_r = tf.multiply(dr, kern)
            vec_r = tf.reduce_sum(vec_r, axis=[1,2])
            vec_c = tf.multiply(dc, kern)
            vec_c = tf.reduce_sum(vec_c, axis=[1, 2])

            vecs.append(tf.stack([vec_r,vec_c], axis=1))
        return vecs


    ## Utility functions
    def stp_transformation(self, prev_image, stp_input, num_masks, reuse= None, suffix = None):
        """Apply spatial transformer predictor (STP) to previous image.

        Args:
          prev_image: previous image to be transformed.
          stp_input: hidden layer to be used for computing STN parameters.
          num_masks: number of masks and hence the number of STP transformations.
        Returns:
          List of images transformed by the predicted STP parameters.
        """
        # Only import spatial transformer if needed.
        from python_visual_mpc.video_prediction.transformer.spatial_transformer import transformer

        identity_params = tf.convert_to_tensor(
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
        transformed = []
        trafos = []
        for i in range(num_masks):
            params = slim.layers.fully_connected(
                stp_input, 6, scope='stp_params' + str(i) + suffix,
                activation_fn=None,
                reuse= reuse) + identity_params
            outsize = (prev_image.get_shape()[1], prev_image.get_shape()[2])
            transformed.append(transformer(prev_image, params, outsize))
            trafos.append(params)

        return transformed, trafos


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data_files points.

    Args:
      ground_truth_x: tensor of ground-truth data_files points.
      generated_x: tensor of generated data_files points.
      batch_size: batch size
      num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)
    return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])
