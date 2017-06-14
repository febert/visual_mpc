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
from video_prediction.lstm_ops import basic_conv_lstm_cell

import pdb


# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
DNA_KERN_SIZE = 9

class Occlusion_Model(object):
    def __init__(self,
                images,
                actions=None,
                states=None,
                iter_num=-1.0,
                conf = None):

        self.actions = actions
        self.iter_num = iter_num
        self.conf = conf
        self.images = images

        self.cdna, self.stp = False, False
        if self.conf['model'] == 'CDNA':
            self.cdna = True
        elif self.conf['model'] == 'STP':
            self.stp = True

        self.k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_masks = conf['num_masks']
        self.context_frames = conf['context_frames']

        print 'constructing occulsion network...'

        self.batch_size, self.img_height, self.img_width, self.color_channels = [int(i) for i in images[0].get_shape()[0:4]]
        self.lstm_func = basic_conv_lstm_cell

        # Generated robot states and images.
        self.gen_states, self.gen_images = [], []
        self.moved_parts = []
        self.moved_masks = []
        self.assembly_masks_list = []
        self.list_of_trafos = []
        self.list_of_comp_factors = []
        self.generation_masks = []
        self.current_state = states[0]
        self.gen_pix_distrib = []

    def build(self):

        if self.k == -1:
            feedself = True
        else:
            # Scheduled sampling:
            # Calculate number of ground-truth frames to pass in.
            num_ground_truth = tf.to_int32(
                tf.round(tf.to_float(self.batch_size) * (self.k / (self.k + tf.exp(self.iter_num / self.k)))))
            feedself = False

        # LSTM state sizes and states.

        if 'lstm_size' in self.conf:
            lstm_size = self.conf['lstm_size']
        else:
            lstm_size = np.int32(np.array([16, 16, 32, 32, 64, 32, 16]))

        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None

        t = -1
        for image, action in zip(self.images[:-1], self.actions[:-1]):
            t +=1
            # Reuse variables after the first timestep.
            reuse = bool(self.gen_images)

            done_warm_start = len(self.gen_images) > self.context_frames - 1
            with slim.arg_scope(
                    [self.lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                     tf_layers.layer_norm, slim.layers.conv2d_transpose],
                    reuse=reuse):

                if feedself and done_warm_start:
                    # Feed in generated image.
                    prev_image = self.gen_images[-1]
                elif done_warm_start:
                    # Scheduled sampling
                    prev_image = scheduled_sample(image, self.gen_images[-1], self.batch_size,
                                                  num_ground_truth)
                else:
                    # Always feed in ground_truth
                    prev_image = image

                print 'building step', t
                state_action = tf.concat(1, [action, self.current_state])

                enc0 = slim.layers.conv2d(    #32x32x32
                    prev_image,
                    32, [5, 5],
                    stride=2,
                    scope='scale1_conv1',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm1'})

                hidden1, lstm_state1 = self.lstm_func(       # 32x32x16
                    enc0, lstm_state1, lstm_size[0], scope='state1')
                hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
                # hidden2, lstm_state2 = lstm_func(
                #     hidden1, lstm_state2, lstm_size[1], scope='state2')
                # hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
                enc1 = slim.layers.conv2d(     # 16x16x16
                    hidden1, hidden1.get_shape()[3], [3, 3], stride=2, scope='conv2')

                hidden3, lstm_state3 = self.lstm_func(   #16x16x32
                    enc1, lstm_state3, lstm_size[2], scope='state3')
                hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')
                # hidden4, lstm_state4 = lstm_func(
                #     hidden3, lstm_state4, lstm_size[3], scope='state4')
                # hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm5')
                enc2 = slim.layers.conv2d(    #8x8x32
                    hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv3')

                # Pass in state and action.
                smear = tf.reshape(
                    state_action,
                    [int(self.batch_size), 1, 1, int(state_action.get_shape()[1])])
                smear = tf.tile(
                    smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
                if self.use_state:
                    enc2 = tf.concat(3, [enc2, smear])
                enc3 = slim.layers.conv2d(   #8x8x32
                    enc2, hidden3.get_shape()[3], [1, 1], stride=1, scope='conv4')

                hidden5, lstm_state5 = self.lstm_func(  #8x8x64
                    enc3, lstm_state5, lstm_size[4], scope='state5')
                hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
                enc4 = slim.layers.conv2d_transpose(  #16x16x64
                    hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

                hidden6, lstm_state6 = self.lstm_func(  #16x16x32
                    enc4, lstm_state6, lstm_size[5], scope='state6')
                hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')

                if not 'noskip' in self.conf:
                    # Skip connection.
                    hidden6 = tf.concat(3, [hidden6, enc1])  # both 16x16

                enc5 = slim.layers.conv2d_transpose(  #32x32x32
                    hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
                hidden7, lstm_state7 = self.lstm_func( # 32x32x16
                    enc5, lstm_state7, lstm_size[6], scope='state7')
                hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

                if not 'noskip' in self.conf:
                    # Skip connection.
                    hidden7 = tf.concat(3, [hidden7, enc0])  # both 32x32

                enc6 = slim.layers.conv2d_transpose(   # 64x64x16
                    hidden7,
                    hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm9'})

                # Using largest hidden state for predicting a new image layer.
                # changed activation to None! so that the sigmoid layer after it can generate
                # the full range of values.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, self.color_channels, 1, stride=1, scope='convt4', activation_fn=None)

                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                generated_pix = tf.nn.sigmoid(enc7)

                # if t ==0:
                #     self.objectmasks = self.decompose_firstimage(enc6)

                stp_input0 = tf.reshape(hidden5, [int(self.batch_size), -1])
                stp_input1 = slim.layers.fully_connected(
                    stp_input0, 100, scope='fc_stp')

                # disabling capability to generete pixels
                reuse_stp = None
                if reuse:
                    reuse_stp = reuse

                if 'pos_dependent_assembly' in self.conf:
                    moved_images = self.stp_transformation(self.images[1], stp_input1, self.num_masks, reuse_stp)
                else:
                    if self.stp:
                        moved_images, moved_masks, transforms = self.stp_transformation_mask(
                                self.images[1], self.objectmasks, stp_input1, self.num_masks, reuse_stp)
                        self.list_of_trafos.append(transforms)

                    if self.cdna:
                        cdna_input = tf.reshape(hidden5, [int(self.batch_size), -1])
                        moved_images, moved_masks, _ = self.cdna_transformation_mask(prev_image, self.objectmasks, cdna_input, self.num_masks,
                                                     reuse_sc=reuse)


                if 'exp_comp' in self.conf or 'quad_comp' in self.conf or 'sign_comp' in self.conf:
                    activation = None
                else:
                    activation = tf.nn.relu

                if 'gen_pix_averagestep' in self.conf:
                    num_comp_fact = self.num_masks + 1
                else: num_comp_fact = self.num_masks
                comp_fact_input = slim.layers.fully_connected(tf.reshape(hidden5, [self.batch_size, -1]),
                                                              num_comp_fact, scope='fc_compfactors',
                                                                    activation_fn= activation)
                if 'exp_comp' in self.conf:
                    comp_fact_input = tf.exp(comp_fact_input)
                elif 'quad_comp' in self.conf:
                    comp_fact_input = tf.square(comp_fact_input)
                elif 'abs_comp' in self.conf:
                    comp_fact_input = tf.abs(comp_fact_input)

                comp_fact_input = tf.split(1, num_comp_fact, comp_fact_input)

                self.list_of_comp_factors.append(comp_fact_input)

                assembly = tf.zeros([self.batch_size, 64, 64, 3], dtype=tf.float32)
                if 'pos_dependent_assembly' in self.conf:
                    masks = slim.layers.conv2d_transpose(
                        enc6, self.num_masks+1, 1, stride=1, scope='convt7_posdep')
                    masks = tf.reshape(
                        tf.nn.softmax(tf.reshape(masks, [-1, self.num_masks+1])),
                        [int(self.batch_size), int(self.img_height), int(self.img_width), self.num_masks+1])
                    assembly_masks = tf.split(3, self.num_masks+1, masks)
                    self.assembly_masks_list.append(assembly_masks)

                    # moved_images += [generated_pix]

                    for part, mask in zip(moved_images, assembly_masks):
                        assembly += part * mask
                else:
                    if 'gen_pix_averagestep' in self.conf:  # insert the genearted pixels when averaging
                        moved_images = [generated_pix] + moved_images
                        moved_masks = [self.get_generationmask2(enc6)] + moved_masks

                    normalizer = tf.zeros([self.batch_size, 64, 64, 1], dtype=tf.float32)
                    for part, moved_mask, cfact in zip(moved_images, moved_masks, comp_fact_input):
                        cfact = tf.reshape(cfact, [self.batch_size, 1, 1, 1])
                        assembly += part*moved_mask*cfact
                        normalizer += moved_mask*cfact
                    assembly /= (normalizer + tf.ones_like(normalizer) * 1e-4)

                # insert the genearted pixels later
                if 'gen_pix' in self.conf:
                    generation_mask = self.get_generationmask(enc6)
                    gen_image = generation_mask[0]*assembly + generation_mask[1]*generated_pix
                    self.generation_masks.append(generation_mask)
                else:
                    gen_image = assembly

                self.moved_parts.append(moved_images)
                if 'pos_dependent_assembly' not in self.conf:
                    self.moved_masks.append(moved_masks)
                self.gen_images.append(gen_image)

                self.current_state = slim.layers.fully_connected(
                    tf.reshape(hidden5, [self.batch_size, -1]),
                    int(self.current_state.get_shape()[1]),
                    scope='state_pred',
                    activation_fn=None)
                self.gen_states.append(self.current_state)

    def decompose_firstimage(self, enc6):
        masks = slim.layers.conv2d_transpose(
            enc6, self.num_masks, 1, stride=1, scope='convt7_objectmask')
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, self.num_masks])),
            [int(self.batch_size), int(self.img_height), int(self.img_width), self.num_masks])
        mask_list = tf.split(3, self.num_masks, masks)

        return mask_list

    def get_generationmask(self, enc6):
        masks = slim.layers.conv2d_transpose(
            enc6, 2, 1, stride=1, scope='convt7_generationmask')
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, 2])),
            [int(self.batch_size), int(self.img_height), int(self.img_width), 2])
        return tf.split(3, 2, masks)

    def get_generationmask2(self, enc6):
        mask = slim.layers.conv2d_transpose(
            enc6, 1, 1, stride=1, scope='convt7_generationmask', activation_fn=None)
        mask = tf.exp(mask)
        return mask


    ## Utility functions
    def stp_transformation(self,prev_image, stp_input, num_masks, reuse=None):
        """Apply spatial transformer predictor (STP) to previous image.

        Args:
          prev_image: previous image to be transformed.
          stp_input: hidden layer to be used for computing STN parameters.
          num_masks: number of masks and hence the number of STP transformations.
        Returns:
          List of images transformed by the predicted STP parameters.
        """
        # Only import spatial transformer if needed.
        from video_prediction.transformer.spatial_transformer import transformer

        identity_params = tf.convert_to_tensor(
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
        transformed = []
        for i in range(num_masks):
            params = slim.layers.fully_connected(
                stp_input, 6, scope='stp_params' + str(i),
                activation_fn=None,
                reuse=reuse) + identity_params
            outsize = (prev_image.get_shape()[1], prev_image.get_shape()[2])
            transformed.append(transformer(prev_image, params, outsize))

        return transformed


    def stp_transformation_mask(self, first_image, prev_mask_list, stp_input, num_masks, reuse= None):
        """Apply spatial transformer predictor (STP) to previous image.
    
        Args:
          first_image: first image which is transformed.
          stp_input: hidden layer to be used for computing STN parameters.
          num_masks: number of masks and hence the number of STP transformations.
        Returns:
          List of images transformed by the predicted STP parameters.
        """
        # Only import spatial transformer if needed.
        from video_prediction.transformer.spatial_transformer import transformer

        identity_params = tf.convert_to_tensor(
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
        identity_params = tf.reshape(identity_params,[1,6])
        identity_params = tf.tile(identity_params,[self.batch_size,1])

        transformed_parts = []
        transformed_masks = []
        transforms = []

        if 'movement_factor' in self.conf:
            movement_factor = self.conf['movement_factor']
        else: movement_factor = 1.

        for i in range(num_masks):
            if 'norotation' in self.conf:
                params = slim.layers.fully_connected(
                    stp_input, 2, scope='stp_params' + str(i),
                    activation_fn=None,
                    reuse=reuse)*movement_factor
                params = tf.reshape(params, [32,2,1])
                init_val = np.stack([np.identity(2) for _ in range(self.batch_size)])
                identity_mat = tf.Variable(init_val, dtype=tf.float32)
                params = tf.reshape(tf.concat(2, [identity_mat, params]),[32, 6])
            else:
                params = slim.layers.fully_connected(
                    stp_input, 6, scope='stp_params' + str(i),
                    activation_fn=None,
                    reuse= reuse)*movement_factor + identity_params

            if i == 0:
                #static background:
                params = identity_params

            transforms.append(params)
            outsize = (first_image.get_shape()[1], first_image.get_shape()[2])

            transformed_part = transformer(first_image, params, outsize)
            transformed_part = tf.reshape(transformed_part, [self.batch_size, 64, 64, 3])
            transformed_parts.append(transformed_part)

            transf_mask = transformer(prev_mask_list[i], params, outsize)
            transf_mask = tf.reshape(transf_mask, [self.batch_size, 64, 64, 1])
            transformed_masks.append(transf_mask)

        return transformed_parts, transformed_masks, transforms

    def cdna_transformation_mask(self, prev_image, init_masks, cdna_input, num_masks, reuse_sc=None):
        """Apply convolutional dynamic neural advection to previous image.

        Args:
          prev_image: previous image to be transformed.
          cdna_input: hidden lyaer to be used for computing CDNA kernels.
          num_masks: the number of masks and hence the number of CDNA transformations.
          color_channels: the number of color channels in the images.
        Returns:
          List of images transformed by the predicted CDNA kernels.
        """

        # Predict kernels using linear function of last hidden layer.
        cdna_kerns = slim.layers.fully_connected(
            cdna_input,
            DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
            scope='cdna_params',
            activation_fn=None,
            reuse=reuse_sc)

        # Reshape and normalize.
        cdna_kerns = tf.reshape(
            cdna_kerns, [self.batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
        cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
        cdna_kerns /= norm_factor
        cdna_kerns_summary = cdna_kerns
        cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, self.color_channels, 1])
        cdna_kerns = tf.split(0, self.batch_size, cdna_kerns)
        prev_images = tf.split(0, self.batch_size, prev_image)
        # Transform image.
        transformed = []
        transformed_masks = []
        for i_b, kernel, preimg in zip(range(self.batch_size), cdna_kerns, prev_images):
            kernel = tf.squeeze(kernel)
            if len(kernel.get_shape()) == 3:
                kernel = tf.expand_dims(kernel, -2)  # correction! ( was -1 before)
            transformed.append(
                tf.nn.depthwise_conv2d(preimg, kernel, [1, 1, 1, 1], 'SAME'))

            sing_ex_trafo = []
            for i_mask in range(self.num_masks):
                sing_ex_mask = tf.slice(init_masks[i_mask],[i_b,0,0,0], [1,-1,-1,-1])
                mask_kernel = tf.slice(kernel,[0,0,0,i_mask],[-1,-1,1,1])
                sing_ex_trafo.append(
                    tf.nn.depthwise_conv2d(sing_ex_mask, mask_kernel, [1, 1, 1, 1], 'SAME'))

            sing_ex_trafo = tf.concat(0, sing_ex_trafo)
            sing_ex_trafo = tf.expand_dims(sing_ex_trafo,0)
            transformed_masks.append(sing_ex_trafo)

        transformed_masks = tf.concat(0, transformed_masks)
        transformed_masks = tf.split(1, self.num_masks, transformed_masks)
        transformed_masks = [tf.reshape(m, [self.batch_size, 64,64,1]) for m in transformed_masks]

        transformed = tf.concat(0, transformed)
        transformed = tf.split(3, num_masks, transformed)

        return transformed, transformed_masks, cdna_kerns_summary


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

