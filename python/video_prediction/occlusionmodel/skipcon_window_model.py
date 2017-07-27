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
from video_prediction.lstm_ops12 import basic_conv_lstm_cell

import pdb


# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


class Skipcon_Window(object):
    def __init__(self,
                images,
                actions=None,
                states=None,
                iter_num=-1.0,
                conf = None,
                pix_distibution=None,
                 ):

        self.actions = actions
        self.iter_num = iter_num
        self.conf = conf
        self.ncontext = conf['context_frames']
        self.images = images
        self.batch_size, self.img_height, self.img_width, self.color_channels = [int(i) for i in
                                                                                 images[0].get_shape()[0:4]]

        if pix_distibution != None:
            self.pix_distribution = [tf.reshape(p, shape=[self.batch_size, 64,64,1]) for p in pix_distibution]
        else:
            self.pix_distribution = None

        self.moved_pix_distrib = []


        self.cdna, self.stp, self.dna = False, False, False
        if self.conf['model'] == 'CDNA':
            self.cdna = True
        elif self.conf['model'] == 'STP':
            self.stp = True
        elif self.conf['model'] == 'DNA':
            self.dna = True
        if self.stp + self.cdna + self.dna != 1:
            raise ValueError("More than one option selected!")


        if 'dna_size' in conf:
            self.DNA_KERN_SIZE = conf['dna_size']
        else:
            self.DNA_KERN_SIZE = 5


        self.k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_objmasks = conf['num_masks']
        self.context_frames = conf['context_frames']

        print 'constructing occulsion network...'


        self.lstm_func = basic_conv_lstm_cell

        self.padding_map = []
        self.background = []
        self.background_mask = []

        # Generated robot states and images.
        self.gen_states, self.gen_images = [], []
        self.moved_imagesl = []
        self.accum_masks_l = []
        self.accum_Images_l =[]
        self.accum_pix_distrib_l = []
        self.comp_masks_l = []
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

        for t in range(len(self.actions)-1):
            # Reuse variables after the first timestep.
            reuse = bool(self.gen_images)

            done_warm_start = len(self.gen_images) > self.context_frames - 1   # context_frames = 10
            with slim.arg_scope(
                    [self.lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                     tf_layers.layer_norm, slim.layers.conv2d_transpose],
                    reuse=reuse):

                if feedself and done_warm_start:
                    # Feed in generated image.
                    prev_image = self.gen_images[-1]
                    next_image = tf.zeros((self.batch_size, 64, 64, 3))
                elif done_warm_start:
                    # Scheduled sampling
                    prev_image = scheduled_sample(self.images[t], self.gen_images[-1], self.batch_size,
                                                  num_ground_truth)
                    next_image = tf.zeros((self.batch_size, 64, 64, 3))
                else:
                    # Always feed in ground_truth
                    prev_image = self.images[t]
                    next_image = self.images[t+1]

                print 'building step', t

                state_action = tf.concat([self.actions[t], self.current_state], axis=1)

                if 'refeed_accum' in self.conf:
                    print 'refeeding accum images'
                    if t > 0:
                        conv1_input = tf.concat([prev_image] + self.accum_Images_l[-1], 3)
                    else:
                        conv1_input  = tf.concat([prev_image, tf.zeros([self.batch_size,64,64,self.conf['num_masks']*3], tf.float32)],3 )
                else:
                    conv1_input = prev_image

                if 'use_next_img' in self.conf:
                    print 'using next img'
                    conv1_input = tf.concat([conv1_input, next_image], 3)

                enc0 = slim.layers.conv2d(    #32x32x32
                    conv1_input,
                    32, [5, 5],
                    stride=2,
                    scope='scale1_conv1',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm1'},
                    )

                hidden1, lstm_state1 = self.lstm_func(       # 32x32x16
                    enc0, lstm_state1, lstm_size[0], scope='state1')
                hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')

                enc1 = slim.layers.conv2d(     # 16x16x16
                    hidden1, hidden1.get_shape()[3], [3, 3], stride=2, scope='conv2')

                hidden3, lstm_state3 = self.lstm_func(   #16x16x32
                    enc1, lstm_state3, lstm_size[2], scope='state3')
                hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')

                enc2 = slim.layers.conv2d(    #8x8x32
                    hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv3')

                # Pass in state and action.
                smear = tf.reshape(
                    state_action,
                    [int(self.batch_size), 1, 1, int(state_action.get_shape()[1])])
                smear = tf.tile(
                    smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
                if self.use_state:
                    enc2 = tf.concat([enc2, smear], 3)
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
                    hidden6 = tf.concat([hidden6, enc1], 3)  # both 16x16

                enc5 = slim.layers.conv2d_transpose(  #32x32x32
                    hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
                hidden7, lstm_state7 = self.lstm_func( # 32x32x16
                    enc5, lstm_state7, lstm_size[6], scope='state7')
                hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

                if not 'noskip' in self.conf:
                    # Skip connection.
                    hidden7 = tf.concat([hidden7, enc0], 3)  # both 32x32

                enc6 = slim.layers.conv2d_transpose(   # 64x64x16
                    hidden7,
                    hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm9'})

                if self.cdna:
                    cdna_input = tf.reshape(hidden5, [int(self.batch_size), -1])

                    if 'no_maintainence' in self.conf:
                        moved_images = self.cdna_transformation(prev_image,
                                                                  cdna_input,
                                                                  reuse_sc=reuse)
                        if self.pix_distribution != None:
                            if t == 0:
                                pix_for_trafo = self.pix_distribution[0]
                            else:
                                pix_for_trafo = self.gen_pix_distrib[-1]
                            moved_pix = self.cdna_transformation(pix_for_trafo,
                                                                 cdna_input,
                                                                 reuse_sc = True)
                            self.moved_pix_distrib.append(moved_pix)
                    else:
                        if t == 0:
                            img_for_trafo = [self.images[0] for _ in range(self.conf['num_masks'])]
                            if self.pix_distribution != None:
                                pix_for_trafo = [tf.reshape(self.pix_distribution[0], [self.batch_size, 64,64,1]) for _ in range(self.conf['num_masks'])]
                        else:
                            img_for_trafo = self.accum_Images_l[-1]
                            if self.pix_distribution != None:
                                pix_for_trafo = self.accum_pix_distrib_l[-1]

                        moved_images = self.cdna_transformation_imagewise(img_for_trafo,
                                                                          cdna_input,
                                                                          reuse_sc=reuse)

                        if self.pix_distribution != None:
                            if t == 0:
                                self.moved_pix_distrib.append([
                                self.pix_distribution[0] for _ in range(self.num_objmasks)])

                            moved_pix = self.cdna_transformation_imagewise(pix_for_trafo,
                                                                                 cdna_input,
                                                                                 reuse_sc = True)
                            self.moved_pix_distrib.append(moved_pix)

                if self.dna:
                    moved_images, moved_masks = self.apply_dna_separately(enc6)
                    self.moved_masksl.append(moved_masks)

                self.moved_imagesl.append(moved_images)

                if 'no_maintainence' in self.conf:
                    total_num_masks = self.num_objmasks + self.ncontext
                else:
                    total_num_masks = self.num_objmasks

                comp_masks = self.get_masks(enc6, total_num_masks, 'convt7_posdep')

                if self.pix_distribution != None:
                    pix_assembly = tf.zeros([self.batch_size, 64, 64, 1], dtype=tf.float32)
                    for pix, mask in zip(self.moved_pix_distrib[-1], comp_masks):
                        pix_assembly += pix * mask
                    self.gen_pix_distrib.append(pix_assembly)

                assembly = tf.zeros([self.batch_size, 64, 64, 3], dtype=tf.float32)

                if 'no_maintainence' in self.conf:

                    if t < self.ncontext:
                        context_img = self.images[:t + 1]
                    else:
                        context_img = self.images[:self.ncontext]
                        context_img += self.gen_images[self.ncontext-1:t]

                    for i in range(self.conf['use_len'] - len(context_img)):
                        context_img.insert(0, self.images[0])

                    for mimage, mask in zip(context_img, comp_masks[:self.ncontext]):
                        assembly += mimage * mask

                    for mimage, mask in zip(self.moved_imagesl[-1], comp_masks[self.ncontext:]):
                        assembly += mimage * mask
                else:
                    for mimage, mask in zip(self.moved_imagesl[-1], comp_masks):
                        assembly += mimage * mask

                self.comp_masks_l.append(comp_masks)
                self.gen_images.append(assembly)

                if 'no_maintainence' not in self.conf:
                    if t < self.conf['context_frames']:
                        accum_masks = self.get_masks(enc6, total_num_masks, 'convt7_accum')
                        self.accum_masks_l.append(accum_masks)

                        accum_Images  = []
                        for i in range(total_num_masks):
                            accum_Images.append(accum_masks[i] * moved_images[i] + (1-accum_masks[i]) * self.images[t+1])
                        print 'making correction with image at t+1=',t+1

                        self.accum_Images_l.append(accum_Images)

                        if self.pix_distribution != None:
                            accum_pix = []
                            for i in range(total_num_masks):
                                accum_pix.append(
                                    accum_masks[i] * moved_pix[i] + (1 - accum_masks[i]) * self.pix_distribution[t + 1])
                            self.accum_pix_distrib_l.append(accum_pix)
                    else:
                        self.accum_Images_l.append(moved_images)
                        if self.pix_distribution != None:
                            self.accum_pix_distrib_l.append(moved_pix)

                self.current_state = slim.layers.fully_connected(
                    tf.reshape(hidden5, [self.batch_size, -1]),
                    int(self.current_state.get_shape()[1]),
                    scope='state_pred',
                    activation_fn=None)
                self.gen_states.append(self.current_state)

    def get_masks(self, enc6, total_num_masks, scope):
        masks = slim.layers.conv2d_transpose(
            enc6, total_num_masks, 1, stride=1, scope=scope)
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, total_num_masks])),
            [int(self.batch_size), int(self.img_height), int(self.img_width), total_num_masks])
        masks = tf.split(masks, total_num_masks, axis=3)
        return masks

    def apply_dna_separately(self, enc6):
        moved_images = []
        moved_masks = []

        for imask in range(self.num_objmasks):
            scope = 'convt_dnainput{}'.format(imask)
            # Using largest hidden state for predicting untied conv kernels.
            enc7 = slim.layers.conv2d_transpose(
                enc6, self.DNA_KERN_SIZE ** 2, 1, stride=1, scope=scope)

            prev_moved_image = self.moved_imagesl[-1][imask]
            moved_images.append(self.dna_transformation(prev_moved_image,enc7))

            prev_moved_mask = self.moved_masksl[-1][imask]
            moved_masks.append(self.dna_transformation(prev_moved_mask,enc7))

        return moved_images, moved_masks


    def dna_transformation(self, prev_image, dna_input):
        """Apply dynamic neural advection to previous image.

        Args:
          prev_image: previous image to be transformed.
          dna_input: hidden lyaer to be used for computing DNA transformation.
        Returns:
          List of images transformed by the predicted CDNA kernels.
        """
        # Construct translated images.

        pad_len = int(np.floor(self.DNA_KERN_SIZE / 2))
        prev_image_pad = tf.pad(prev_image, [[0, 0], [pad_len, pad_len], [pad_len, pad_len], [0, 0]])
        image_height = int(prev_image.get_shape()[1])
        image_width = int(prev_image.get_shape()[2])

        inputs = []
        for xkern in range(self.DNA_KERN_SIZE):
            for ykern in range(self.DNA_KERN_SIZE):
                inputs.append(
                    tf.expand_dims(
                        tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                                 [-1, image_height, image_width, -1]), [3]))
        inputs = tf.concat(inputs, 3)

        # Normalize channels to 1.
        kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
        kernel = tf.expand_dims(
            kernel / tf.reduce_sum(
                kernel, [3], keep_dims=True), [4])

        return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)

    def cdna_transformation_imagewise(self, prev_image_list, cdna_input, reuse_sc=None):
        """Apply convolutional dynamic neural advection to previous image.

        Args:
          prev_image_list: list of previous images to be transformed. len=num_masks with each element batchsize, 64, 64 3
          cdna_input: hidden lyaer to be used for computing CDNA kernels.
          num_masks: the number of masks and hence the number of CDNA transformations.
          color_channels: the number of color channels in the images.
        Returns:
          List of images transformed by the predicted CDNA kernels.
        """
        num_masks = self.conf['num_masks']
        DNA_KERN_SIZE = self.conf['kern_size']
        batch_size = int(cdna_input.get_shape()[0])
        height = int(prev_image_list[0].get_shape()[1])
        width = int(prev_image_list[0].get_shape()[2])

        # Predict kernels using linear function of last hidden layer.
        cdna_kerns = slim.layers.fully_connected(
            cdna_input,
            DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
            scope='cdna_params',
            activation_fn=None,
            reuse=reuse_sc)

        # Reshape and normalize.
        cdna_kerns = tf.reshape(
            cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
        cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
        cdna_kerns /= norm_factor

        # Transpose and reshape.
        cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])   #

        cdna_kerns = tf.reshape(cdna_kerns, [DNA_KERN_SIZE, DNA_KERN_SIZE, batch_size, num_masks])

        transformed_list = []
        for i in range(num_masks):
            cdna_kern = tf.reshape(cdna_kerns[:, :, :, i],[DNA_KERN_SIZE, DNA_KERN_SIZE, batch_size, 1])

            prev_image = tf.transpose(prev_image_list[i], [3, 1, 2, 0])  # 3,64,64,batchsize
            transformed = tf.nn.depthwise_conv2d(prev_image, cdna_kern, [1, 1, 1, 1], 'SAME')
            transformed = tf.transpose(transformed, [3, 1, 2, 0])

            transformed_list.append(transformed)

        return transformed_list

    def cdna_transformation(self, prev_image, cdna_input, reuse_sc=None):
        """Apply convolutional dynamic neural advection to previous image.

        Args:
          prev_image: previous image to be transformed.
          cdna_input: hidden lyaer to be used for computing CDNA kernels.
          num_masks: the number of masks and hence the number of CDNA transformations.
          color_channels: the number of color channels in the images.
        Returns:
          List of images transformed by the predicted CDNA kernels.
        """
        DNA_KERN_SIZE = self.conf['kern_size']
        num_masks = self.conf['num_masks']
        color_channels = int(prev_image.get_shape()[3])

        batch_size = int(cdna_input.get_shape()[0])
        height = int(prev_image.get_shape()[1])
        width = int(prev_image.get_shape()[2])

        # Predict kernels using linear function of last hidden layer.
        cdna_kerns = slim.layers.fully_connected(
            cdna_input,
            DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
            scope='cdna_params',
            activation_fn=None,
            reuse=reuse_sc)

        # Reshape and normalize.
        cdna_kerns = tf.reshape(
            cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
        cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
        cdna_kerns /= norm_factor
        cdna_kerns_summary = cdna_kerns

        # Transpose and reshape.
        cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
        cdna_kerns = tf.reshape(cdna_kerns, [DNA_KERN_SIZE, DNA_KERN_SIZE, batch_size, num_masks])
        prev_image = tf.transpose(prev_image, [3, 1, 2, 0])

        transformed = tf.nn.depthwise_conv2d(prev_image, cdna_kerns, [1, 1, 1, 1], 'SAME')

        # Transpose and reshape.
        transformed = tf.reshape(transformed, [color_channels, height, width, batch_size, num_masks])
        transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
        transformed = tf.unstack(value=transformed, axis=-1)

        return transformed


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

