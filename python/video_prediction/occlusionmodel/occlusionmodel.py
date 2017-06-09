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

        self.k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_masks = conf['num_masks']
        self.context_frames = conf['context_frames']

        print 'constructing occulsion network...'

        self.batch_size, self.img_height, self.img_width, self.color_channels = [int(i) for i in images[0].get_shape()[0:4]]
        self.lstm_func = basic_conv_lstm_cell

        # Generated robot states and images.
        self.gen_states, self.gen_images, self.gen_masks = [], [], []
        self.background_masks = []
        self.generation_masks = []
        self.list_of_trafos = []
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
                enc7 = slim.layers.conv2d_transpose(
                    enc6, self.color_channels, 1, stride=1, scope='convt4')

                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                generated_pix = tf.nn.sigmoid(enc7)

                if t ==0:
                    self.image_parts, self.objectmasks = self.decompose_firstimage(enc6)

                stp_input0 = tf.reshape(hidden5, [int(self.batch_size), -1])
                stp_input1 = slim.layers.fully_connected(
                    stp_input0, 100, scope='fc_stp')

                # disabling capability to generete pixels
                reuse_stp = None
                if reuse:
                    reuse_stp = reuse

                moved_parts, moved_masks, tansforms = self.stp_transformation_mask(self.image_parts, self.objectmasks, stp_input1, self.num_masks, reuse_stp)
                self.list_of_trafos.append(tansforms)

                comp_fact_input = slim.layers.fully_connected(tf.reshape(hidden5,[self.batch_size, -1]),
                                                    self.num_masks, scope='fc_compfactors')
                comp_factors = tf.split(1, self.num_masks, tf.nn.softmax(comp_fact_input))

                pre_assembly = tf.zeros([self.batch_size, 64, 64, 3], dtype=tf.float32)
                for part, factor in zip(moved_parts, comp_factors):
                    factor = tf.reshape(factor, [self.batch_size, 1, 1, 1])
                    pre_assembly += tf.mul(part, factor)

                composed_mask = tf.zeros([self.batch_size, 64, 64, 3], dtype=tf.float32)
                for mask, factor in zip(moved_masks, comp_factors):
                    factor = tf.reshape(factor, [self.batch_size, 1, 1, 1])
                    composed_mask += tf.mul(mask, factor)

                backgd_mask = 1. - composed_mask
                assembled_image = tf.mul(backgd_mask,self.images[0]) + tf.mul(composed_mask, pre_assembly)

                self.background_masks.append(backgd_mask)

                generation_mask = self.get_generationmask(enc6)
                gen_image = generation_mask[0]*assembled_image + generation_mask[1]*generated_pix
                self.generation_masks.append(generation_mask)
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

        image_partlist = []
        for mask in mask_list:
            image_partlist.append(tf.mul(self.images[0], mask))

        return image_partlist, mask_list

    def get_generationmask(self, enc6):
        masks = slim.layers.conv2d_transpose(
            enc6, 2, 1, stride=1, scope='convt7_generationmask')
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, 2])),
            [int(self.batch_size), int(self.img_height), int(self.img_width), 2])
        return tf.split(3, 2, masks)


    ## Utility functions
    def stp_transformation_mask(self, prev_image_list, prev_mask_list, stp_input, num_masks, reuse= None):
        """Apply spatial transformer predictor (STP) to previous image.
    
        Args:
          prev_image_list: previous image to be transformed.
          stp_input: hidden layer to be used for computing STN parameters.
          num_masks: number of masks and hence the number of STP transformations.
        Returns:
          List of images transformed by the predicted STP parameters.
        """
        # Only import spatial transformer if needed.
        from video_prediction.transformer.spatial_transformer import transformer

        identity_params = tf.convert_to_tensor(
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
        transformed_parts = []
        transformed_masks = []
        transforms = []
        prev_mask_list = [tf.tile(m, [1, 1, 1, 3]) for m in prev_mask_list]  # copy the color channel

        for i in range(num_masks):
            params = slim.layers.fully_connected(
                stp_input, 6, scope='stp_params' + str(i),
                activation_fn=None,
                reuse= reuse) + identity_params
            transforms.append(params)

            outsize = (prev_image_list[0].get_shape()[1], prev_image_list[0].get_shape()[2])

            transformed_part = tf.reshape(transformer(prev_image_list[i], params, outsize), [self.batch_size, 64, 64, 3])
            transformed_parts.append(transformed_part)

            transf_mask = tf.slice(transformer(prev_mask_list[i], params, outsize), [0, 0, 0, 0], [-1, -1, -1, 1])
            transf_mask = tf.reshape(transf_mask, [self.batch_size, 64, 64, 1])
            transformed_masks.append(transf_mask)

        return transformed_parts, transformed_masks, transforms


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

##############
## Costmask code:

def make_initial_pixdistrib(conf, init_object_pos):
    desig_pix = mujoco_to_imagespace_tf(init_object_pos)

    flat_ind = []
    for b in range(conf['batch_size']):
        r = tf.slice(desig_pix, [b, 0], [1, 1])
        c = tf.slice(desig_pix, [b, 1], [1, 1])

        flat_ind.append(r * 64+ c)

    flat_ind = tf.concat(0, flat_ind)
    one_hot = tf.one_hot(flat_ind, depth=64 ** 2, axis=-1)
    one_hot = tf.reshape(one_hot, [conf['batch_size'], 64, 64])

    return [one_hot, one_hot]


def get_new_retinapos(conf, prev_pix_distrib, init_obj_pos, t, iter_num):

    if 'moving_retina' in conf:
        print 'using moving retina'
        if t < 1:
            ret_pix = mujoco_to_imagespace_tf(init_obj_pos)
        else:
            ret_pix = get_max_coord(conf, prev_pix_distrib)
    else:
        ret_pix = mujoco_to_imagespace_tf(init_obj_pos)

    half_rh = conf['retina_size'] / 2
    orig_imh = 64
    current_rpos = tf.clip_by_value(tf.cast(ret_pix, dtype=tf.int32), half_rh, orig_imh - half_rh - 1)

    return current_rpos


def get_max_coord(conf, pix_distrib):
    """
    get new retina centerpos by selecting the pixel with the maximum probability in pix_distrib
    :param conf: 
    :param pix_distrib: 
    :param current_rpos: 
    :return: 
    """
    pix_distrib_shape = pix_distrib.get_shape()[1:]
    maxcoord = tf.arg_max(tf.reshape(pix_distrib, [conf['batch_size'], -1]), dimension=1)
    maxcoord = unravel_argmax(maxcoord, pix_distrib_shape)

    return maxcoord

def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax / (shape[0]))
    output_list.append(argmax % shape[1])
    return tf.cast(tf.pack(output_list, 1), dtype=tf.int32)

def mujoco_to_imagespace_tf(mujoco_coord, numpix = 64):
    """
    convert form Mujoco-Coord to numpix x numpix image space:
    :param numpix: number of pixels of square image
    :param mujoco_coord: batch_size x 2
    :return: pixel_coord: batch_size x 2
    """
    mujoco_coord = tf.cast(mujoco_coord, tf.float32)

    viewer_distance = .75  # distance from camera to the viewing plane
    window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
    pixelheight = window_height / numpix  # height of one pixel
    middle_pixel = numpix / 2
    r = -tf.slice(mujoco_coord,[0,1], [-1,1])
    c =  tf.slice(mujoco_coord,[0,0], [-1,1])
    pixel_coord = tf.concat(1, [r,c])/pixelheight
    pixel_coord += middle_pixel
    pixel_coord = tf.round(pixel_coord)
    pixel_coord = tf.cast(pixel_coord, tf.int32)

    return pixel_coord