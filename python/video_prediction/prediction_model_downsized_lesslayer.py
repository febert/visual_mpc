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

import pdb

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12





def construct_model(images,
                    actions=None,
                    states=None,
                    iter_num=-1.0,
                    k=-1,
                    use_state=True,
                    num_masks=10,
                    stp=False,
                    cdna=True,
                    dna=False,
                    context_frames=2,
                    pix_distributions=None,
                    conf = None):
    """Build convolutional lstm video predictor using STP, CDNA, or DNA.

    Args:
      images: tensor of ground truth image sequences
      actions: tensor of action sequences
      states: tensor of ground truth state sequences
      iter_num: tensor of the current training iteration (for sched. sampling)
      k: constant used for scheduled sampling. -1 to feed in own prediction.
      use_state: True to include state and action in prediction
      num_masks: the number of different pixel motion predictions (and
                 the number of masks for each of those predictions)
      stp: True to use Spatial Transformer Predictor (STP)
      cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
      dna: True to use Dynamic Neural Advection (DNA)
      context_frames: number of ground truth frames to pass in before
                      feeding in own predictions
      pix_distrib: the initial one-hot distriubtion for designated pixels
    Returns:
      gen_images: predicted future image frames
      gen_states: predicted future states

    Raises:
      ValueError: if more than one network option specified or more than 1 mask
      specified for DNA model.
    """

    if 'dna_size' in conf.keys():
        DNA_KERN_SIZE = conf['dna_size']
    else:
        DNA_KERN_SIZE = 5

    print 'constructing network with less layers...'

    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')
    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    lstm_func = basic_conv_lstm_cell

    # Generated robot states and images.
    gen_states, gen_images, gen_masks = [], [], []
    current_state = states[0]
    gen_pix_distrib = []


    summaries = []

    retina_pos_list = []

    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(
            tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
        feedself = False

    # LSTM state sizes and states.

    if 'lstm_size' in conf:
        lstm_size = conf['lstm_size']
    else:
        lstm_size = np.int32(np.array([16, 16, 32, 32, 64, 32, 16]))

    lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
    lstm_state5, lstm_state6, lstm_state7 = None, None, None

    t = -1
    for image, action in zip(images[:-1], actions[:-1]):
        t +=1
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                # Feed in generated image.
                prev_image = gen_images[-1]
                if pix_distributions != None:
                    prev_pix_distrib = gen_pix_distrib[-1]
            elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(image, gen_images[-1], batch_size,
                                              num_ground_truth)
                if pix_distributions != None:
                    prev_pix_distrib = gen_pix_distrib[-1]
            else:
                # Always feed in ground_truth
                prev_image = image
                if pix_distributions != None:
                    prev_pix_distrib = pix_distributions[t]
                    prev_pix_distrib = tf.expand_dims(prev_pix_distrib, -1)

            if 'transform_from_firstimage' in conf:
                assert stp
                if t > 1:
                    prev_image = images[1]
                    print 'using image 1'

            state_action = tf.concat(1, [action, current_state])

            enc0 = slim.layers.conv2d(    #32x32x32
                prev_image,
                32, [5, 5],
                stride=2,
                scope='scale1_conv1',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm1'})

            hidden1, lstm_state1 = lstm_func(       # 32x32x16
                enc0, lstm_state1, lstm_size[0], scope='state1')
            hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
            # hidden2, lstm_state2 = lstm_func(
            #     hidden1, lstm_state2, lstm_size[1], scope='state2')
            # hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
            enc1 = slim.layers.conv2d(     # 16x16x16
                hidden1, hidden1.get_shape()[3], [3, 3], stride=2, scope='conv2')

            hidden3, lstm_state3 = lstm_func(   #16x16x32
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
                [int(batch_size), 1, 1, int(state_action.get_shape()[1])])
            smear = tf.tile(
                smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
            if use_state:
                enc2 = tf.concat(3, [enc2, smear])
            enc3 = slim.layers.conv2d(   #8x8x32
                enc2, hidden3.get_shape()[3], [1, 1], stride=1, scope='conv4')

            hidden5, lstm_state5 = lstm_func(  #8x8x64
                enc3, lstm_state5, lstm_size[4], scope='state5')
            hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
            enc4 = slim.layers.conv2d_transpose(  #16x16x64
                hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

            hidden6, lstm_state6 = lstm_func(  #16x16x32
                enc4, lstm_state6, lstm_size[5], scope='state6')
            hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')

            if not 'noskip' in conf:
                # Skip connection.
                hidden6 = tf.concat(3, [hidden6, enc1])  # both 16x16

            enc5 = slim.layers.conv2d_transpose(  #32x32x32
                hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
            hidden7, lstm_state7 = lstm_func( # 32x32x16
                enc5, lstm_state7, lstm_size[6], scope='state7')
            hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

            if not 'noskip' in conf:
                # Skip connection.
                hidden7 = tf.concat(3, [hidden7, enc0])  # both 32x32

            enc6 = slim.layers.conv2d_transpose(   # 64x64x16
                hidden7,
                hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm9'})

            if dna:
                # Using largest hidden state for predicting untied conv kernels.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, DNA_KERN_SIZE ** 2, 1, stride=1, scope='convt4')
            else:
                # Using largest hidden state for predicting a new image layer.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, color_channels, 1, stride=1, scope='convt4')
                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                transformed = [tf.nn.sigmoid(enc7)]

            if stp:
                stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
                stp_input1 = slim.layers.fully_connected(
                    stp_input0, 100, scope='fc_stp')

                # disabling capability to generete pixels
                reuse_stp = None
                if reuse:
                    reuse_stp = reuse
                transformed = stp_transformation(prev_image, stp_input1, num_masks, reuse_stp)
                # transformed += stp_transformation(prev_image, stp_input1, num_masks, reuse_stp)

                if pix_distributions != None:
                    transf_distrib = stp_transformation(prev_pix_distrib, stp_input1, num_masks, reuse=True)

            elif cdna:
                cdna_input = tf.reshape(hidden5, [int(batch_size), -1])

                new_transformed, new_cdna_filter = cdna_transformation(prev_image,
                                                                cdna_input,
                                                                num_masks,
                                                                int(color_channels),
                                                                reuse_sc= reuse)
                transformed += new_transformed

                summaries+= make_cdna_kerns_summary(new_cdna_filter, t, 'image')

                if pix_distributions != None:
                    transf_distrib, new_cdna_distrib_filter = cdna_transformation(prev_pix_distrib,
                                                                           cdna_input,
                                                                           num_masks,
                                                                           1,
                                                                           reuse_sc= True)
                    summaries += make_cdna_kerns_summary(new_cdna_distrib_filter, t, 'distrib')


            elif dna:
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7, DNA_KERN_SIZE)]

            masks = slim.layers.conv2d_transpose(
                enc6, num_masks + 1, 1, stride=1, scope='convt7')
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
                [int(batch_size), int(img_height), int(img_width), num_masks + 1])
            mask_list = tf.split(3, num_masks + 1, masks)
            output = mask_list[0] * prev_image
            for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask
            gen_images.append(output)
            gen_masks.append(mask_list)

            if dna and pix_distributions != None:
                transf_distrib = [dna_transformation(prev_pix_distrib, enc7, DNA_KERN_SIZE)]

            if pix_distributions!=None:
                pix_distrib_output = mask_list[0] * prev_pix_distrib
                mult_list = []
                for i in range(num_masks):
                    mult_list.append(transf_distrib[i] * mask_list[i+1])
                    pix_distrib_output += mult_list[i]

                gen_pix_distrib.append(pix_distrib_output)

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)


    if pix_distributions != None:
        return gen_images, gen_states, gen_masks, gen_pix_distrib
    else:
        return gen_images, gen_states, gen_masks, None


## Utility functions
def stp_transformation(prev_image, stp_input, num_masks, reuse= None):
    """Apply spatial transformer predictor (STP) to previous image.

    Args:
      prev_image: previous image to be transformed.
      stp_input: hidden layer to be used for computing STN parameters.
      num_masks: number of masks and hence the number of STP transformations.
    Returns:
      List of images transformed by the predicted STP parameters.
    """
    # Only import spatial transformer if needed.
    from transformer.spatial_transformer import transformer

    identity_params = tf.convert_to_tensor(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
    transformed = []
    for i in range(num_masks):
        params = slim.layers.fully_connected(
            stp_input, 6, scope='stp_params' + str(i),
            activation_fn=None,
            reuse= reuse) + identity_params
        outsize = (prev_image.get_shape()[1], prev_image.get_shape()[2])
        transformed.append(transformer(prev_image, params, outsize))

    return transformed


def cdna_transformation(prev_image, cdna_input, num_masks, color_channels, reuse_sc = None):
    """Apply convolutional dynamic neural advection to previous image.

    Args:
      prev_image: previous image to be transformed.
      cdna_input: hidden lyaer to be used for computing CDNA kernels.
      num_masks: the number of masks and hence the number of CDNA transformations.
      color_channels: the number of color channels in the images.
    Returns:
      List of images transformed by the predicted CDNA kernels.
    """
    batch_size = int(cdna_input.get_shape()[0])

    # Predict kernels using linear function of last hidden layer.
    cdna_kerns = slim.layers.fully_connected(
        cdna_input,
        DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
        scope='cdna_params',
        activation_fn=None,
        reuse = reuse_sc)


    # Reshape and normalize.
    cdna_kerns = tf.reshape(
        cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
    cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
    norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
    cdna_kerns /= norm_factor
    cdna_kerns_summary = cdna_kerns

    cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, color_channels, 1])
    cdna_kerns = tf.split(0, batch_size, cdna_kerns)
    prev_images = tf.split(0, batch_size, prev_image)

    # Transform image.
    transformed = []
    for kernel, preimg in zip(cdna_kerns, prev_images):
        kernel = tf.squeeze(kernel)
        if len(kernel.get_shape()) == 3:
            kernel = tf.expand_dims(kernel, -2)   #correction! ( was -1 before)
        transformed.append(
            tf.nn.depthwise_conv2d(preimg, kernel, [1, 1, 1, 1], 'SAME'))
    transformed = tf.concat(0, transformed)
    transformed = tf.split(3, num_masks, transformed)
    return transformed, cdna_kerns_summary


def dna_transformation(prev_image, dna_input, DNA_KERN_SIZE):
    """Apply dynamic neural advection to previous image.

    Args:
      prev_image: previous image to be transformed.
      dna_input: hidden lyaer to be used for computing DNA transformation.
    Returns:
      List of images transformed by the predicted CDNA kernels.
    """
    # Construct translated images.
    pad_len = int(np.floor(DNA_KERN_SIZE / 2))
    prev_image_pad = tf.pad(prev_image, [[0, 0], [pad_len, pad_len], [pad_len, pad_len], [0, 0]])
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


def make_cdna_kerns_summary(cdna_kerns, t, suffix):

    sum = []
    cdna_kerns = tf.split(4, 10, cdna_kerns)
    for i, kern in enumerate(cdna_kerns):
        kern = tf.squeeze(kern)
        kern = tf.expand_dims(kern,-1)
        sum.append(
            tf.image_summary('step' + str(t) +'_filter'+ str(i)+ suffix, kern)
        )

    return  sum


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