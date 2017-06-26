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
# from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func
import pdb

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12



from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers


def init_state(inputs,
               state_shape,
               state_initializer=tf.zeros_initializer(),
               dtype=tf.float32):
  """Helper function to create an initial state given inputs.

  Args:
    inputs: input Tensor, at least 2D, the first dimension being batch_size
    state_shape: the shape of the state.
    state_initializer: Initializer(shape, dtype) for state Tensor.
    dtype: Optional dtype, needed when inputs is None.
  Returns:
     A tensors representing the initial state.
  """
  if inputs is not None:
    # Handle both the dynamic shape as well as the inferred shape.
    inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
    dtype = inputs.dtype
  else:
    inferred_batch_size = 0
  initial_state = state_initializer(
      [inferred_batch_size] + state_shape, dtype=dtype)
  return initial_state


@add_arg_scope
def basic_conv_lstm_cell(inputs,
                         state,
                         num_channels,
                         filter_size=5,
                         forget_bias=1.0,
                         scope=None,
                         reuse=None):
  """Basic LSTM recurrent network cell, with 2D convolution connctions.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  Args:
    inputs: input Tensor, 4D, batch x height x width x channels.
    state: state Tensor, 4D, batch x height x width x channels.
    num_channels: the number of output channels in the layer.
    filter_size: the shape of the each convolution filter.
    forget_bias: the initial value of the forget biases.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and the variables should be reused.

  Returns:
     a tuple of tensors representing output and the new state.
  """
  spatial_size = inputs.get_shape()[1:3]
  if state is None:
    state = init_state(inputs, list(spatial_size) + [2 * num_channels])
  with tf.variable_scope(scope,
                         'BasicConvLstmCell',
                         [inputs, state],
                         reuse=reuse):
    inputs.get_shape().assert_has_rank(4)
    state.get_shape().assert_has_rank(4)
    c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
    inputs_h = tf.concat(axis=3, values=[inputs, h])
    # Parameters of gates are concatenated into one conv for efficiency.
    i_j_f_o = layers.conv2d(inputs_h,
                            4 * num_channels, [filter_size, filter_size],
                            stride=1,
                            activation_fn=None,
                            scope='Gates')

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o)

    new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
    new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(axis=3, values=[new_c, new_h])


lstm_func = basic_conv_lstm_cell




def create_accum_tf_factorized_generator(images, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                                         num_masks=None, dependent_mask=None, stp=False, cdna=True, dna=False, pix_distributions=None, **kwargs):
    """
    like create_accum_tf_generator except the transformed images are the inputs of the convnet
    and that the masks depend on the transformed images
    """
    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    gen_pix_distrib = []
    gen_transformed_images = []
    gen_transformed_distrib = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    # LSTM state sizes and states.
    lstm_size = np.int32(np.array([16, 16, 32, 32, 64, 32, 16]))
    lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
    lstm_state5, lstm_state6, lstm_state7 = None, None, None

    # actual mask is unused
    for image, action, t in zip(images[:-1], actions[:-1], range(len(images))):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            print t
            if feedself and done_warm_start:
                prev_image = gen_images[-1]
                transformed_images = gen_transformed_images[-1][2:]

                if pix_distributions != None:
                    prev_pix_distrib = gen_pix_distrib[-1]
                    transformed_pix_distrib = gen_transformed_distrib[-1][2:]

            elif done_warm_start:
                # this is not compatible with python 2
                # prev_image, *transformed_images = \
                #     scheduled_samples([image] * num_masks,
                #                       [gen_images[-1], *gen_transformed_images[-1][2:]],
                #                       batch_size, num_ground_truth)
                prev_and_transformed_images = \
                    scheduled_samples([image] * num_masks,
                                      [gen_images[-1]] + list(gen_transformed_images[-1][2:]),
                                      batch_size, num_ground_truth)
                prev_images, transformed_images = prev_and_transformed_images[0], prev_and_transformed_images[1:]
            else:
                prev_image = image
                transformed_images = [image] * (num_masks - 1)

                if pix_distributions != None:
                    prev_pix_distrib = tf.expand_dims(pix_distributions[t], axis=-1)
                    transformed_pix_distrib = [tf.expand_dims(pix_distributions[t], -1)] * (num_masks - 1)

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])

            enc0 = slim.layers.conv2d(    #32x32x32
                tf.concat(axis=3, values=transformed_images),
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
            smear = tf.expand_dims(tf.expand_dims(state_action, 1), 1)
            smear = tf.tile(
                smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
            enc2 = tf.concat(axis=3, values=[enc2, smear])
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

            # Skip connection.
            hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16

            enc5 = slim.layers.conv2d_transpose(  #32x32x32
                hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
            hidden7, lstm_state7 = lstm_func( # 32x32x16
                enc5, lstm_state7, lstm_size[6], scope='state7')
            hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

            # Skip connection.
            hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

            enc6 = slim.layers.conv2d_transpose(   # 64x64x16
                hidden7,
                hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm9'})

            DNA_KERN_SIZE = 5
            if dna:
                raise NotImplementedError
                # Using largest hidden state for predicting untied conv kernels.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, DNA_KERN_SIZE**2, 1, stride=1, scope='convt4')
            else:
                # Using largest hidden state for predicting a new image layer.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, color_channels, 1, stride=1, scope='convt4')
                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                transformed = [tf.nn.sigmoid(enc7)]

            if stp:
                raise NotImplementedError
                stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
                stp_input1 = slim.layers.fully_connected(
                    stp_input0, 100, scope='fc_stp')
                transformed += stp_transformation(prev_image, stp_input1, num_masks)
            elif cdna:
                cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
                assert len(transformed_images) == num_masks - 1
                transformed += cdna_transformations(transformed_images, cdna_input, num_masks - 1,
                                                    int(color_channels), reuse_sc= reuse)
                if pix_distributions != None:
                    transf_distrib = cdna_transformations(transformed_pix_distrib,cdna_input,
                                                               num_masks - 1,1,reuse_sc= True)
            elif dna:
                raise NotImplementedError
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7)]

            if dependent_mask:
                masks = slim.layers.conv2d_transpose(
                    tf.concat(axis=3, values=[enc6] + transformed), num_masks + 1, 1, stride=1, scope='convt7')
            else:
                masks = slim.layers.conv2d_transpose(
                    enc6, num_masks + 1, 1, stride=1, scope='convt7')
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
                [int(batch_size), int(img_height), int(img_width), num_masks + 1])
            mask_list = tf.split(axis=3, num_or_size_splits=num_masks + 1, value=masks)
            transformed.insert(0, prev_image)
            output = 0
            for layer, mask in zip(transformed, mask_list):
                output += layer * mask
            gen_images.append(output)
            gen_masks.append(mask_list)

            transf_distrib.insert(0, prev_pix_distrib)
            transf_distrib.insert(1, 0)  # to replace the generated pixels which don't exist for pixdistrib
            if pix_distributions!=None:
                pix_distrib_output = 0
                for layer, mask in zip(transf_distrib, mask_list):
                    pix_distrib_output += layer * mask

                gen_pix_distrib.append(pix_distrib_output)

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)

            gen_transformed_distrib.append(transf_distrib)
            gen_transformed_images.append(transformed)

    return gen_images, gen_masks, gen_states, gen_transformed_images, gen_pix_distrib


def construct_model(images,
                    actions=None,
                    states=None,
                    iter_num=-1.0,
                    k=-1,
                    num_masks=10,
                    context_frames=2,
                    pix_distributions=None,
                    conf=None):
    if 'dna_size' in conf.keys():
        DNA_KERN_SIZE = conf['dna_size']
    else:
        DNA_KERN_SIZE = 5
    print 'constructing sawyer network'
    assert DNA_KERN_SIZE == 5
    with tf.variable_scope("generator") as scope:
        gen_images, gen_masks, gen_states, gen_transformed_images, gen_pix_distrib = \
            create_accum_tf_factorized_generator(images, states, actions,
                                                 iter_num=iter_num,
                                                 schedule_sampling_k=k,
                                                 num_masks=num_masks,
                                                 context_frames=context_frames,
                                                 dependent_mask=True,
                                                 stp=conf['model'] == 'STP',
                                                 cdna=conf['model'] == 'CDNA',
                                                 dna=conf['model'] == 'DNA',
                                                 pix_distributions= pix_distributions)
    return gen_images, gen_states, gen_masks, gen_pix_distrib


def cdna_transformations(prev_images, cdna_input, num_masks, color_channels, reuse_sc=None):
    """Apply convolutional dynamic neural advection to previous image.

    Args:
        prev_image: previous image to be transformed.
        cdna_input: hidden lyaer to be used for computing CDNA kernels.
        num_masks: the number of masks and hence the number of CDNA transformations.
        color_channels: the number of color channels in the images.
    Returns:
        List of images transformed by the predicted CDNA kernels.
    """
    DNA_KERN_SIZE = 5
    RELU_SHIFT = 1e-12

    batch_size = int(cdna_input.get_shape()[0])
    height = int(prev_images[0].get_shape()[1])
    width  = int(prev_images[0].get_shape()[2])

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
    cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
    cdna_kerns = tf.reshape(cdna_kerns, [DNA_KERN_SIZE, DNA_KERN_SIZE, -1, 1])
    prev_images = tf.stack(prev_images, axis=0)
    prev_images = tf.transpose(prev_images, [4, 2, 3, 1, 0])
    prev_images = tf.reshape(prev_images, [color_channels, height, width, -1])

    transformed_images = tf.nn.depthwise_conv2d(prev_images, cdna_kerns, [1, 1, 1, 1], 'SAME')

    # Transpose and reshape.
    transformed_images = tf.reshape(transformed_images, [color_channels, height, width, batch_size, num_masks])
    transformed_images = tf.transpose(transformed_images, [3, 1, 2, 0, 4])
    transformed_images = tf.unstack(transformed_images, axis=-1)
    return transformed_images


def scheduled_samples(ground_truth_xs, generated_xs, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data points.

    Args:
        ground_truth_x: tensor of ground-truth data points.
        generated_x: tensor of generated data points.
        batch_size: batch size
        num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
        New batch with num_ground_truth sampled from ground_truth_x and the rest
        from generated_x.
    """
    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    outs = []
    for ground_truth_x, generated_x in zip(ground_truth_xs, generated_xs):
        ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
        generated_examps = tf.gather(generated_x, generated_idx)
        out = tf.dynamic_stitch([ground_truth_idx, generated_idx],
                                [ground_truth_examps, generated_examps])
        out = tf.reshape(out, [batch_size] + out.get_shape().as_list()[1:])  # "reshape" it to resolve the dynamic shape
        outs.append(out)
    # special case to simplify the computation graph when num_ground_truth == 0
    return tf.cond(tf.equal(num_ground_truth, 0), lambda: generated_xs, lambda: outs)
