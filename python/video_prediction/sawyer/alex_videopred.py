from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.layers.convolutional import Conv2D, Conv2DTranspose

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, images, masks, states, actions, count, steps_per_epoch")
AfnModel = collections.namedtuple("AfnModel", "loss, recon_losses, psnr_losses, outputs, train")
Model = collections.namedtuple("Model", "gen_images, gen_masks, gen_states, gen_transformed_images, preactivation_feature_maps, feature_maps, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_GAN_loss, gen_L1_loss, gen_L2_loss, gen_mask_loss, gen_state_loss, gen_psnr_loss, gen_grads_and_vars, train")


class OnlineStatistics(object):
    def __init__(self, axis=0):
        self.axis = axis
        self.n = None
        self.s = None
        self.s2 = None
        self.reset()

    def reset(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def add_data(self, data):
        if isinstance(self.axis, (list, tuple)):
            self.n += np.prod([data.shape[axis] for axis in self.axis])
        else:
            self.n += data.shape[self.axis]
        self.s += data.sum(axis=self.axis)
        self.s2 += (data ** 2).sum(axis=self.axis)

    @property
    def mean(self):
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt((self.s2 - (self.s ** 2) / self.n) / self.n)


def draw_circle(image, center, radius, color, thickness=1, line_type=cv2.LINE_AA, shift=8):
    center = tuple(np.round(np.asarray(center) * (2 ** shift)).astype(int))
    radius = int(np.round(radius * (2 ** shift)))
    cv2.circle(image, center, radius, color, thickness=thickness, lineType=line_type, shift=shift)


def state_to_image(state, image_size=64, env=None):
    qpos, qvel = np.split(state, [2])
    if env is not None:
        env.set_state(qpos, qvel)
        image = env._get_obs()['image']
    else:
        distance = .75 - 0.03
        pixels_per_unit = image_size / (2 * np.tan(np.deg2rad(75. / 2)) * distance)
        center = np.array([qpos[0], -qpos[1]]) * pixels_per_unit + image_size / 2.0
        radius = 0.05 * pixels_per_unit
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        draw_circle(image, center, radius, (255,) * 3, thickness=-1)
    return image


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def conv(batch_input, out_channels, stride, kernel_size=(4, 4), padding='SAME'):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", list(kernel_size) + [in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding=padding)
        return conv


def deconv(batch_input, out_channels, stride, kernel_size=(4, 4), padding='SAME'):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", list(kernel_size) + [out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        if padding == 'SAME':
            output_shape = [batch, in_height * stride, in_width * stride, out_channels]
        else:
            output_shape = [batch, in_height * stride + kernel_size[0] - 1, in_width * stride + kernel_size[1] - 1, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, output_shape, [1, stride, stride, 1], padding=padding)
        return conv


def dense(inputs, num_units):
    with tf.variable_scope('dense'):
        input_shape = inputs.get_shape().as_list()
        kernel_shape = [input_shape[1], num_units]
        V_ub = 0.01 / np.sqrt(num_units)
        V = tf.get_variable('V', kernel_shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_ub, V_ub))
        g = tf.get_variable('g', (num_units,), dtype=tf.float32, initializer=tf.ones_initializer())
        bias = tf.get_variable('bias', (num_units,), dtype=tf.float32, initializer=tf.zeros_initializer())
        outputs = tf.matmul(inputs, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        outputs = tf.reshape(scaler, [1, num_units]) * outputs + tf.reshape(bias, [1, num_units])
        return outputs


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def prelu(x, alpha_init=0.2, shared_axes=None):
    """Parametric Rectified Linear Unit.

    It follows:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alpha` is a learned array with the same shape as x.

    Args:
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py
    """
    with tf.variable_scope('prelu'):
        x = tf.identity(x)
        param_shape = list(x.get_shape()[1:])
        if shared_axes is not None:
            for i in shared_axes:
                param_shape[i - 1] = 1
        alpha = tf.get_variable('alpha', param_shape, dtype=tf.float32, initializer=tf.constant_initializer(alpha_init))
        pos = tf.nn.relu(x)
        neg = -alpha * tf.nn.relu(-x)
        return pos + neg


def tprelu(x, alpha_init=0.2, shared_axes=None):
    """Translated Parametric Rectified Linear Unit.

    https://github.com/stormraiser/GAN-weight-norm/blob/master/pytorch/modules/TPReLU.py
    """
    with tf.variable_scope('tprelu'):
        x = tf.identity(x)
        param_shape = list(x.get_shape()[1:])
        if shared_axes is not None:
            for i in shared_axes:
                param_shape[i - 1] = 1
        bias = tf.get_variable('bias', param_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        return prelu(x - bias, alpha_init=alpha_init, shared_axes=shared_axes) + bias


def layernorm(input):
    with tf.variable_scope("layernorm"):
        return tf.contrib.layers.layer_norm(input)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def conv2d(inputs, num_filters, kernel_size=(4, 4), strides=(1, 1), padding='SAME'):
    with tf.variable_scope('conv2d'):
        input_shape = inputs.get_shape().as_list()
        filter_shape = list(kernel_size) + [input_shape[-1], num_filters]
        V_ub = 1.0 / np.sqrt(np.prod(kernel_size) * input_shape[-1])
        V = tf.get_variable('V', filter_shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_ub, V_ub))
        g = tf.get_variable('g', (num_filters,), dtype=tf.float32, initializer=tf.ones_initializer())
        bias = tf.get_variable('bias', (num_filters,), dtype=tf.float32, initializer=tf.zeros_initializer())
        filter = tf.reshape(g, (1, 1, 1, num_filters)) * tf.nn.l2_normalize(V, (0, 1, 2))
        strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
        outputs = tf.nn.conv2d(inputs, filter, [1] + strides + [1], padding=padding)
        outputs = tf.nn.bias_add(outputs, bias)
        return outputs


def deconv2d(inputs, num_filters, kernel_size=(4, 4), strides=(1, 1), padding='SAME'):
    with tf.variable_scope('deconv2d'):
        input_shape = inputs.get_shape().as_list()
        filter_shape = list(kernel_size) + [num_filters, input_shape[-1]]
        V_ub = 1.0 / np.sqrt(np.prod(kernel_size) * input_shape[-1])
        V = tf.get_variable('V', filter_shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_ub, V_ub))
        g = tf.get_variable('g', (num_filters,), dtype=tf.float32, initializer=tf.ones_initializer())
        bias = tf.get_variable('bias', (num_filters,), dtype=tf.float32, initializer=tf.zeros_initializer())
        filter = tf.reshape(g, (1, 1, num_filters, 1)) * tf.nn.l2_normalize(V, (0, 1, 3))
        strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
        if padding == 'SAME':
            output_shape = [input_shape[0], input_shape[1] * strides[0], input_shape[2] * strides[1], num_filters]
        else:
            output_shape = [input_shape[0], input_shape[1] * strides[0] + kernel_size[0] - 1, input_shape[2] * strides[1] + kernel_size[1] - 1, num_filters]
        outputs = tf.nn.conv2d_transpose(inputs, filter, output_shape, [1] + strides + [1], padding=padding)
        outputs = tf.nn.bias_add(outputs, bias)
        return outputs


def affine(inputs):
    with tf.variable_scope('affine'):
        input_shape = inputs.get_shape().as_list()
        scale = tf.get_variable('scale', (input_shape[-1],), dtype=tf.float32, initializer=tf.ones_initializer())
        bias = tf.get_variable('bias', (input_shape[-1],), dtype=tf.float32, initializer=tf.zeros_initializer())
        outputs = inputs * scale + bias
        return outputs


def peak_signal_to_noise_ratio(true, pred):
    """Image quality metric based on maximal signal power vs. power of the noise.

    Args:
        true: the ground truth image.
        pred: the predicted image.
    Returns:
        peak signal to noise ratio (PSNR)
    """
    return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
        true: the ground truth image.
        pred: the predicted image.
    Returns:
        mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


def flow_transformer(im, flow, out_size, name='FlowTransformer', **kwargs):
    """Flow Transformer Layer

     Implements a flow transformer layer as described in [1]_.
     Based on [2]_.

     References
     ----------
     .. [1]  View Synthesis by Appearance Flow
             Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, Alexei A. Efros

     .. [2]  https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py

    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat])
            return grid

    def _sample(im, flow, out_size):
        with tf.variable_scope('_sample'):
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            num_channels = tf.shape(im)[3]

            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.transpose(grid)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, -1, 2]))

            flow = tf.reshape(flow, tf.stack([num_batch, -1, 2]))

            T_g = grid + flow
            x_s = tf.slice(T_g, [0, 0, 0], [-1, -1, 1])
            y_s = tf.slice(T_g, [0, 0, 1], [-1, -1, 1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                im, x_s_flat, y_s_flat, out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _sample(im, flow, out_size)
        return output


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


def basic_conv_lstm_cell(inputs,
                         state,
                         num_channels,
                         filter_size=5,
                         forget_bias=1.0,
                         scope=None):
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

    Returns:
         a tuple of tensors representing output and the new state.
    """
    spatial_size = inputs.get_shape()[1:3]
    if state is None:
        state = init_state(inputs, list(spatial_size) + [2 * num_channels])
    with tf.variable_scope(scope,
                           'BasicConvLstmCell',
                           [inputs, state]):
        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)
        c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
        inputs_h = tf.concat(axis=3, values=[inputs, h])
        # Parameters of gates are concatenated into one conv for efficiency.
        with tf.variable_scope('Gates'):
            i_j_f_o = tf.layers.conv2d(inputs_h,
                                       4 * num_channels, [filter_size, filter_size],
                                       padding='same')

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o)

        new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_h, tf.concat(axis=3, values=[new_c, new_h])


def load_examples(input_dir=None, num_epochs=None, mode=None, train_val_split=None, sequence_length=None, frame_skip=None,
                  scale_size=None, batch_size=None, preprocess_image=None, remove_background=None, **kwargs):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    filenames = gfile.Glob(os.path.join(input_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')
    if mode in ('train', 'val'):
        index = int(np.floor(train_val_split * len(filenames)))
        if mode == 'train':
            filenames = filenames[:index]
        else:
            filenames = filenames[index:]

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=mode == 'train')
    reader = tf.TFRecordReader()
    paths, serialized_example = reader.read(filename_queue)

    image_seq, mask_seq, state_seq, action_seq = [], [], [], []

    # load_indx = range(0, 30, frame_skip + 1)
    # load_indx = load_indx[:sequence_length]
    # print('using frame sequence: ', load_indx)
    #
    # ACTION_DIM = 4
    # STATE_DIM = 3
    #
    # for i in load_indx:
    #     image_aux1_name = str(i) + '/image_aux1/encoded'
    #     action_name = str(i) + '/action'
    #     endeffector_pos_name = str(i) + '/endeffector_pos'
    #
    #     features = {
    #
    #         image_aux1_name: tf.FixedLenFeature([1], tf.string),
    #         action_name: tf.FixedLenFeature([ACTION_DIM], tf.float32),
    #         endeffector_pos_name: tf.FixedLenFeature([STATE_DIM], tf.float32),
    #     }
    #
    #     features = tf.parse_single_example(serialized_example, features=features)
    #
    #     COLOR_CHAN = 3
    #     ORIGINAL_WIDTH = 64
    #     ORIGINAL_HEIGHT = 64
    #     IMG_WIDTH = 64
    #     IMG_HEIGHT = 64
    #
    #     image = tf.decode_raw(features[image_aux1_name], tf.uint8)
    #     image = tf.cast(image, tf.float32) / 255.0
    #     image = image / 1.5
    #     image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
    #     image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
    #     if IMG_HEIGHT != IMG_WIDTH:
    #         raise ValueError('Unequal height and width unsupported')
    #     crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
    #     image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    #     image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
    #     image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
    #     image_seq.append(image)
    #
    #     endeffector_pos = tf.reshape(features[endeffector_pos_name], shape=[1, STATE_DIM])
    #     state_seq.append(endeffector_pos)
    #     action = tf.reshape(features[action_name], shape=[1, ACTION_DIM])
    #     action_seq.append(action)
    #
    # image_seq = tf.concat(image_seq, 0)
    #
    # state_seq = tf.concat(state_seq, 0)
    # action_seq = tf.concat(action_seq, 0)
    # paths_batch, image_batch, state_batch, action_batch = tf.train.batch([paths, image_seq, action_seq, state_seq], batch_size)
    # steps_per_epoch = int(math.ceil(len(filenames) / batch_size))
    #
    # return Examples(
    #     paths=paths_batch,
    #     images=image_batch,
    #     masks=None,
    #     states=state_batch,
    #     actions=action_batch,
    #     count=len(filenames),
    #     steps_per_epoch=steps_per_epoch,
    # )

    if 'pushing_data' in input_dir:
        if 'softmotion' in input_dir:
            ORIGINAL_WIDTH = 64
            ORIGINAL_HEIGHT = 64
            COLOR_CHAN = 3

            STATE_DIM = 3
            ACTION_DIM = 4

            image_suffix_name = '%d/image_aux1/encoded'
            state_suffix_name = '%d/endeffector_pos'
            action_suffix_name = '%d/action'
        else:
            ORIGINAL_WIDTH = 64
            ORIGINAL_HEIGHT = 64
            COLOR_CHAN = 3
            MASK_CHAN = 6

            STATE_DIM = 4
            ACTION_DIM = 2

            image_suffix_name = 'move/%d/image/encoded'
            mask_suffix_name = 'move/%d/mask/encoded'
            state_suffix_name = 'move/%d/state'
            action_suffix_name = 'move/%d/action'
    else:
        ORIGINAL_WIDTH = 640
        ORIGINAL_HEIGHT = 512
        COLOR_CHAN = 3

        STATE_DIM = 5
        ACTION_DIM = 5

        image_suffix_name = 'move/%d/image/encoded'
        state_suffix_name = 'move/%d/endeffector/vec_pitch_yaw'
        action_suffix_name = 'move/%d/commanded_pose/vec_pitch_yaw'

    assert frame_skip >= 0
    for i in range((sequence_length - 1) * (frame_skip + 1) + 1):
        if i % (frame_skip + 1) == 0:
            image_name = image_suffix_name % i
            state_name = state_suffix_name % i
            action_name = action_suffix_name % i
            features = {image_name: tf.FixedLenFeature([1], tf.string),
                        state_name: tf.FixedLenFeature([STATE_DIM], tf.float32),
                        action_name: tf.FixedLenFeature([ACTION_DIM], tf.float32)}
            if 'pushing_data' in input_dir and 'masks' in input_dir:
                mask_name = mask_suffix_name % i
                features.update({mask_name: tf.FixedLenFeature([1], tf.string)})
            features = tf.parse_single_example(serialized_example, features=features)

            if 'pushing_data' in input_dir:
                image = tf.decode_raw(features[image_name], tf.uint8)
            else:
                image_buffer = tf.reshape(features[image_name], shape=[])
                image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.reshape(image, [ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

            height = tf.shape(image)[0]
            width = tf.shape(image)[1]

            crop_size = tf.minimum(height, width)

            if 'pushing_data' in input_dir and 'masks' in input_dir:
                mask = tf.decode_raw(features[mask_name], tf.uint8)
                mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
                mask = tf.reshape(mask, [ORIGINAL_HEIGHT, ORIGINAL_WIDTH, MASK_CHAN])

                mask = tf.image.resize_image_with_crop_or_pad(mask, crop_size, crop_size)
                mask = tf.image.resize_images(mask, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
                # if preprocess_image:
                #     mask = preprocess(mask)
                mask_seq.append(mask)

            image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
            image = tf.image.resize_images(image, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
            if remove_background:
                filtered_image = 0
                for i in range(MASK_CHAN):
                    if i == 0:
                        continue
                    filtered_image += mask[..., i:i+1] * image
                image = tf.clip_by_value(filtered_image, 0.0, 1.0)
            if preprocess_image:
                image = preprocess(image)
            image_seq.append(image)

            state = features[state_name]
            action = features[action_name]
            state_seq.append(state)
            action_seq.append(action)
        else:
            action_name = action_suffix_name % i
            features = {action_name: tf.FixedLenFeature([ACTION_DIM], tf.float32)}
            features = tf.parse_single_example(serialized_example, features=features)

            action = features[action_name]
            action_seq.append(action)

    if frame_skip > 0:
        action_seq = [action_seq[(frame_skip + 1) * i:(frame_skip + 1) * i + frame_skip + 1] for i in range(sequence_length)]
        action_seq[-1].append(tf.zeros(frame_skip * ACTION_DIM, dtype=action_seq[-1][0].dtype))
        action_seq = [tf.concat(action, axis=0) for action in action_seq]

    assert len(image_seq) == sequence_length
    assert len(state_seq) == sequence_length
    assert len(action_seq) == sequence_length
    image_seq = tf.stack(image_seq)
    state_seq = tf.stack(state_seq)
    action_seq = tf.stack(action_seq)
    if mask_seq:
        assert len(mask_seq) == sequence_length
        mask_seq = tf.stack(mask_seq)
        paths_batch, image_batch, mask_batch, state_batch, action_batch = tf.train.batch([paths, image_seq, mask_seq, state_seq, action_seq], batch_size=batch_size)
    else:
        paths_batch, image_batch, state_batch, action_batch = tf.train.batch([paths, image_seq, state_seq, action_seq], batch_size=batch_size)
        mask_batch = None
    steps_per_epoch = int(math.ceil(len(filenames) / batch_size))

    return Examples(
        paths=paths_batch,
        images=image_batch,
        masks=mask_batch,
        states=state_batch,
        actions=action_batch,
        count=len(filenames),
        steps_per_epoch=steps_per_epoch,
    )


def create_afn_generator(images, states, actions, num_masks=None, scale_size=None, iter_num=None, schedule_sampling_k=None, context_frames=None, **kwargs):
    conv_layers = []
    state_action_fc_layers = []
    flow_fc_layers = []
    flow_deconv_layers = []
    mask_fc_layers = []
    mask_deconv_layers = []

    conv_specs = [
        (16, 1),   # conv1: [batch, 256, 256, in_channels] => [batch, 128, 128, 16]
        (32, 2),   # conv2: [batch, 128, 128, 16] => [batch, 64, 64, 32]
        (64, 2),   # conv3: [batch, 64, 64, 32] => [batch, 32, 32, 64]
        (128, 2),  # conv4: [batch, 32, 32, 64] => [batch, 16, 16, 128]
        (256, 2),  # conv5: [batch, 16, 16, 128] => [batch, 8, 8, 256]
        (512, 2),  # conv6: [batch, 8, 8, 256] => [batch, 4, 4, 512]
        (2048, 2)  # conv7: [batch, 4, 4, 512] => [batch, 1, 1, 2048]
    ]
    state_action_fc_specs = [
        128,
        256
    ]
    flow_fc_specs = [
        2048,
        2048
    ]
    mask_fc_specs = [
        1024,
        1024
    ]
    flow_deconv_specs = [
        (512, 2),  # deconv7: [batch, 1, 1, 2048] => [batch, 4, 4, 512]
        (256, 2),  # deconv6: [batch, 4, 4, 512] => [batch, 8, 8, 256]
        (128, 2),  # deconv5: [batch, 8, 8, 256] => [batch, 16, 16, 128]
        (64, 2),   # deconv4: [batch, 16, 16, 128] => [batch, 32, 32, 64]
        (32, 2),   # deconv3: [batch, 32, 32, 64] => [batch, 64, 64, 32]
        (16, 2),   # deconv2: [batch, 64, 64, 32] => [batch, 128, 128, 16]
    ]

    if num_masks > 1:
        flow_deconv_specs.append((2 * (num_masks - 1), 1))  # deconv1: [batch, 128, 128, 16] => [batch, 256, 256, 2 * (num_masks - 1)]

        mask_deconv_specs = [
            (512, 2),  # deconv7: [batch, 1, 1, 2048] => [batch, 4, 4, 512]
            (256, 2),  # deconv6: [batch, 4, 4, 512] => [batch, 8, 8, 256]
            (128, 2),  # deconv5: [batch, 8, 8, 256] => [batch, 16, 16, 128]
            (64, 2),   # deconv4: [batch, 16, 16, 128] => [batch, 32, 32, 64]
            (32, 2),   # deconv3: [batch, 32, 32, 64] => [batch, 64, 64, 32]
            (16, 2),   # deconv2: [batch, 64, 64, 32] => [batch, 128, 128, 16]
            (num_masks, 1)  # deconv1: [batch, 128, 128, 16] => [batch, 256, 256, num_masks]
        ]
    else:
        flow_deconv_specs.append((2, 1))

    for k, (filter_dim, stride) in enumerate(conv_specs):
        with tf.variable_scope('conv%d' % (k + 1)):
            conv_layer = Conv2D(filter_dim, (3, 3), strides=stride, padding='same', activation=tf.nn.relu)
            conv_layers.append(conv_layer)

    for k, unit_dim in enumerate(state_action_fc_specs):
        with tf.variable_scope('state_action_fc%d' % (k + 1)):
            state_action_fc_layer = Dense(unit_dim, activation=tf.nn.relu)
            state_action_fc_layers.append(state_action_fc_layer)

    for k, unit_dim in enumerate(flow_fc_specs):
        with tf.variable_scope('flow_fc%d' % (k + 1)):
            flow_fc_layer = Dense(unit_dim, activation=tf.nn.relu)
            flow_fc_layers.append(flow_fc_layer)

    for k, (filter_dim, stride) in enumerate(flow_deconv_specs):
        with tf.variable_scope('flow_deconv%d' % (k + 1)):
            flow_deconv_layer = Conv2DTranspose(filter_dim, (3, 3), strides=stride, padding='same',
                                                activation=None if k == (len(flow_deconv_specs) - 1) else tf.nn.relu)
            flow_deconv_layers.append(flow_deconv_layer)

    if num_masks > 1:
        for k, unit_dim in enumerate(mask_fc_specs):
            with tf.variable_scope('mask_fc%d' % (k + 1)):
                mask_fc_layer = Dense(unit_dim, activation=tf.nn.relu)
                mask_fc_layers.append(mask_fc_layer)

        for k, (filter_dim, stride) in enumerate(mask_deconv_specs):
            with tf.variable_scope('mask_deconv%d' % (k + 1)):
                mask_deconv_layer = Conv2DTranspose(filter_dim, (3, 3), strides=stride, padding='same',
                                               activation=None if k == (len(mask_deconv_specs) - 1) else tf.nn.relu)
                mask_deconv_layers.append(mask_deconv_layer)

    with tf.variable_scope('state_fc'):
        state_fc_layer = Dense(int(states[0].get_shape()[1]), activation=None)

    # TODO: LSTM
    # TODO: scheduled sampling

    gen_images = []
    gen_masks = [] if num_masks > 1 else None
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = images[0].get_shape()[0]
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    for image, action in zip(images[:-1], actions[:-1]):
        with tf.variable_scope("afn_generator_single_step", reuse=len(gen_images) > 0) as scope:
            done_warm_start = len(gen_images) > context_frames - 1
            if feedself and done_warm_start:
                # Feed in generated image.
                image = gen_images[-1]
            elif done_warm_start:
                # Scheduled sampling
                from prediction_model import scheduled_sample
                image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
            # else: Always feed in ground_truth

            image_enc = image
            for conv_layer in conv_layers:
                image_enc = conv_layer(image_enc)

            # Predicted state is always fed back in
            state_action = tf.concat([current_state, action], axis=-1)
            state_action_enc = state_action
            for state_action_fc_layer in state_action_fc_layers:
                state_action_enc = state_action_fc_layer(state_action_enc)

            feat = tf.concat([tf.expand_dims(tf.expand_dims(state_action_enc, 1), 1), image_enc], axis=-1)

            flow_dec = feat
            for flow_fc_layer in flow_fc_layers:
                flow_dec = flow_fc_layer(flow_dec)
            for flow_deconv_layer in flow_deconv_layers:
                flow_dec = flow_deconv_layer(flow_dec)

            if num_masks > 1:
                mask_dec = feat
                for mask_fc_layer in mask_fc_layers:
                    mask_dec = mask_fc_layer(mask_dec)
                for mask_deconv_layer in mask_deconv_layers:
                    mask_dec = mask_deconv_layer(mask_dec)

                flows = tf.split(flow_dec, num_masks - 1, axis=3)
                masks = tf.nn.softmax(mask_dec)
                masks = tf.split(masks, num_masks, axis=3)

                output = image * masks[-1]  # background mask
                for mask, flow in zip(masks, flows):
                    output += mask * flow_transformer(image, flow, (scale_size, scale_size))
                gen_masks.append(masks)
            else:
                flow = flow_dec
                output = flow_transformer(image, flow, (scale_size, scale_size))

            gen_images.append(output)

            current_state = state_fc_layer(state_action)
            gen_states.append(current_state)

    return gen_images, gen_masks, gen_states


def create_rafn_generator(images, states, actions, num_masks=None, scale_size=None, iter_num=None, schedule_sampling_k=None, context_frames=None, **kwargs):
    conv_specs = [
        (16, 2),
        (32, 2),
        (64, 2),
    ]
    state_action_fc_specs = [
        64,
        64
    ]
    flow_fc_specs = [
        64,
        64
    ]
    mask_fc_specs = [
        64,
        64
    ]
    flow_deconv_specs = [
        (64, 2),
        (32, 2),
        (16, 2),
    ]

    if num_masks > 1:
        flow_deconv_specs.append((2 * (num_masks - 1), 1))
        mask_deconv_specs = [
            (64, 2),
            (32, 2),
            (16, 2),
            (num_masks, 1)
        ]
    else:
        mask_deconv_specs = []
        flow_deconv_specs.append((2, 1))

    gen_images = []
    gen_masks = [] if num_masks > 1 else None
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = images[0].get_shape()[0]
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    from prediction_model import basic_conv_lstm_cell as lstm_func
    lstm_states = [None] * (len(conv_specs) + len(flow_fc_specs) + len(flow_deconv_specs) + len(mask_fc_specs) + len(mask_deconv_specs))

    """
    64x64x3

    conv, 32, 5, 2  # 32x32x32
    layer_norm

    lstm, [16]  # 32x32x16
    layer_norm
    conv, 16, 3, 2  # 16x16x16

    lstm, [32]  # 16x16x32
    layer_norm
    conv, 32, 3, 2  # 8x8x32

    smear  # 8x8x38

    conv, 32, 3, 1  # 8x8x32

    lstm, [64]  # 8x8x64
    layer_norm
    convt, 64, 3, 2  # 16x16x64

    lstm, [32]  # 16x16x32
    layer_norm
    concat
    convt, 32, 3, 2  # 32x32x32

    lstm, [16]  # 32x32x16
    layer_norm
    concat
    convt, 16, 3, 2  # 64x64x16

    layer_norm

    convt, K*K, 1, 1  # 64x64xK*K

    """

    for image, action in zip(images[:-1], actions[:-1]):
        with tf.variable_scope("rafn_generator_single_step", reuse=len(gen_images) > 0) as scope:
            done_warm_start = len(gen_images) > context_frames - 1
            if feedself and done_warm_start:
                # Feed in generated image.
                image = gen_images[-1]
            elif done_warm_start:
                # Scheduled sampling
                from prediction_model import scheduled_sample
                image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
            # else: Always feed in ground_truth

            image_enc = image
            with tf.variable_scope('conv0'):
                image_enc = tf.layers.conv2d(image_enc, 32, (5, 5), strides=1, padding='same', activation=tf.nn.relu)
            with tf.variable_scope('layer_norm0'):
                image_enc = tf.contrib.layers.layer_norm(image_enc)

            lstm_state_k = 0

            for k, (filter_dim, stride) in enumerate(conv_specs):
                with tf.variable_scope('lstm%d' % (lstm_state_k + 1)):
                    hidden, lstm_states[lstm_state_k] = lstm_func(image_enc, lstm_states[lstm_state_k], filter_dim)
                    lstm_state_k += 1
                with tf.variable_scope('layer_norm%d' % (k + 1)):
                    hidden = tf.contrib.layers.layer_norm(hidden)
                with tf.variable_scope('conv%d' % (k + 1)):
                    image_enc = tf.layers.conv2d(hidden, filter_dim, (3, 3), strides=stride, padding='same', activation=tf.nn.relu)

            # Predicted state is always fed back in
            state_action = tf.concat([current_state, action], axis=-1)
            state_action_enc = state_action
            for k, unit_dim in enumerate(state_action_fc_specs):
                with tf.variable_scope('state_action_fc%d' % (k + 1)):
                    state_action_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)

            tile_pattern = [1, int(image_enc.shape[1]), int(image_enc.shape[2]), 1]
            feat = tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(state_action_enc, 1), 1), tile_pattern), image_enc], axis=-1)

            flow_dec = feat
            for k, unit_dim in enumerate(flow_fc_specs):
                with tf.variable_scope('lstm%d' % (lstm_state_k + 1)):
                    hidden, lstm_states[lstm_state_k] = lstm_func(flow_dec, lstm_states[lstm_state_k], unit_dim)
                    lstm_state_k += 1
                with tf.variable_scope('flow_fc_layer_norm%d' % (k + 1)):
                    hidden = tf.contrib.layers.layer_norm(hidden)
                with tf.variable_scope('flow_fc%d' % (k + 1)):
                    flow_dec = tf.layers.conv2d(hidden, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu)

            for k, (filter_dim, stride) in enumerate(flow_deconv_specs):
                with tf.variable_scope('lstm%d' % (lstm_state_k + 1)):
                    hidden, lstm_states[lstm_state_k] = lstm_func(flow_dec, lstm_states[lstm_state_k], filter_dim)
                    lstm_state_k += 1
                with tf.variable_scope('flow_deconv_layer_norm%d' % (k + 1)):
                    hidden = tf.contrib.layers.layer_norm(hidden)
                with tf.variable_scope('flow_deconv%d' % (k + 1)):
                    flow_dec = tf.layers.conv2d_transpose(hidden, filter_dim, (3, 3), strides=stride, padding='same',
                                                          activation=None if k == (len(flow_deconv_specs) - 1) else tf.nn.relu)

            if num_masks > 1:
                mask_dec = feat
                for k, unit_dim in enumerate(mask_fc_specs):
                    with tf.variable_scope('lstm%d' % (lstm_state_k + 1)):
                        hidden, lstm_states[lstm_state_k] = lstm_func(mask_dec, lstm_states[lstm_state_k], unit_dim)
                        lstm_state_k += 1
                    with tf.variable_scope('mask_fc_layer_norm%d' % (k + 1)):
                        hidden = tf.contrib.layers.layer_norm(hidden)
                    with tf.variable_scope('mask_fc%d' % (k + 1)):
                        mask_dec = tf.layers.conv2d(hidden, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu)

                for k, (filter_dim, stride) in enumerate(mask_deconv_specs):
                    with tf.variable_scope('lstm%d' % (lstm_state_k + 1)):
                        hidden, lstm_states[lstm_state_k] = lstm_func(mask_dec, lstm_states[lstm_state_k], filter_dim)
                        lstm_state_k += 1
                    with tf.variable_scope('mask_deconv_layer_norm%d' % (k + 1)):
                        hidden = tf.contrib.layers.layer_norm(hidden)
                    with tf.variable_scope('mask_deconv%d' % (k + 1)):
                        mask_dec = tf.layers.conv2d_transpose(hidden, filter_dim, (3, 3), strides=stride, padding='same',
                                                              activation=None if k == (len(mask_deconv_specs) - 1) else tf.nn.relu)

                flows = tf.split(flow_dec, num_masks - 1, axis=3)
                masks = tf.nn.softmax(mask_dec)
                masks = tf.split(masks, num_masks, axis=3)

                output = image * masks[-1]  # background mask
                for mask, flow in zip(masks, flows):
                    output += mask * flow_transformer(image, flow, (scale_size, scale_size))
                gen_masks.append(masks)
            else:
                flow = flow_dec
                output = flow_transformer(image, flow, (scale_size, scale_size))

            gen_images.append(output)

            with tf.variable_scope('state_fc'):
                current_state = tf.layers.dense(state_action, int(states[0].get_shape()[1]), activation=None)
            gen_states.append(current_state)

    return gen_images, gen_masks, gen_states


def create_afn_model(images, states, actions, lr=None, beta1=None, beta2=None, context_frames=None, **kwargs):
    images = tf.unstack(images, axis=1)
    states = tf.unstack(states, axis=1)
    actions = tf.unstack(actions, axis=1)

    # with tf.variable_scope("afn_generator") as scope:
    #     gen_images = create_afn_generator(images, states, actions, **kwargs)

    with tf.variable_scope("afn_generator") as scope:
        gen_images = create_afn_generator(images, states, actions, **kwargs)

    recon_losses = []
    psnr_losses = []
    for image, gen_image in zip(images[context_frames:], gen_images[context_frames - 1:]):
        psnr_loss = peak_signal_to_noise_ratio(image, gen_image)
        recon_loss = tf.reduce_mean(tf.square(image - gen_image))
        # recon_loss = tf.reduce_mean(tf.abs(image - gen_image))
        psnr_losses.append(psnr_loss)
        recon_losses.append(recon_loss)
    gen_images = tf.stack(gen_images, axis=1)
    loss = sum(recon_losses) / np.float32(len(recon_losses))

    optimizer = tf.train.AdamOptimizer(lr, beta1, beta2)
    train = optimizer.minimize(loss)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([loss])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return AfnModel(
        loss=loss,
        recon_losses=recon_losses,
        psnr_losses=psnr_losses,
        outputs=gen_images,
        train=tf.group(update_losses, incr_global_step, train)
    )


def create_p2p_generator(images, states, actions, ngf=None, iter_num=None, schedule_sampling_k=None, context_frames=None, **kwargs):
    enc_layer_specs = [
        (ngf * 2, 1),     # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        (ngf * 4, 2),     # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        (ngf * 8, 2),     # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        (ngf * 8, 2),     # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        (ngf * 8, 2),     # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        (ngf * 8, 2),     # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        (ngf * 8, 2),     # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    state_action_fc_layer_specs = [
        512,
        512
    ]

    fc_layer_specs = [
        512,
        ngf * 8
    ]

    dec_layer_specs = [
        (ngf * 8, 2, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 2, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 1, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    gen_images = []
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = images[0].get_shape()[0]
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    from prediction_model import basic_conv_lstm_cell as lstm_func
    lstm_states = [None] * len(fc_layer_specs)

    for image, action in zip(images[:-1], actions[:-1]):
        with tf.variable_scope("p2p_generator_single_step", reuse=len(gen_images) > 0) as scope:
            done_warm_start = len(gen_images) > context_frames - 1
            if feedself and done_warm_start:
                # Feed in generated image.
                image = None
                image_enc = next_image_enc
            elif done_warm_start:
                # Scheduled sampling
                # split into current_image and image_enc from previous next_image_enc
                idx = tf.random_shuffle(tf.range(int(batch_size)))
                image_idx = tf.gather(idx, tf.range(num_ground_truth))
                image_enc_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))
                image = tf.gather(image, image_idx)
                image_enc = tf.gather(next_image_enc, image_enc_idx)
                prev_layers = [tf.gather(layer, image_enc_idx) for layer in layers]
            else:
                # Always feed in ground_truth
                image_enc = None

            if image is not None:
                layers = []
                with tf.variable_scope("encoder_1"):
                    output = conv(image, ngf, stride=1)
                    layers.append(output)

                for out_channels, stride in enc_layer_specs:
                    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                        rectified = lrelu(layers[-1], 0.2)
                        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                        convolved = conv(rectified, out_channels, stride=stride)
                        output = batchnorm(convolved)
                        layers.append(output)

                if image_enc is not None:
                    # combine image_enc derived from current image or derived from the previous next_image_enc
                    image_enc = tf.dynamic_stitch([image_idx, image_enc_idx], [output, image_enc])
                    image_enc = tf.reshape(image_enc, [int(batch_size)] + image_enc.shape.as_list()[1:])

                    curr_layers = layers
                    layers = []
                    for prev_layer, curr_layer in zip(prev_layers, curr_layers):
                        layer = tf.dynamic_stitch([image_idx, image_enc_idx], [curr_layer, prev_layer])
                        layer = tf.reshape(layer, [int(batch_size)] + layer.shape.as_list()[1:])
                        layers.append(layer)
                else:
                    image_enc = output
            else:
                assert image_enc is not None
                layers = prev_layers

            # Predicted state is always fed back in
            state_action = tf.concat([current_state, action], axis=-1)
            state_action_enc = state_action
            for k, unit_dim in enumerate(state_action_fc_layer_specs):
                with tf.variable_scope('state_action_fc%d' % (k + 1)):
                    state_action_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)

            tile_pattern = [1, int(image_enc.shape[1]), int(image_enc.shape[2]), 1]
            next_image_enc = tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(state_action_enc, 1), 1), tile_pattern), image_enc], axis=-1)

            lstm_state_k = 0

            for k, unit_dim in enumerate(fc_layer_specs):
                with tf.variable_scope('lstm%d' % (lstm_state_k + 1)):
                    hidden, lstm_states[lstm_state_k] = lstm_func(next_image_enc, lstm_states[lstm_state_k], unit_dim)
                    lstm_state_k += 1
                with tf.variable_scope('layer_norm%d' % (k + 1)):
                    hidden = tf.contrib.layers.layer_norm(hidden)
                with tf.variable_scope('fc%d' % (k + 1)):
                    next_image_enc = tf.layers.conv2d(hidden, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu)

            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, stride, dropout) in enumerate(dec_layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = next_image_enc
                    else:
                        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                    rectified = tf.nn.relu(input)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    output = deconv(rectified, out_channels, stride=stride)
                    output = batchnorm(output)

                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    layers.append(output)

            with tf.variable_scope("decoder_1"):
                input = tf.concat([layers[-1], layers[0]], axis=3)
                rectified = tf.nn.relu(input)
                output = deconv(rectified, 3, stride=1)
                output = tf.tanh(output)
                layers.append(output)

            gen_images.append(output)

            with tf.variable_scope('state_fc'):
                current_state = tf.layers.dense(state_action, int(states[0].get_shape()[1]), activation=None)
            gen_states.append(current_state)

    return gen_images, None, gen_states


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
    from keras import backend as K
    return K.switch(tf.equal(num_ground_truth, 0), generated_xs, outs)


def create_mask_generator(images, masks, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                          preprocess_image=None, remove_background=None, **kwargs):
    from transformer.spatial_transformer import transformer

    filter_dims = [16, 16, 16, 32, 32, 32]
    enc_unit_dims = [32, 32]
    dec_unit_dims = [32, 32, 2]
    pair_enc_reuse = False

    gen_images = []
    gen_masks = []

    batch_size = int(images[0].get_shape()[0])
    num_masks = int(masks[0].get_shape()[-1])

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    identity_params = tf.convert_to_tensor(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
    identity_params = tf.tile(identity_params[None, :, None], (batch_size, 1, num_masks))

    regularization_losses = []

    for image, mask, state, action in zip(images[:-1], masks[:-1], states[:-1], actions[:-1]):
        with tf.variable_scope("generator_single_step", reuse=len(gen_masks) > 0) as scope:
            done_warm_start = len(gen_images) > context_frames - 1
            if feedself and done_warm_start:
                # same base_mask and params as before
                pass
            elif done_warm_start:
                base_image, base_mask, params = scheduled_samples([image, mask, identity_params], [base_image, base_mask, params], batch_size, num_ground_truth)
            else:
                base_image = image
                base_mask = mask
                params = identity_params

            delta_params = []
            for i in range(num_masks):
                sum_enc = 0
                for j in range(num_masks):
                    if i == j:
                        continue
                    enc = tf.stack([mask[..., i], mask[..., j]], axis=-1)
                    with tf.variable_scope('conv_0', reuse=pair_enc_reuse):
                        enc_0 = conv(enc, 4, stride=1, kernel_size=(10, 10))
                    with tf.variable_scope('conv_1', reuse=pair_enc_reuse):
                        enc_1 = conv(enc, 16, stride=1, kernel_size=(3, 3))
                    enc = tf.concat([enc_0, enc_1], axis=-1)
                    enc = tf.nn.relu(enc)
                    with tf.variable_scope('conv_2', reuse=pair_enc_reuse):
                        enc = conv(enc, 16, stride=1, kernel_size=(3, 3))
                        enc = tf.nn.relu(enc)
                    for k, filter_dim in enumerate(filter_dims):
                        with tf.variable_scope('conv_%d' % (k + 3), reuse=pair_enc_reuse):
                            enc = conv(enc, filter_dim, stride=1, kernel_size=(3, 3))
                            enc = tf.nn.relu(enc)
                            enc = tf.layers.max_pooling2d(enc, 2, strides=2, padding='same')
                    enc = tf.reshape(enc, (batch_size, -1))
                    for k, unit_dim in enumerate(enc_unit_dims):
                        with tf.variable_scope('enc_dense_%d' % k, reuse=pair_enc_reuse):
                            enc = tf.layers.dense(enc, unit_dim, activation=None if k == (len(enc_unit_dims) - 1) else tf.nn.relu)
                    sum_enc += enc
                    pair_enc_reuse = True
                dec = tf.concat([sum_enc, state, action], axis=-1)
                for k, unit_dim in enumerate(dec_unit_dims):
                    with tf.variable_scope('dec_dense_%d_%d' % (i, k)):
                        dec = tf.layers.dense(dec, unit_dim, activation=None if k == (len(dec_unit_dims) - 1) else tf.nn.relu)
                delta_params.append(tf.concat([tf.zeros((batch_size, 2), dtype=np.float32), dec[:, 0:1],
                                               tf.zeros((batch_size, 2), dtype=np.float32), dec[:, 1:2]], axis=-1))
            delta_params = tf.stack(delta_params, axis=-1)
            # TODO: use general addition for arbitrary transformations
            params = params + delta_params

            regularization_losses.append(tf.reduce_mean(tf.square(delta_params)))

            gen_mask = []
            for i in range(num_masks):
                gen_mask.append(transformer(base_mask[..., i:i+1], params[..., i], base_mask.shape[1:3]))
            gen_mask = tf.reshape(tf.concat(gen_mask, axis=-1), base_mask.shape)
            gen_masks.append(gen_mask)

            gen_image = 0
            for i in range(num_masks):
                if i == 0:
                    continue
                gen_image += transformer(base_image * base_mask[..., i:i+1], params[..., i], base_mask.shape[1:3])
            assert not preprocess_image and remove_background
            gen_image = tf.clip_by_value(gen_image, 0.0, 1.0)
            gen_images.append(gen_image)

            # # generate image from image, masks, and 1x1 convolution for mixing
            # gen_image = []
            # for i in range(num_masks):
            #     gen_image.append(transformer(base_image * base_mask[..., i:i+1], params[..., i], base_mask.shape[1:3]))
            # gen_image = tf.reshape(tf.concat(gen_image, axis=-1), (batch_size,
            #                                                        int(base_image.shape[1]),
            #                                                        int(base_image.shape[2]),
            #                                                        int(base_image.shape[3]) * num_masks))
            # with tf.variable_scope('mix_conv'):
            #     gen_image = conv(gen_image, int(base_image.shape[3]), stride=1, kernel_size=(1, 1))
            # gen_images.append(gen_image)

    return gen_images, gen_masks, regularization_losses


def cdna_transformations(prev_images, cdna_input, num_masks, color_channels):
    """Apply convolutional dynamic neural advection to previous image.

    Args:
        prev_image: previous image to be transformed.
        cdna_input: hidden lyaer to be used for computing CDNA kernels.
        num_masks: the number of masks and hence the number of CDNA transformations.
        color_channels: the number of color channels in the images.
    Returns:
        List of images transformed by the predicted CDNA kernels.
    """
    import tensorflow.contrib.slim as slim
    DNA_KERN_SIZE = 5
    RELU_SHIFT = 1e-12

    batch_size = int(cdna_input.get_shape()[0])

    # Predict kernels using linear function of last hidden layer.
    cdna_kerns = slim.layers.fully_connected(
        cdna_input,
        DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
        scope='cdna_params',
        activation_fn=None)

    # Reshape and normalize.
    cdna_kerns = tf.reshape(
        cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
    cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
    norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
    cdna_kerns /= norm_factor

    cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, color_channels, 1])
    cdna_kerns = tf.split(axis=0, num_or_size_splits=batch_size, value=cdna_kerns)
    prev_images = tf.stack(prev_images, axis=-1)
    prev_images = tf.split(axis=0, num_or_size_splits=batch_size, value=prev_images)

    transformed_images = []
    for kernel, prev_image in zip(cdna_kerns, prev_images):
        kernel = tf.squeeze(kernel, axis=0)
        transformed_images_ = []
        for i in range(num_masks):
            transformed_image = tf.nn.depthwise_conv2d(prev_image[..., i], kernel[..., i:i + 1], [1, 1, 1, 1], 'SAME')
            transformed_images_.append(transformed_image)
        transformed_images.append(tf.stack(transformed_images_, axis=-1))
    transformed_images = tf.concat(axis=0, values=transformed_images)
    transformed_images = tf.unstack(transformed_images, axis=-1)  # same as below
    # transformed_images = tf.split(axis=4, num_or_size_splits=num_masks, value=transformed_images)
    # transformed_images = [tf.squeeze(transformed_image, axis=-1) for transformed_image in transformed_images]
    return transformed_images


def create_accum_tf_generator(images, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                              num_masks=None, stp=False, cdna=True, dna=False, **kwargs):
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib.layers.python import layers as tf_layers
    from transformer.spatial_transformer import transformer
    from prediction_model import cdna_transformation, dna_transformation, stp_transformation
    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)
    from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    gen_transformed_images = []
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
    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                prev_image = gen_images[-1]
                transformed_images = gen_transformed_images[-1][1:]
            elif done_warm_start:
                prev_image, *transformed_images = \
                    scheduled_samples([image] * num_masks,
                                      [gen_images[-1], *gen_transformed_images[-1][1:]],
                                      batch_size, num_ground_truth)
            else:
                prev_image = image
                transformed_images = [image] * (num_masks - 1)

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])

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
                                                    int(color_channels))
            elif dna:
                raise NotImplementedError
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7)]

            masks = slim.layers.conv2d_transpose(
                enc6, num_masks + 1, 1, stride=1, scope='convt7')
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
                [int(batch_size), int(img_height), int(img_width), num_masks + 1])
            mask_list = tf.split(axis=3, num_or_size_splits=num_masks + 1, value=masks)
            output = mask_list[0] * prev_image
            for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask
            gen_images.append(output)
            gen_masks.append(mask_list)

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)

            gen_transformed_images.append(transformed)

    return gen_images, gen_masks, gen_states, gen_transformed_images


def create_accum_tf_factorized_generator(images, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                                         num_masks=None, dependent_mask=None, stp=False, cdna=True, dna=False, **kwargs):
    """
    like create_accum_tf_generator except the transformed images are the inputs of the convnet
    and that the masks depend on the transformed images
    """
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib.layers.python import layers as tf_layers
    from transformer.spatial_transformer import transformer
    from prediction_model import cdna_transformation, dna_transformation, stp_transformation
    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)
    from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    gen_transformed_images = []
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
    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                prev_image = gen_images[-1]
                transformed_images = gen_transformed_images[-1][2:]
            elif done_warm_start:
                prev_image, *transformed_images = \
                    scheduled_samples([image] * num_masks,
                                      [gen_images[-1], *gen_transformed_images[-1][2:]],
                                      batch_size, num_ground_truth)
            else:
                prev_image = image
                transformed_images = [image] * (num_masks - 1)

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])

            enc0 = slim.layers.conv2d(    #32x32x32
                tf.concat(transformed_images, axis=-1),
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
                                                    int(color_channels))
            elif dna:
                raise NotImplementedError
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7)]

            if dependent_mask:
                masks = slim.layers.conv2d_transpose(
                    tf.concat([enc6] + transformed, axis=-1), num_masks + 1, 1, stride=1, scope='convt7')
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

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)

            gen_transformed_images.append(transformed)

    return gen_images, gen_masks, gen_states, gen_transformed_images


def create_accum_tf_factorized_unet_generator(images, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                                              truncate_hidden_decs=None,
                                              num_masks=None, dependent_mask=None, stp=False, cdna=True, dna=False, **kwargs):
    """
    like create_accum_tf_generator except the transformed images are the inputs of the convnet
    and that the masks depend on the transformed images
    """
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib.layers.python import layers as tf_layers
    from transformer.spatial_transformer import transformer
    from prediction_model import cdna_transformation, dna_transformation, stp_transformation
    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)
    from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    gen_transformed_images = []
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
    ngf = 16
    enc_layer_specs = [
        ngf * 2,
        ngf * 4,
        ngf * 4,
        ngf * 4,
        ngf * 4,
        ngf * 4,
    ]
    dec_layer_specs = [
        ngf * 4,
        ngf * 4,
        ngf * 4,
        ngf * 4,
        ngf * 4,
        ngf * 2,
    ]
    lstm_states = [None] * (len(enc_layer_specs) + len(dec_layer_specs))

    # actual mask is unused
    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                prev_image = gen_images[-1]
                transformed_images = gen_transformed_images[-1][2:]
            elif done_warm_start:
                prev_image, *transformed_images = \
                    scheduled_samples([image] * num_masks,
                                      [gen_images[-1], *gen_transformed_images[-1][2:]],
                                      batch_size, num_ground_truth)
            else:
                prev_image = image
                transformed_images = [image] * (num_masks - 1)

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])

            encs = []
            enc = slim.layers.conv2d(tf.concat(transformed_images, axis=-1),
                                     ngf, [3, 3], stride=1, scope='encoder_conv1')
            enc = tf_layers.layer_norm(enc, scope='encoder_layer_norm1')
            encs.append(enc)

            lstm_state_k = 0
            for k, out_channels in enumerate(enc_layer_specs):
                hidden, lstm_states[lstm_state_k] = lstm_func(
                    enc, lstm_states[lstm_state_k], out_channels, scope='encoder_lstm%d' % (len(encs) + 1))
                lstm_state_k += 1
                hidden = tf_layers.layer_norm(hidden, scope='encoder_layer_norm%d' % (len(encs) + 1))
                enc = slim.layers.conv2d(
                    hidden, hidden.get_shape()[3], [3, 3], stride=2, scope='encoder_conv%d' % (len(encs) + 1))
                encs.append(enc)

            # action-contined encodings
            ac_encs = []
            for enc in encs:
                smear = tf.expand_dims(tf.expand_dims(state_action, 1), 1)
                smear = tf.tile(
                    smear, [1, int(enc.get_shape()[1]), int(enc.get_shape()[2]), 1])
                ac_enc = tf.concat([enc, smear], axis=3)
                ac_enc = slim.layers.conv2d(
                    ac_enc, enc.get_shape()[3], [1, 1], stride=1, scope='smear_conv%d' % (len(ac_encs) + 1))
                ac_encs.append(ac_enc)

            decs = []
            hidden_decs = []
            for k, out_channels in enumerate(dec_layer_specs):
                skip_k = len(ac_encs) - k - 1
                if k == 0:
                    dec = ac_encs[-1]
                else:
                    dec = decs[-1]
                hidden, lstm_states[lstm_state_k] = lstm_func(
                    dec, lstm_states[lstm_state_k], out_channels, scope='decoder_lstm%d' % (skip_k + 1))
                lstm_state_k += 1
                hidden = tf_layers.layer_norm(hidden, scope='decoder_layer_norm%d' % (skip_k + 1))
                hidden_decs.append(hidden)
                if k != 0:
                    hidden = tf.concat([hidden, ac_encs[skip_k]], axis=3)
                dec = slim.layers.conv2d_transpose(
                    hidden, hidden.get_shape()[3], [3, 3], stride=2, scope='decoder_convt%d' % (skip_k + 1))
                decs.append(dec)

            enc6 = decs[-1]
            if truncate_hidden_decs:
                hidden5 = tf.concat([tf.reshape(hidden_dec, [int(batch_size), -1]) for hidden_dec in hidden_decs[:4]], axis=1)
            else:
                hidden5 = tf.concat([tf.reshape(hidden_dec, [int(batch_size), -1]) for hidden_dec in hidden_decs], axis=1)

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
                                                    int(color_channels))
            elif dna:
                raise NotImplementedError
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7)]

            if dependent_mask:
                # TODO: include prev_image in transformed?
                masks = slim.layers.conv2d_transpose(
                    tf.concat([enc6] + transformed, axis=-1), num_masks + 1, 1, stride=1, scope='convt7')
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

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)

            gen_transformed_images.append(transformed)

    return gen_images, gen_masks, gen_states, gen_transformed_images


def vgg_assign_from_values_fn(var_name_prefix='generator/',
                              var_name_kernel_postfix='/kernel:0',
                              var_name_bias_postfix='/bias:0'):
    from keras.utils.data_utils import get_file
    import h5py
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    weights_file = h5py.File(weights_path, 'r')

    weight_name_kernel_postfix = '_W_1:0'
    weight_name_bias_postfix = '_b_1:0'
    var_names_to_values = {}
    for block_id in range(5):
        for conv_id in range(3):
            if block_id < 2 and conv_id == 2:
                continue
            name = 'block%d_conv%d' % (block_id + 1, conv_id + 1)
            var_names_to_values[var_name_prefix + name + var_name_kernel_postfix] = \
                weights_file[name][name + weight_name_kernel_postfix][()]
            var_names_to_values[var_name_prefix + name + var_name_bias_postfix] = \
                weights_file[name][name + weight_name_bias_postfix][()]
    return tf.contrib.framework.assign_from_values_fn(var_names_to_values)


def vgg_stats_assign_from_values_fn(var_name_prefix='generator/'):
    var_names = []
    for block_id in range(5):
        var_names.append(var_name_prefix + 'block%d_standarize/offset:0' % (block_id + 1))
        var_names.append(var_name_prefix + 'block%d_standarize/scale:0' % (block_id + 1))
    assign_ops = []
    placeholder_values = []
    for var_name in var_names:
        var, = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_name)
        placeholder_name = 'placeholder/' + var.op.name
        placeholder_value = tf.placeholder(
            dtype=var.dtype.base_dtype,
            shape=var.get_shape(),
            name=placeholder_name)
        assign_ops.append(var.assign(placeholder_value))
        placeholder_values.append(placeholder_value)
    assign_op = tf.group(*assign_ops)
    def callback(session, values):
        return session.run(assign_op, dict(zip(placeholder_values, values)))
    return callback


def create_vgg_afn_generator(images, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                             num_masks=None, dependent_mask=None, preprocess_image=None, **kwargs):
    """
    [x] VGG16 to extract features
    [x] from those features, predict masks with softmax
    [x]     from those, features, predict appearance flows
    [x] use gt state
    [x] blinear interaction as in oh et all paper?
    recurrence on outputs, e.g. flow and masks
    [x] freeeze vgg weights
    should masks have a relu before?
    """
    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    current_state = states[0]
    gen_transformed_images = []

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    # actual mask is unused
    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        if feedself and done_warm_start:
            prev_image = gen_images[-1]
        elif done_warm_start:
            from prediction_model import scheduled_sample
            prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
            prev_image = tf.reshape(prev_image, image.shape)
        else:
            prev_image = image

        # Predicted state is always fed back in
        # state_action = tf.concat(axis=1, values=[action, current_state])
        # TODO: using ground truth state for now
        state_action = tf.concat(axis=1, values=[action, state])

        # vgg16 preprocessing
        x = deprocess(prev_image) if preprocess_image else prev_image
        x = x * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]

        # Block1
        x = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block1_conv1', reuse=reuse)
        x = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block1_conv2', reuse=reuse)
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block1_pool')

        # Block2
        x = tf.layers.conv2d(x, 128, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block2_conv1', reuse=reuse)
        x = tf.layers.conv2d(x, 128, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block2_conv2', reuse=reuse)
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block2_pool')

        # Block3
        x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv1', reuse=reuse)
        x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv2', reuse=reuse)
        x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv3', reuse=reuse)
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block3_pool')

        # Block4
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv1', reuse=reuse)
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv2', reuse=reuse)
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv3', reuse=reuse)
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block4_pool')

        # Block5
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv1', reuse=reuse)
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv2', reuse=reuse)
        x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv3', reuse=reuse)
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block5_pool')

        x = tf.reshape(x, (batch_size, 2048))
        x = tf.layers.dense(x, 2048, activation=tf.nn.relu, name='x_dense1', reuse=reuse)
        x = tf.layers.dense(x, 2048, activation=None, name='x_dense2', reuse=reuse)
        u = tf.layers.dense(state_action, 2048, activation=None, name='u_dense', reuse=reuse)
        xu = tf.layers.dense(x * u, 2048, activation=None, name='xu_dense1', reuse=reuse)
        xu = tf.layers.dense(xu, 2048, activation=tf.nn.relu, name='xu_dense2', reuse=reuse)
        xu = tf.reshape(xu, (batch_size, 2, 2, 512))

        # flow decoder
        flows = tf.layers.conv2d_transpose(xu, 512, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='flows_deconv1', reuse=reuse)
        flows = tf.layers.conv2d_transpose(flows, 256, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='flows_deconv2', reuse=reuse)
        flows = tf.layers.conv2d_transpose(flows, 128, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='flows_deconv3', reuse=reuse)
        flows = tf.layers.conv2d_transpose(flows, 64, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='flows_deconv4', reuse=reuse)
        flows = tf.layers.conv2d_transpose(flows, 32, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='flows_deconv5', reuse=reuse)
        flows = tf.layers.conv2d_transpose(flows, 2 * num_masks, (3, 3), (1, 1), padding='same', activation=None, name='flows_deconv6', reuse=reuse)
        # flows = 5 * tf.nn.tanh(flows)
        flows = tf.split(flows, num_masks, axis=3)

        # transformed images
        transformed_images = [prev_image]
        for flow in flows:
            transformed_image = flow_transformer(prev_image, flow, (int(img_height), int(img_width)))
            transformed_image = tf.reshape(transformed_image, prev_image.shape)
            transformed_images.append(transformed_image)

        # mask decoder
        masks = tf.layers.conv2d_transpose(xu, 512, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='masks_deconv1', reuse=reuse)
        masks = tf.layers.conv2d_transpose(masks, 256, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='masks_deconv2', reuse=reuse)
        masks = tf.layers.conv2d_transpose(masks, 128, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='masks_deconv3', reuse=reuse)
        masks = tf.layers.conv2d_transpose(masks, 64, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='masks_deconv4', reuse=reuse)
        masks = tf.layers.conv2d_transpose(masks, 32, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='masks_deconv5', reuse=reuse)
        if dependent_mask:
            masks = tf.concat([masks] + transformed_images, axis=3)
        masks = tf.layers.conv2d_transpose(masks, num_masks + 1, (3, 3), (1, 1), padding='same', activation=None, name='masks_deconv6', reuse=reuse)
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
            [int(batch_size), int(img_height), int(img_width), num_masks + 1])
        masks = tf.split(masks, num_masks + 1, axis=3)

        # composition
        output = 0
        for transformed_image, mask in zip(transformed_images, masks):
            output += transformed_image * mask
        gen_images.append(output)
        gen_masks.append(masks)

        current_state = tf.layers.dense(state_action, int(current_state.get_shape()[1]), activation=None, name='state_pred', reuse=reuse)
        gen_states.append(current_state)

        gen_transformed_images.append(transformed_images)

    return gen_images, gen_masks, gen_states, gen_transformed_images


def create_vgg_accum_tf_factorized_generator(images, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                                             preprocess_image=None, use_layer_norm=None, use_lstm=None,
                                             num_masks=None, dependent_mask=None, stp=False, cdna=True, dna=False,
                                             **kwargs):
    import tensorflow.contrib.slim as slim
    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)
    from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    current_state = states[0]
    gen_transformed_images = []
    preactivation_feature_maps = []
    feature_maps = []

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    lstm_state_x0t = None
    lstm_state_x0tm = None

    # actual mask is unused
    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                prev_image = gen_images[-1]
                transformed_images = gen_transformed_images[-1][2:]
            elif done_warm_start:
                prev_image, *transformed_images = \
                    scheduled_samples([image] * num_masks,
                                      [gen_images[-1], *gen_transformed_images[-1][2:]],
                                      batch_size, num_ground_truth)
            else:
                prev_image = image
                transformed_images = [image] * (num_masks - 1)

            # Predicted state is always fed back in
            # state_action = tf.concat(axis=1, values=[action, current_state])
            # TODO: using ground truth state for now
            state_action = tf.concat(axis=1, values=[action, state])

            # vgg16 preprocessing
            x = deprocess(prev_image) if preprocess_image else prev_image
            # 'RGB'->'BGR'
            x = x[:, :, :, ::-1]
            x = x * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))

            def standarize(xlevel, name, reuse):
                with tf.variable_scope(name, reuse=reuse):
                    x_offset = tf.get_variable('offset', xlevel.shape[3], dtype=xlevel.dtype,
                                               initializer=tf.zeros_initializer(), trainable=False)
                    x_scale = tf.get_variable('scale', xlevel.shape[3], dtype=xlevel.dtype,
                                              initializer=tf.ones_initializer(), trainable=False)
                    return (xlevel - x_offset[None, None, None, :]) / x_scale[None, None, None, :]


            # Block1
            x1 = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block1_conv1', reuse=reuse)
            x1 = tf.layers.conv2d(x1, 64, (3, 3), padding='same', activation=None, trainable=False, name='block1_conv2', reuse=reuse)
            preactivation_x1 = x1
            x1 = tf.nn.relu(x1)
            x1 = tf.layers.max_pooling2d(x1, (2, 2), (2, 2), padding='same', name='block1_pool')

            # Block2
            x2 = tf.layers.conv2d(x1, 128, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block2_conv1', reuse=reuse)
            x2 = tf.layers.conv2d(x2, 128, (3, 3), padding='same', activation=None, trainable=False, name='block2_conv2', reuse=reuse)
            preactivation_x2 = x2
            x2 = tf.nn.relu(x2)
            x2 = tf.layers.max_pooling2d(x2, (2, 2), (2, 2), padding='same', name='block2_pool')

            # Block3
            x3 = tf.layers.conv2d(x2, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv1', reuse=reuse)
            x3 = tf.layers.conv2d(x3, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv2', reuse=reuse)
            x3 = tf.layers.conv2d(x3, 256, (3, 3), padding='same', activation=None, trainable=False, name='block3_conv3', reuse=reuse)
            preactivation_x3 = x3
            x3 = tf.nn.relu(x3)
            # x3 = tf.layers.max_pooling2d(x3, (2, 2), (2, 2), padding='same', name='block3_pool')

            # Block4
            x4 = tf.layers.conv2d(x3, 512, (3, 3), padding='same', dilation_rate=(2, 2), activation=tf.nn.relu, trainable=False, name='block4_conv1', reuse=reuse)
            x4 = tf.layers.conv2d(x4, 512, (3, 3), padding='same', dilation_rate=(2, 2), activation=tf.nn.relu, trainable=False, name='block4_conv2', reuse=reuse)
            x4 = tf.layers.conv2d(x4, 512, (3, 3), padding='same', dilation_rate=(2, 2), activation=None, trainable=False, name='block4_conv3', reuse=reuse)
            preactivation_x4 = x4
            x4 = tf.nn.relu(x4)
            # x4 = tf.layers.max_pooling2d(x4, (2, 2), (2, 2), padding='same', name='block4_pool')

            # Block5
            x5 = tf.layers.conv2d(x4, 512, (3, 3), padding='same', dilation_rate=(4, 4), activation=tf.nn.relu, trainable=False, name='block5_conv1', reuse=reuse)
            x5 = tf.layers.conv2d(x5, 512, (3, 3), padding='same', dilation_rate=(4, 4), activation=tf.nn.relu, trainable=False, name='block5_conv2', reuse=reuse)
            x5 = tf.layers.conv2d(x5, 512, (3, 3), padding='same', dilation_rate=(4, 4), activation=None, trainable=False, name='block5_conv3', reuse=reuse)
            preactivation_x5 = x5
            x5 = tf.nn.relu(x5)
            # x5 = tf.layers.max_pooling2d(x5, (2, 2), (2, 2), padding='same', name='block5_pool')

            # preactivation_xlevels = [preactivation_x1, preactivation_x2, preactivation_x3, preactivation_x4, preactivation_x5]
            # xlevels = [x1, x2, x3, x4, x5]
            # for level, preactivation_xlevel in enumerate(preactivation_xlevels):
            #     preactivation_xlevel = standarize(preactivation_xlevel, name='block%d_standarize' % (level + 1), reuse=reuse)
            #     preactivation_xlevels[level] = preactivation_xlevel
            #     xlevel = tf.nn.relu(preactivation_xlevel)
            #     if level < 2:
            #         xlevel = tf.layers.max_pooling2d(xlevel, (2, 2), (2, 2), padding='same', name='block%d_pool' % (level + 1))
            #     assert xlevel.shape.as_list() == xlevels[level].shape.as_list()
            #     xlevels[level] = xlevel

            xlevels = [x1, x2, x3, x4, x5]
            for level, xlevel in enumerate(xlevels):
                xlevels[level] = standarize(xlevel, name='block%d_standarize' % (level + 1), reuse=reuse)
            preactivation_xlevels = xlevels

            # Pass in state and action.
            next_xlevels = []
            for level, xlevel in enumerate(xlevels):
                smear = tf.expand_dims(tf.expand_dims(state_action, 1), 1)
                smear = tf.tile(smear, [1, int(xlevel.get_shape()[1]), int(xlevel.get_shape()[2]), 1])
                next_xlevel = tf.concat([xlevel, smear], axis=3)
                next_xlevel = tf.layers.conv2d(next_xlevel, xlevel.get_shape()[3], (1, 1),
                                               padding='same', activation=tf.nn.relu,
                                               name='smear_conv%d' % (level + 1), reuse=reuse)
                next_xlevels.append(next_xlevel)
            next_x1, next_x2, next_x3, next_x4, next_x5 = next_xlevels

            if use_layer_norm:
                x4t = tf.layers.conv2d_transpose(next_x5,
                                                 512, (3, 3), (1, 1), padding='same', activation=None, name='block5_deconv', reuse=reuse)
                x4t = tf.contrib.layers.layer_norm(x4t, activation_fn=tf.nn.relu, scope='block5_deconv_ln', reuse=reuse)
                x3t = tf.layers.conv2d_transpose(tf.concat([next_x4, x4t], axis=3),
                                                 512, (3, 3), (1, 1), padding='same', activation=None, name='block4_deconv', reuse=reuse)
                x3t = tf.contrib.layers.layer_norm(x3t, activation_fn=tf.nn.relu, scope='block4_deconv_ln', reuse=reuse)
                x2t = tf.layers.conv2d_transpose(tf.concat([next_x3, x3t], axis=3),
                                                 256, (3, 3), (1, 1), padding='same', activation=None, name='block3_deconv', reuse=reuse)
                x2t = tf.contrib.layers.layer_norm(x2t, activation_fn=tf.nn.relu, scope='block3_deconv_ln', reuse=reuse)
                x1t = tf.layers.conv2d_transpose(tf.concat([next_x2, x2t], axis=3),
                                                 128, (3, 3), (2, 2), padding='same', activation=None, name='block2_deconv', reuse=reuse)
                x1t = tf.contrib.layers.layer_norm(x1t, activation_fn=tf.nn.relu, scope='block2_deconv_ln', reuse=reuse)
                x0t = tf.layers.conv2d_transpose(tf.concat([next_x1, x1t], axis=3),
                                                 64, (3, 3), (2, 2), padding='same', activation=None, name='block1_deconv', reuse=reuse)
                x0t = tf.contrib.layers.layer_norm(x0t, activation_fn=tf.nn.relu, scope='block1_deconv_ln', reuse=reuse)

                x4tm = tf.layers.conv2d_transpose(next_x5,
                                                  512, (3, 3), (1, 1), padding='same', activation=None, name='block5_mask_deconv', reuse=reuse)
                x4tm = tf.contrib.layers.layer_norm(x4tm, activation_fn=tf.nn.relu, scope='block5_mask_deconv_ln', reuse=reuse)
                x3tm = tf.layers.conv2d_transpose(tf.concat([next_x4, x4tm], axis=3),
                                                  512, (3, 3), (1, 1), padding='same', activation=None, name='block4_mask_deconv', reuse=reuse)
                x3tm = tf.contrib.layers.layer_norm(x3tm, activation_fn=tf.nn.relu, scope='block4_mask_deconv_ln', reuse=reuse)
                x2tm = tf.layers.conv2d_transpose(tf.concat([next_x3, x3tm], axis=3),
                                                  256, (3, 3), (1, 1), padding='same', activation=None, name='block3_mask_deconv', reuse=reuse)
                x2tm = tf.contrib.layers.layer_norm(x2tm, activation_fn=tf.nn.relu, scope='block3_mask_deconv_ln', reuse=reuse)
                x1tm = tf.layers.conv2d_transpose(tf.concat([next_x2, x2tm], axis=3),
                                                  128, (3, 3), (2, 2), padding='same', activation=None, name='block2_mask_deconv', reuse=reuse)
                x1tm = tf.contrib.layers.layer_norm(x1tm, activation_fn=tf.nn.relu, scope='block2_mask_deconv_ln', reuse=reuse)
                x0tm = tf.layers.conv2d_transpose(tf.concat([next_x1, x1tm], axis=3),
                                                  64, (3, 3), (2, 2), padding='same', activation=None, name='block1_mask_deconv', reuse=reuse)
                x0tm = tf.contrib.layers.layer_norm(x0tm, activation_fn=tf.nn.relu, scope='block1_mask_deconv_ln', reuse=reuse)
            else:
                x4t = tf.layers.conv2d_transpose(next_x5,
                                                 512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='block5_deconv', reuse=reuse)
                x3t = tf.layers.conv2d_transpose(tf.concat([next_x4, x4t], axis=3),
                                                 512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='block4_deconv', reuse=reuse)
                x2t = tf.layers.conv2d_transpose(tf.concat([next_x3, x3t], axis=3),
                                                 256, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='block3_deconv', reuse=reuse)
                x1t = tf.layers.conv2d_transpose(tf.concat([next_x2, x2t], axis=3),
                                                 128, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='block2_deconv', reuse=reuse)
                x0t = tf.layers.conv2d_transpose(tf.concat([next_x1, x1t], axis=3),
                                                 64, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='block1_deconv', reuse=reuse)

                x4tm = tf.layers.conv2d_transpose(next_x5,
                                                  512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='block5_mask_deconv', reuse=reuse)
                x3tm = tf.layers.conv2d_transpose(tf.concat([next_x4, x4tm], axis=3),
                                                  512, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='block4_mask_deconv', reuse=reuse)
                x2tm = tf.layers.conv2d_transpose(tf.concat([next_x3, x3tm], axis=3),
                                                  256, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='block3_mask_deconv', reuse=reuse)
                x1tm = tf.layers.conv2d_transpose(tf.concat([next_x2, x2tm], axis=3),
                                                  128, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='block2_mask_deconv', reuse=reuse)
                x0tm = tf.layers.conv2d_transpose(tf.concat([next_x1, x1tm], axis=3),
                                                  64, (3, 3), (2, 2), padding='same', activation=tf.nn.relu, name='block1_mask_deconv', reuse=reuse)

            if use_lstm:
                x0t, lstm_state_x0t = lstm_func(x0t, lstm_state_x0t, 64, scope='lstm_x0t')
                x0t = tf.contrib.layers.layer_norm(x0t, scope='lstm_x0t_ln', reuse=reuse)
                x0tm, lstm_state_x0tm = lstm_func(x0tm, lstm_state_x0tm, 64, scope='lstm_x0tm')
                x0tm = tf.contrib.layers.layer_norm(x0tm, scope='lstm_x0tm_ln', reuse=reuse)

            # separate decoders for transformations and masks
            hidden5 = next_x5
            enc6 = x0t
            DNA_KERN_SIZE = 5
            if dna:
                raise NotImplementedError
                # Using largest hidden state for predicting untied conv kernels.
                enc7 = tf.layers.conv2d_transpose(
                    enc6, DNA_KERN_SIZE ** 2, (1, 1), activation=tf.nn.relu, name='convt4', reuse=reuse)
            else:
                # Using largest hidden state for predicting a new image layer.
                enc7 = tf.layers.conv2d_transpose(
                    enc6, color_channels, (1, 1), activation=tf.nn.relu, name='convt4', reuse=reuse)
                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                transformed = [tf.nn.sigmoid(enc7)]

            if stp:
                raise NotImplementedError
                stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
                stp_input1 = tf.layers.dense(
                    stp_input0, 100, name='fc_stp', reuse=reuse)
                transformed += stp_transformation(prev_image, stp_input1, num_masks)
            elif cdna:
                cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
                assert len(transformed_images) == num_masks - 1
                transformed += cdna_transformations(transformed_images, cdna_input, num_masks - 1,
                                                    int(color_channels))
            elif dna:
                raise NotImplementedError
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7)]

            transformed.insert(0, prev_image)

            masks = x0tm
            if dependent_mask:
                masks = tf.concat([masks] + transformed, axis=3)
            # TODO: no relu before softmax?
            masks = tf.layers.conv2d_transpose(masks, num_masks + 1, (3, 3), (1, 1), padding='same', activation=tf.nn.relu, name='masks_deconv6', reuse=reuse)
            masks = tf.reshape(
                tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
                [int(batch_size), int(img_height), int(img_width), num_masks + 1])
            mask_list = tf.split(masks, num_masks + 1, axis=3)

            # composition
            output = 0
            for layer, mask in zip(transformed, mask_list):
                output += layer * mask
            gen_images.append(output)
            gen_masks.append(mask_list)

            current_state = tf.layers.dense(state_action, int(current_state.get_shape()[1]), activation=None, name='state_pred', reuse=reuse)
            gen_states.append(current_state)

            gen_transformed_images.append(transformed)

            preactivation_feature_maps.append(preactivation_xlevels)
            feature_maps.append(xlevels)

    return gen_images, gen_masks, gen_states, gen_transformed_images, preactivation_feature_maps, feature_maps


def create_cdna_mask_generator(images, masks, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                               preprocess_image=None, stp=False, cdna=True, dna=False, **kwargs):
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib.layers.python import layers as tf_layers
    from transformer.spatial_transformer import transformer
    from prediction_model import cdna_transformation, dna_transformation, stp_transformation
    if stp + cdna + dna != 1:
        raise ValueError('More than one, or no network option specified.')

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)
    num_masks = int(masks[0].get_shape()[-1])
    from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
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
    for image, mask, state, action in zip(images[:-1], masks[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                # same base_mask and params as before
                prev_image = gen_images[-1]
                prev_mask = gen_masks[-1]
            elif done_warm_start:
                prev_image, prev_mask = scheduled_samples([image, mask], [gen_images[-1], gen_masks[-1]], batch_size, num_ground_truth)
            else:
                prev_image = image
                prev_mask = mask

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])

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

            if dna:
                # Using largest hidden state for predicting untied conv kernels.
                DNA_KERN_SIZE = 5
                enc7 = slim.layers.conv2d_transpose(
                    enc6, DNA_KERN_SIZE ** 2, 1, stride=1, activation_fn=None, scope='convt4')
            else:
                # Using largest hidden state for predicting a new image layer.
                enc7 = slim.layers.conv2d_transpose(
                    enc6, color_channels, 1, stride=1, activation_fn=None, scope='convt4')
                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                transformed = [tf.nn.sigmoid(enc7)]

            if stp:
                stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
                stp_input1 = slim.layers.fully_connected(
                    stp_input0, 100, scope='fc_stp')
                transformed += stp_transformation(prev_image, stp_input1, num_masks)
            elif cdna:
                cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
                # transformed += cdna_transformation(prev_image, cdna_input, num_masks,
                #                                    int(color_channels))
            elif dna:
                # Only one mask is supported (more should be unnecessary).
                if num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [dna_transformation(prev_image, enc7)]

            assert not preprocess_image

            RELU_SHIFT = 1e-12
            DNA_KERN_SIZE = 5
            # Predict kernels using linear function of last hidden layer.
            cdna_kerns = slim.layers.fully_connected(
                cdna_input,
                DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
                scope='cdna_params',
                activation_fn=None)

            # Reshape and normalize.
            cdna_kerns = tf.reshape(
                cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
            cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
            norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
            cdna_kerns /= norm_factor

            # TODO: normalize with softmax?
            # TODO: different transform fordifferent channels in image. separable convolution?

            tiled_cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, int(color_channels), 1])
            tiled_cdna_kerns = tf.split(axis=0, num_or_size_splits=batch_size, value=tiled_cdna_kerns)
            cdna_kerns = tf.split(axis=0, num_or_size_splits=batch_size, value=cdna_kerns)
            prev_images = tf.split(axis=0, num_or_size_splits=batch_size, value=prev_image)
            prev_masks = tf.split(prev_mask, batch_size, axis=0)

            # Transform image.
            transformed_masked_images = []
            transformed_masks = []
            for kernel, tiled_kernel, prev_image, prev_mask in zip(cdna_kerns, tiled_cdna_kerns, prev_images, prev_masks):
                kernel = tf.squeeze(kernel, axis=0)
                tiled_kernel = tf.squeeze(tiled_kernel, axis=0)
                # mask-dependent normalization factors
                # normalization_factors_ = []
                # for i in range(num_masks):
                #     normalization_factor = tf.nn.depthwise_conv2d(prev_mask[..., i:i+1], tf.ones_like(kernel[..., i:i+1]), [1, 1, 1, 1], 'SAME')
                #     normalization_factors_.append(normalization_factor / (DNA_KERN_SIZE * DNA_KERN_SIZE) + RELU_SHIFT)
                transformed_masked_images_ = []
                for i in range(num_masks):
                    transformed_masked_image = tf.nn.depthwise_conv2d(prev_mask[..., i:i+1] * prev_image, tiled_kernel[..., i:i+1], [1, 1, 1, 1], 'SAME')
                    # transformed_masked_image /= normalization_factors_[i]
                    transformed_masked_images_.append(transformed_masked_image)
                transformed_masks_ = []
                for i in range(num_masks):
                    transformed_mask = tf.nn.depthwise_conv2d(prev_mask[..., i:i+1], kernel[..., i:i+1], [1, 1, 1, 1], 'SAME')
                    # transformed_mask /= normalization_factors_[i]
                    transformed_masks_.append(transformed_mask)
                # normalization_factors_ = []
                # for i in range(num_masks):
                #     normalization_factor = tf.reduce_sum(prev_mask[..., i:i+1], [1, 2, 3], keep_dims=True) /\
                #                            (tf.reduce_sum(transformed_masks_[i], [1, 2, 3], keep_dims=True) + RELU_SHIFT)
                #     normalization_factors_.append(normalization_factor)
                # for i in range(num_masks):
                #     transformed_masked_images_[i] *= normalization_factors_[i]
                #     transformed_masks_[i] *= normalization_factors_[i]
                transformed_masked_images.append(tf.concat(transformed_masked_images_, axis=-1))
                transformed_masks.append(tf.concat(transformed_masks_, axis=-1))
            transformed_masked_images = tf.concat(axis=0, values=transformed_masked_images)
            transformed_masked_images = tf.split(axis=3, num_or_size_splits=num_masks, value=transformed_masked_images)
            transformed_masks = tf.concat(axis=0, values=transformed_masks)
            gen_masks.append(transformed_masks)
            transformed_masks = tf.split(axis=3, num_or_size_splits=num_masks, value=transformed_masks)

            gen_image = transformed[0]
            for i in range(num_masks):
                if i == 0:
                    continue
                gen_image = (1 - transformed_masks[i]) * gen_image + transformed_masks[i] * transformed_masked_images[i]
            gen_images.append(gen_image)

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)

    return gen_images, gen_masks, gen_states


def create_stp_mask_generator(images, masks, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                              preprocess_image=None, use_gt_mask=None, **kwargs):
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib.layers.python import layers as tf_layers
    from transformer.spatial_transformer import transformer

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)
    num_masks = int(masks[0].get_shape()[-1])
    from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    identity_params = tf.convert_to_tensor(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
    identity_params = tf.tile(identity_params[None, :, None], (batch_size, 1, num_masks))

    regularization_losses = []

    # LSTM state sizes and states.
    lstm_size = np.int32(np.array([16, 16, 32, 32, 64, 32, 16]))
    lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
    lstm_state5, lstm_state6, lstm_state7 = None, None, None

    # actual mask is unused
    for image, mask, state, action in zip(images[:-1], masks[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                # same base_mask and params as before
                prev_image = gen_images[-1]
            elif done_warm_start:
                # prev_image, base_image, base_mask, params = scheduled_samples([image, image, mask, identity_params],
                #                                                               [gen_images[-1], base_image, base_mask, params],
                #                                                               batch_size, num_ground_truth)
                # prev_image, base_image, base_action, base_current_state, params = \
                #     scheduled_samples([image, image, action, current_state, identity_params],
                #                       [gen_images[-1], base_image, base_action, base_current_state, params],
                #                       batch_size, num_ground_truth)
                prev_image, base_image, base_mask, base_action, base_current_state, params = \
                    scheduled_samples([image, image, mask, action, current_state, identity_params],
                                      [gen_images[-1], base_image, base_mask, base_action, base_current_state, params],
                                      batch_size, num_ground_truth)
            else:
                prev_image = image
                base_image = image
                base_mask = mask
                base_action = action
                base_current_state = current_state
                params = identity_params

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])
            base_state_action = tf.concat(axis=1, values=[base_action, base_current_state])

            if use_gt_mask:
                enc0 = slim.layers.conv2d(    #32x32x32
                    prev_image,
                    32, [5, 5],
                    stride=2,
                    scope='scale1_conv1',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm1'})
            else:
                enc0 = slim.layers.conv2d(    #32x32x32
                    tf.concat([prev_image, base_image], axis=0),
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
            if use_gt_mask:
                smear = tf.expand_dims(tf.expand_dims(state_action, 1), 1)
            else:
                smear = tf.expand_dims(tf.expand_dims(tf.concat([state_action, base_state_action], axis=0), 1), 1)
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

            # Using largest hidden state for predicting a new image layer.
            enc7 = slim.layers.conv2d_transpose(
                enc6[:batch_size], color_channels, 1, stride=1, activation_fn=None, scope='convt4')

            hidden5 = hidden5[:batch_size]

            if not use_gt_mask:
                base_mask = slim.layers.conv2d_transpose(
                    enc6[batch_size:], num_masks, 1, stride=1, activation_fn=None, scope='convtmask')
                base_mask = tf.nn.sigmoid(base_mask)

            assert not preprocess_image

            # This allows the network to also generate one image from scratch,
            # which is useful when regions of the image become unoccluded.
            gen_image = tf.nn.sigmoid(enc7)

            stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
            # stp_input1 = slim.layers.fully_connected(
            #     stp_input0, 100, scope='fc_stp')
            stp_input1 = slim.layers.fully_connected(
                stp_input0, 32, scope='fc_stp')
            stp_input1 = tf.concat([stp_input1, state_action], axis=-1)

            delta_params = []
            for i in range(num_masks):
                fc_stp2 = slim.layers.fully_connected(stp_input1, 2, scope='fc_stp2_%d' % i, activation_fn=None, reuse=reuse)
                delta_params.append(tf.concat([tf.zeros((batch_size, 2), dtype=np.float32), fc_stp2[:, 0:1],
                                               tf.zeros((batch_size, 2), dtype=np.float32), fc_stp2[:, 1:2]], axis=-1))
            delta_params = tf.stack(delta_params, axis=-1)
            # TODO: use general addition for arbitrary transformations
            params = params + delta_params

            regularization_losses.append(tf.reduce_mean(tf.square(delta_params)))

            gen_mask = []
            for i in range(num_masks):
                gen_mask.append(transformer(base_mask[..., i:i+1], params[..., i], base_mask.shape[1:3]))
            gen_mask = tf.reshape(tf.concat(gen_mask, axis=-1), base_mask.shape)
            gen_masks.append(gen_mask)

            # comp_fact_input = tf.nn.sigmoid(slim.layers.fully_connected(tf.reshape(hidden5, [batch_size, -1]),
            #                                                             num_masks, scope='fc_compfactors',
            #                                                             activation_fn=None))

            for i in range(num_masks):
                if i == 0:
                    continue
                fg_image = transformer(base_image * base_mask[..., i:i+1], params[..., i], base_mask.shape[1:3])
                gen_image = (1 - gen_mask[..., i:i+1]) * gen_image + gen_mask[..., i:i+1] * fg_image
            gen_images.append(gen_image)

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)

    return gen_images, gen_masks, gen_states, regularization_losses


def create_stp_mask_basefirst_generator(images, masks, states, actions, iter_num=None, schedule_sampling_k=None, context_frames=None,
                              preprocess_image=None, use_gt_mask=None, **kwargs):
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib.layers.python import layers as tf_layers
    from transformer.spatial_transformer import transformer

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)
    num_masks = int(masks[0].get_shape()[-1])
    from video_prediction.lstm_ops import basic_conv_lstm_cell as lstm_func

    # Generated robot states and images.
    gen_images = []
    gen_masks = []
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    base_mask = None
    identity_params = tf.convert_to_tensor(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
    identity_params = tf.tile(identity_params[None, :, None], (batch_size, 1, num_masks))
    params = identity_params

    regularization_losses = []

    # LSTM state sizes and states.
    lstm_size = np.int32(np.array([16, 16, 32, 32, 64, 32, 16]))
    lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
    lstm_state5, lstm_state6, lstm_state7 = None, None, None

    # actual mask is unused
    for image, mask, state, action in zip(images[:-1], masks[:-1], states[:-1], actions[:-1]):
        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                # same base_mask and params as before
                prev_image = gen_images[-1]
            elif done_warm_start:
                # prev_image, base_image, base_mask, params = scheduled_samples([image, image, mask, identity_params],
                #                                                               [gen_images[-1], base_image, base_mask, params],
                #                                                               batch_size, num_ground_truth)
                # prev_image, base_image, base_action, base_current_state, params = \
                #     scheduled_samples([image, image, action, current_state, identity_params],
                #                       [gen_images[-1], base_image, base_action, base_current_state, params],
                #                       batch_size, num_ground_truth)
                # prev_image, base_image, base_mask, base_action, base_current_state, params = \
                #     scheduled_samples([image, image, mask, action, current_state, identity_params],
                #                       [gen_images[-1], base_image, base_mask, base_action, base_current_state, params],
                #                       batch_size, num_ground_truth)
                from prediction_model import scheduled_sample
                prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
            else:
                prev_image = image
                # base_image = image
                # base_mask = mask
                # base_action = action
                # base_current_state = current_state
                # params = identity_params

            # Predicted state is always fed back in
            state_action = tf.concat(axis=1, values=[action, current_state])

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

            # Using largest hidden state for predicting a new image layer.
            enc7 = slim.layers.conv2d_transpose(
                enc6, color_channels, 1, stride=1, activation_fn=None, scope='convt4')

            if base_mask is None:
                base_image = images[0]
                if use_gt_mask:
                    base_mask = masks[0]
                else:
                    base_mask = slim.layers.conv2d_transpose(
                        enc6, num_masks, 1, stride=1, activation_fn=None, scope='convtmask')
                    base_mask = tf.nn.sigmoid(base_mask)

            assert not preprocess_image

            # This allows the network to also generate one image from scratch,
            # which is useful when regions of the image become unoccluded.
            gen_image = tf.nn.sigmoid(enc7)

            stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
            stp_input1 = slim.layers.fully_connected(
                stp_input0, 100, scope='fc_stp')

            delta_params = []
            for i in range(num_masks):
                fc_stp2 = slim.layers.fully_connected(stp_input1, 2, scope='fc_stp2_%d' % i, activation_fn=None, reuse=reuse)
                delta_params.append(tf.concat([tf.zeros((batch_size, 2), dtype=np.float32), fc_stp2[:, 0:1],
                                               tf.zeros((batch_size, 2), dtype=np.float32), fc_stp2[:, 1:2]], axis=-1))
            delta_params = tf.stack(delta_params, axis=-1)
            # TODO: use general addition for arbitrary transformations
            params = params + delta_params

            regularization_losses.append(tf.reduce_mean(tf.square(delta_params)))

            gen_mask = []
            for i in range(num_masks):
                gen_mask.append(transformer(base_mask[..., i:i+1], params[..., i], base_mask.shape[1:3]))
            gen_mask = tf.reshape(tf.concat(gen_mask, axis=-1), base_mask.shape)
            gen_masks.append(gen_mask)

            # comp_fact_input = tf.nn.sigmoid(slim.layers.fully_connected(tf.reshape(hidden5, [batch_size, -1]),
            #                                                             num_masks, scope='fc_compfactors',
            #                                                             activation_fn=None))

            for i in range(num_masks):
                if i == 0:
                    continue
                fg_image = transformer(base_image * base_mask[..., i:i+1], params[..., i], base_mask.shape[1:3])
                gen_image = (1 - gen_mask[..., i:i+1]) * gen_image + gen_mask[..., i:i+1] * fg_image
            gen_images.append(gen_image)

            current_state = slim.layers.fully_connected(
                state_action,
                int(current_state.get_shape()[1]),
                scope='state_pred',
                activation_fn=None)
            gen_states.append(current_state)

    return gen_images, gen_masks, gen_states, regularization_losses


def create_acp2p_generator(images, states, actions, ngf=None, iter_num=None, schedule_sampling_k=None, state_schedule_sampling_k=None,
                           context_frames=None, num_layers=None, num_lstm_layers=None, interleave_lstm=None,
                           use_kernel_initializer=None, use_nonlinear_dynamics=None, include_image_enc=None, **kwargs):
    if num_layers == 8:
        enc_layer_specs = [
            (ngf * 2, 1),     # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (ngf * 4, 2),     # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 8, 2),     # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 2),     # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (ngf * 8, 2),     # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (ngf * 8, 2),     # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            (ngf * 8, 2),     # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        dec_layer_specs = [
            (ngf * 8, 2, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 2, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 1, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    elif num_layers == 7:
        # NEW
        enc_layer_specs = [
            (ngf * 2, 2),     # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (ngf * 4, 2),     # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 8, 2),     # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 2),     # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (ngf * 8, 2),     # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (ngf * 8, 2),     # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]

        dec_layer_specs = [
            (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 2, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    else:
        raise NotImplementedError("num_layers=%d" % num_layers)

    state_action_fc_layer_specs = [
        256,
        256
    ]

    # NEW
    # num_fc_layers = 4
    # num_fc_layers = 2

    # things to try:
    # num_fc_layers = 4, without intermediate conv2d
    # less steep scheduled sampling
    # action-conditioned discriminator

    gen_images = []
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = images[0].get_shape()[0]
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    k = state_schedule_sampling_k
    if k == -1:
        state_feedself = True
    elif k == float('inf'):
        state_feedself = False
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = states[0].get_shape()[0]
        state_num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        state_feedself = False

    # from prediction_model import basic_conv_lstm_cell as lstm_func
    # NEW
    from prediction_model import layer_norm_basic_conv_lstm_cell as lstm_func
    # from RWACell import RWACell

    conv_lstm_states = [None] * ((len(enc_layer_specs) + 1) * num_lstm_layers)

    state_rnn_unit_dim = 256
    state_rnn_cells = []
    for k in range(num_lstm_layers):
        with tf.variable_scope('lstm_%d' % k):
            state_rnn_cells.append(tf.contrib.rnn.LayerNormBasicLSTMCell(state_rnn_unit_dim, reuse=tf.get_variable_scope().reuse))
        # with tf.variable_scope('rwa_%d' % k):  #, reuse=tf.get_variable_scope().reuse):
        #     state_rnn_cells.append(RWACell(state_rnn_unit_dim))
    state_rnn_stacked = tf.contrib.rnn.MultiRNNCell(state_rnn_cells)

    rnn_states = []
    for k, cell in enumerate(state_rnn_stacked._cells):
        with tf.variable_scope('rnn_zero_state_%d' % k):
            rnn_states.append(cell.zero_state(batch_size, tf.float32))

    # rnn_states = state_rnn_stacked.zero_state(batch_size, tf.float32)

    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        with tf.variable_scope("acp2p_generator_single_step", reuse=len(gen_images) > 0) as scope:
            done_warm_start = len(gen_images) > context_frames - 1
            if feedself and done_warm_start:
                # Feed in generated image.
                # feed in image_encs from previous next_image_encs
                image = None
                prev_image_encs = next_image_encs
            elif done_warm_start:
                # Scheduled sampling
                # split into current_image and image_encs from previous next_image_encs
                idx = tf.random_shuffle(tf.range(int(batch_size)))
                # image_idx = tf.gather(idx, tf.range(num_ground_truth))
                image_idx = tf.gather(idx, tf.range(tf.maximum(num_ground_truth, 1)))  # TODO: this is wasteful
                image_enc_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))
                image = tf.gather(image, image_idx)
                prev_image_encs = [tf.gather(next_image_enc, image_enc_idx) for next_image_enc in next_image_encs]
            else:
                # Always feed in ground_truth
                # feed in current_image
                prev_image_encs = None

            if state_feedself and done_warm_start:
                pass
                # feed in predicted state
            elif done_warm_start and state_schedule_sampling_k != float('inf'):
                # feed in a mix of predicted and ground truth state
                from prediction_model import scheduled_sample
                current_state = scheduled_sample(state, current_state, batch_size, state_num_ground_truth)
            else:
                # feed in ground truth state
                current_state = state

            if image is not None:
                image_encs = []
                with tf.variable_scope("encoder_1"):
                    image_enc = conv(image, ngf, stride=1)
                    image_encs.append(image_enc)

                for out_channels, stride in enc_layer_specs:
                    with tf.variable_scope("encoder_%d" % (len(image_encs) + 1)):
                        rectified = lrelu(image_encs[-1], 0.2)
                        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                        convolved = conv(rectified, out_channels, stride=stride)
                        image_enc = batchnorm(convolved)
                        image_encs.append(image_enc)

                if prev_image_encs is not None:
                    # combine image_enc derived from current image or derived from the previous next_image_enc
                    curr_image_encs = image_encs
                    image_encs = []
                    for prev_image_enc, curr_image_enc in zip(prev_image_encs, curr_image_encs):
                        # image_enc = tf.dynamic_stitch([image_idx, image_enc_idx], [curr_image_enc, prev_image_enc])
                        # image_enc = tf.reshape(image_enc, [int(batch_size)] + image_enc.shape.as_list()[1:])
                        image_enc = tf.cond(tf.equal(num_ground_truth, 0),
                                            lambda: prev_image_enc,
                                            lambda: tf.dynamic_stitch([image_idx, image_enc_idx], [curr_image_enc, prev_image_enc]))
                        # TODO: investigate error that happens when num_ground_truth == 0
                        # NEW
                        image_enc = tf.nn.relu(image_enc)
                        image_encs.append(image_enc)
            else:
                assert prev_image_encs is not None
                image_encs = prev_image_encs

            # Predicted state is always fed back in
            state_action = tf.concat([current_state, action], axis=-1)
            state_action_enc = state_action
            for k, unit_dim in enumerate(state_action_fc_layer_specs):
                with tf.variable_scope('state_action_fc_%d' % (k + 1)):
                    state_action_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)

            conv_lstm_state_k = 0

            next_image_encs = []
            for i, image_enc in enumerate(image_encs):
                unit_dim = int(image_enc.shape[-1])
                with tf.variable_scope('state_action_fc_last_%d' % (i + 1)):
                    state_action_last_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)

                tile_pattern = [1, int(image_enc.shape[1]), int(image_enc.shape[2]), 1]
                next_image_enc = tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(state_action_last_enc, 1), 1), tile_pattern), image_enc], axis=-1)
                with tf.variable_scope('conv_smear_%d' % (i + 1)):
                    kernel_initializer = tf.random_normal_initializer(0, 0.02) if use_kernel_initializer else None
                    next_image_enc = tf.layers.conv2d(next_image_enc, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu,
                                                      kernel_initializer=kernel_initializer)

                for k in range(num_lstm_layers):
                    with tf.variable_scope('conv_lstm_%d_%d' % (k + 1, i + 1)):
                        # hidden, rnn_states[lstm_state_k] = lstm_func(next_image_enc, rnn_states[lstm_state_k], unit_dim)
                        # NEW
                        hidden, conv_lstm_states[conv_lstm_state_k] = lstm_func(next_image_enc, conv_lstm_states[conv_lstm_state_k], unit_dim, filter_size=4,
                                                                                norm_fn=batchnorm)
                        conv_lstm_state_k += 1
                    # with tf.variable_scope('layer_norm_%d_%d' % (k + 1, i + 1)):
                    #     hidden = tf.contrib.layers.layer_norm(hidden)
                    if interleave_lstm:
                        with tf.variable_scope('fc_%d_%d' % (k + 1, i + 1)):
                            next_image_enc = tf.layers.conv2d(hidden, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu,
                                                              kernel_initializer=tf.random_normal_initializer(0, 0.02))
                    else:
                        next_image_enc = hidden
                next_image_encs.append(next_image_enc)

            num_encoder_layers = len(next_image_encs)
            next_image_decs = []
            for decoder_layer, (out_channels, stride, dropout) in enumerate(dec_layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = next_image_encs[-1]
                    else:
                        input = tf.concat([next_image_decs[-1], next_image_encs[skip_layer]], axis=3)

                    rectified = tf.nn.relu(input)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    output = deconv(rectified, out_channels, stride=stride)
                    output = batchnorm(output)

                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    next_image_decs.append(output)

            with tf.variable_scope("decoder_1"):
                input = tf.concat([next_image_decs[-1], next_image_encs[0]], axis=3)
                rectified = tf.nn.relu(input)
                output = deconv(rectified, 3, stride=1)
                output = tf.tanh(output)
                next_image_decs.append(output)

            gen_images.append(next_image_decs[-1])

            state_enc, rnn_states = state_rnn_stacked(state_action_enc, rnn_states)
            with tf.variable_scope('state_fc'):
                current_state = tf.layers.dense(state_enc, int(states[0].get_shape()[1]), activation=None)
            gen_states.append(current_state)

            # state_enc = state_action
            # if include_image_enc:
            #     pooled_image_encs = []
            #     for i, next_image_enc in enumerate(next_image_encs):
            #         filters = max(next_image_enc.shape[-1], ngf)
            #         pooled_image_enc = next_image_enc
            #         for k in range(2):
            #             with tf.variable_scope('pooled_image_enc_%d_%d' % (k + 1, i + 1)):
            #                 pooled_image_enc = tf.layers.conv2d(pooled_image_enc, filters, (3, 3), strides=1, padding='same',
            #                                                     activation=tf.nn.relu,
            #                                                     kernel_initializer=tf.random_normal_initializer(0, 0.02))
            #             if pooled_image_enc.shape[1] != 1 and pooled_image_enc.shape[2] != 1:
            #                 pooled_image_enc = tf.layers.max_pooling2d(pooled_image_enc, 2, strides=2, padding='same')
            #         pooled_image_enc = tf.reduce_max(pooled_image_enc, axis=(1, 2))
            #         pooled_image_encs.append(pooled_image_enc)
            #     state_enc = tf.concat([state_enc] + pooled_image_encs, axis=-1)
            # if use_nonlinear_dynamics:
            #     for k, unit_dim in enumerate(state_action_fc_layer_specs):
            #         with tf.variable_scope('state_fc_%d' % (k + 1)):
            #             state_enc = tf.layers.dense(state_enc, unit_dim, activation=tf.nn.relu)
            # with tf.variable_scope('state_fc'):
            #     current_state = tf.layers.dense(state_enc, int(states[0].get_shape()[1]), activation=None)
            # gen_states.append(current_state)
    return gen_images, None, gen_states


def create_acp2pln_generator(images, states, actions, ngf=None, iter_num=None, schedule_sampling_k=None, state_schedule_sampling_k=None,
                             context_frames=None, num_layers=None, num_lstm_layers=None, interleave_lstm=None,
                             use_kernel_initializer=None, use_nonlinear_dynamics=None, include_image_enc=None, **kwargs):
    if num_layers == 8:
        enc_layer_specs = [
            (ngf * 2, 1),     # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (ngf * 4, 2),     # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 8, 2),     # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 2),     # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (ngf * 8, 2),     # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (ngf * 8, 2),     # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            (ngf * 8, 2),     # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        dec_layer_specs = [
            (ngf * 8, 2, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 2, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 1, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    elif num_layers == 7:
        # NEW
        enc_layer_specs = [
            (ngf * 2, 2),     # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (ngf * 4, 2),     # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 8, 2),     # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 2),     # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (ngf * 8, 2),     # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (ngf * 8, 2),     # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]

        dec_layer_specs = [
            (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 2, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    else:
        raise NotImplementedError("num_layers=%d" % num_layers)

    state_action_fc_layer_specs = [
        256,
        256
    ]

    # NEW
    # num_fc_layers = 4
    # num_fc_layers = 2

    # things to try:
    # num_fc_layers = 4, without intermediate conv2d
    # less steep scheduled sampling
    # action-conditioned discriminator

    gen_images = []
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = images[0].get_shape()[0]
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    k = state_schedule_sampling_k
    if k == -1:
        state_feedself = True
    elif k == float('inf'):
        state_feedself = False
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = states[0].get_shape()[0]
        state_num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        state_feedself = False

    # from prediction_model import basic_conv_lstm_cell as lstm_func
    # NEW
    from prediction_model import layer_norm_basic_conv_lstm_cell as lstm_func

    conv_lstm_states = [None] * ((len(enc_layer_specs) + 1) * num_lstm_layers)

    lstm_unit_dim = 256
    lstm_cells = []
    for k in range(num_lstm_layers):
        with tf.variable_scope('lstm_%d' % k):
            lstm_cells.append(tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_unit_dim, reuse=tf.get_variable_scope().reuse))
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    lstm_states = stacked_lstm.zero_state(batch_size, tf.float32)

    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        with tf.variable_scope("acp2p_generator_single_step", reuse=len(gen_images) > 0) as scope:
            done_warm_start = len(gen_images) > context_frames - 1
            if feedself and done_warm_start:
                # Feed in generated image.
                # feed in image_encs from previous next_image_encs
                image = None
                prev_image_encs = next_image_encs
            elif done_warm_start:
                # Scheduled sampling
                # split into current_image and image_encs from previous next_image_encs
                idx = tf.random_shuffle(tf.range(int(batch_size)))
                # image_idx = tf.gather(idx, tf.range(num_ground_truth))
                image_idx = tf.gather(idx, tf.range(tf.maximum(num_ground_truth, 1)))  # TODO: this is wasteful
                image_enc_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))
                image = tf.gather(image, image_idx)
                prev_image_encs = [tf.gather(next_image_enc, image_enc_idx) for next_image_enc in next_image_encs]
            else:
                # Always feed in ground_truth
                # feed in current_image
                prev_image_encs = None

            if state_feedself and done_warm_start:
                pass
                # feed in predicted state
            elif done_warm_start and state_schedule_sampling_k != float('inf'):
                # feed in a mix of predicted and ground truth state
                from prediction_model import scheduled_sample
                current_state = scheduled_sample(state, current_state, batch_size, state_num_ground_truth)
            else:
                # feed in ground truth state
                current_state = state

            if image is not None:
                image_encs = []
                with tf.variable_scope("encoder_1"):
                    image_enc = conv(image, ngf, stride=1)
                    image_encs.append(image_enc)

                for out_channels, stride in enc_layer_specs:
                    with tf.variable_scope("encoder_%d" % (len(image_encs) + 1)):
                        rectified = lrelu(image_encs[-1], 0.2)
                        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                        convolved = conv(rectified, out_channels, stride=stride)
                        image_enc = layernorm(convolved)
                        image_encs.append(image_enc)

                if prev_image_encs is not None:
                    # combine image_enc derived from current image or derived from the previous next_image_enc
                    curr_image_encs = image_encs
                    image_encs = []
                    for prev_image_enc, curr_image_enc in zip(prev_image_encs, curr_image_encs):
                        # image_enc = tf.dynamic_stitch([image_idx, image_enc_idx], [curr_image_enc, prev_image_enc])
                        # image_enc = tf.reshape(image_enc, [int(batch_size)] + image_enc.shape.as_list()[1:])
                        image_enc = tf.cond(tf.equal(num_ground_truth, 0),
                                            lambda: prev_image_enc,
                                            lambda: tf.dynamic_stitch([image_idx, image_enc_idx], [curr_image_enc, prev_image_enc]))
                        # TODO: investigate error that happens when num_ground_truth == 0
                        # NEW
                        image_enc = tf.nn.relu(image_enc)
                        image_encs.append(image_enc)
            else:
                assert prev_image_encs is not None
                image_encs = prev_image_encs

            # Predicted state is always fed back in
            state_action = tf.concat([current_state, action], axis=-1)
            state_action_enc = state_action
            for k, unit_dim in enumerate(state_action_fc_layer_specs):
                with tf.variable_scope('state_action_fc_%d' % (k + 1)):
                    state_action_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)

            conv_lstm_state_k = 0

            next_image_encs = []
            for i, image_enc in enumerate(image_encs):
                unit_dim = int(image_enc.shape[-1])
                with tf.variable_scope('state_action_fc_last_%d' % (i + 1)):
                    state_action_last_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)

                tile_pattern = [1, int(image_enc.shape[1]), int(image_enc.shape[2]), 1]
                next_image_enc = tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(state_action_last_enc, 1), 1), tile_pattern), image_enc], axis=-1)
                with tf.variable_scope('conv_smear_%d' % (i + 1)):
                    kernel_initializer = tf.random_normal_initializer(0, 0.02) if use_kernel_initializer else None
                    next_image_enc = tf.layers.conv2d(next_image_enc, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu,
                                                      kernel_initializer=kernel_initializer)

                for k in range(num_lstm_layers):
                    with tf.variable_scope('conv_lstm_%d_%d' % (k + 1, i + 1)):
                        # hidden, lstm_states[lstm_state_k] = lstm_func(next_image_enc, lstm_states[lstm_state_k], unit_dim)
                        # NEW
                        hidden, conv_lstm_states[conv_lstm_state_k] = lstm_func(next_image_enc, conv_lstm_states[conv_lstm_state_k], unit_dim, filter_size=4)
                        conv_lstm_state_k += 1
                    # with tf.variable_scope('layer_norm_%d_%d' % (k + 1, i + 1)):
                    #     hidden = tf.contrib.layers.layer_norm(hidden)
                    if interleave_lstm:
                        with tf.variable_scope('fc_%d_%d' % (k + 1, i + 1)):
                            next_image_enc = tf.layers.conv2d(hidden, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu,
                                                              kernel_initializer=tf.random_normal_initializer(0, 0.02))
                    else:
                        next_image_enc = hidden
                next_image_encs.append(next_image_enc)

            num_encoder_layers = len(next_image_encs)
            next_image_decs = []
            for decoder_layer, (out_channels, stride, dropout) in enumerate(dec_layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = next_image_encs[-1]
                    else:
                        input = tf.concat([next_image_decs[-1], next_image_encs[skip_layer]], axis=3)

                    rectified = tf.nn.relu(input)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    output = deconv(rectified, out_channels, stride=stride)
                    output = layernorm(output)

                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    next_image_decs.append(output)

            with tf.variable_scope("decoder_1"):
                input = tf.concat([next_image_decs[-1], next_image_encs[0]], axis=3)
                rectified = tf.nn.relu(input)
                output = deconv(rectified, 3, stride=1)
                output = tf.tanh(output)
                next_image_decs.append(output)

            gen_images.append(next_image_decs[-1])

            state_enc, lstm_states = stacked_lstm(state_action_enc, lstm_states)
            with tf.variable_scope('state_fc'):
                current_state = tf.layers.dense(state_enc, int(states[0].get_shape()[1]), activation=None)
            gen_states.append(current_state)

            # state_enc = state_action
            # if include_image_enc:
            #     pooled_image_encs = []
            #     for i, next_image_enc in enumerate(next_image_encs):
            #         filters = max(next_image_enc.shape[-1], ngf)
            #         pooled_image_enc = next_image_enc
            #         for k in range(2):
            #             with tf.variable_scope('pooled_image_enc_%d_%d' % (k + 1, i + 1)):
            #                 pooled_image_enc = tf.layers.conv2d(pooled_image_enc, filters, (3, 3), strides=1, padding='same',
            #                                                     activation=tf.nn.relu,
            #                                                     kernel_initializer=tf.random_normal_initializer(0, 0.02))
            #             if pooled_image_enc.shape[1] != 1 and pooled_image_enc.shape[2] != 1:
            #                 pooled_image_enc = tf.layers.max_pooling2d(pooled_image_enc, 2, strides=2, padding='same')
            #         pooled_image_enc = tf.reduce_max(pooled_image_enc, axis=(1, 2))
            #         pooled_image_encs.append(pooled_image_enc)
            #     state_enc = tf.concat([state_enc] + pooled_image_encs, axis=-1)
            # if use_nonlinear_dynamics:
            #     for k, unit_dim in enumerate(state_action_fc_layer_specs):
            #         with tf.variable_scope('state_fc_%d' % (k + 1)):
            #             state_enc = tf.layers.dense(state_enc, unit_dim, activation=tf.nn.relu)
            # with tf.variable_scope('state_fc'):
            #     current_state = tf.layers.dense(state_enc, int(states[0].get_shape()[1]), activation=None)
            # gen_states.append(current_state)
    return gen_images, None, gen_states


def create_acp2pwn_generator(images, states, actions, ngf=None, iter_num=None, schedule_sampling_k=None, state_schedule_sampling_k=None,
                             context_frames=None, num_layers=None, num_lstm_layers=None, interleave_lstm=None,
                             use_kernel_initializer=None, use_nonlinear_dynamics=None, include_image_enc=None, **kwargs):
    if num_layers == 8:
        enc_layer_specs = [
            (ngf * 2, 1),     # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (ngf * 4, 2),     # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 8, 2),     # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 2),     # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (ngf * 8, 2),     # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (ngf * 8, 2),     # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            (ngf * 8, 2),     # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        dec_layer_specs = [
            (ngf * 8, 2, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 2, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 1, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    elif num_layers == 7:
        # NEW
        enc_layer_specs = [
            (ngf * 2, 2),     # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (ngf * 4, 2),     # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 8, 2),     # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 2),     # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (ngf * 8, 2),     # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (ngf * 8, 2),     # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]

        dec_layer_specs = [
            (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 2, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
    else:
        raise NotImplementedError("num_layers=%d" % num_layers)

    state_action_fc_layer_specs = [
        256,
        256
    ]

    # NEW
    # num_fc_layers = 4
    # num_fc_layers = 2

    # things to try:
    # num_fc_layers = 4, without intermediate conv2d
    # less steep scheduled sampling
    # action-conditioned discriminator

    gen_images = []
    gen_states = []
    current_state = states[0]

    k = schedule_sampling_k
    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = images[0].get_shape()[0]
        num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        feedself = False

    k = state_schedule_sampling_k
    if k == -1:
        state_feedself = True
    elif k == float('inf'):
        state_feedself = False
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        batch_size = states[0].get_shape()[0]
        state_num_ground_truth = tf.to_int32(tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(tf.to_float(iter_num) / k)))))
        state_feedself = False

    # from prediction_model import basic_conv_lstm_cell as lstm_func
    # NEW
    from prediction_model import layer_norm_basic_conv_lstm_cell as lstm_func

    conv_lstm_states = [None] * ((len(enc_layer_specs) + 1) * num_lstm_layers)

    lstm_unit_dim = 256
    lstm_cells = []
    for k in range(num_lstm_layers):
        with tf.variable_scope('lstm_%d' % k):
            lstm_cells.append(tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_unit_dim, reuse=tf.get_variable_scope().reuse))
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    lstm_states = stacked_lstm.zero_state(batch_size, tf.float32)

    for image, state, action in zip(images[:-1], states[:-1], actions[:-1]):
        with tf.variable_scope("acp2p_generator_single_step", reuse=len(gen_images) > 0) as scope:
            done_warm_start = len(gen_images) > context_frames - 1
            if feedself and done_warm_start:
                # Feed in generated image.
                # feed in image_encs from previous next_image_encs
                image = None
                prev_image_encs = next_image_encs
            elif done_warm_start:
                # Scheduled sampling
                # split into current_image and image_encs from previous next_image_encs
                idx = tf.random_shuffle(tf.range(int(batch_size)))
                # image_idx = tf.gather(idx, tf.range(num_ground_truth))
                image_idx = tf.gather(idx, tf.range(tf.maximum(num_ground_truth, 1)))  # TODO: this is wasteful
                image_enc_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))
                image = tf.gather(image, image_idx)
                prev_image_encs = [tf.gather(next_image_enc, image_enc_idx) for next_image_enc in next_image_encs]
            else:
                # Always feed in ground_truth
                # feed in current_image
                prev_image_encs = None

            if state_feedself and done_warm_start:
                pass
                # feed in predicted state
            elif done_warm_start and state_schedule_sampling_k != float('inf'):
                # feed in a mix of predicted and ground truth state
                from prediction_model import scheduled_sample
                current_state = scheduled_sample(state, current_state, batch_size, state_num_ground_truth)
            else:
                # feed in ground truth state
                current_state = state

            if image is not None:
                image_encs = []
                with tf.variable_scope("encoder_1"):
                    image_enc = conv2d(image, ngf, (4, 4), 1)
                    image_encs.append(image_enc)

                for out_channels, stride in enc_layer_specs:
                    with tf.variable_scope("encoder_%d" % len(image_encs)):
                        rectified = tprelu(image_encs[-1], 0.2, shared_axes=(1, 2))
                    with tf.variable_scope("encoder_%d" % (len(image_encs) + 1)):
                        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                        image_enc = conv2d(rectified, out_channels, (4, 4), stride)
                        image_encs.append(image_enc)

                if prev_image_encs is not None:
                    # combine image_enc derived from current image or derived from the previous next_image_enc
                    curr_image_encs = image_encs
                    image_encs = []
                    for prev_image_enc, curr_image_enc in zip(prev_image_encs, curr_image_encs):
                        # image_enc = tf.dynamic_stitch([image_idx, image_enc_idx], [curr_image_enc, prev_image_enc])
                        # image_enc = tf.reshape(image_enc, [int(batch_size)] + image_enc.shape.as_list()[1:])
                        image_enc = tf.cond(tf.equal(num_ground_truth, 0),
                                            lambda: prev_image_enc,
                                            lambda: tf.dynamic_stitch([image_idx, image_enc_idx], [curr_image_enc, prev_image_enc]))
                        # TODO: investigate error that happens when num_ground_truth == 0
                        # TODO: how to use parametric relu in here?
                        # with tf.variable_scope("combined_encoder_%d" % (len(image_encs) + 1)):
                        #     image_enc = tprelu(image_enc, 0.2, shared_axes=(1, 2))
                        image_encs.append(image_enc)
            else:
                assert prev_image_encs is not None
                image_encs = prev_image_encs

            # Predicted state is always fed back in
            state_action = tf.concat([current_state, action], axis=-1)
            state_action_enc = state_action
            for k, unit_dim in enumerate(state_action_fc_layer_specs):
                with tf.variable_scope('state_action_fc_%d' % (k + 1)):
                    state_action_enc = dense(state_action_enc, unit_dim)
                    state_action_enc = tprelu(state_action_enc, 0.2)

            conv_lstm_state_k = 0

            next_image_encs = []
            for i, image_enc in enumerate(image_encs):
                unit_dim = int(image_enc.shape[-1])
                with tf.variable_scope('state_action_fc_last_%d' % (i + 1)):
                    state_action_last_enc = dense(state_action_enc, unit_dim)
                    state_action_last_enc = tprelu(state_action_last_enc, 0.2)

                tile_pattern = [1, int(image_enc.shape[1]), int(image_enc.shape[2]), 1]
                next_image_enc = tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(state_action_last_enc, 1), 1), tile_pattern), image_enc], axis=-1)
                with tf.variable_scope('conv_smear_%d' % (i + 1)):
                    next_image_enc = conv2d(next_image_enc, unit_dim, (1, 1), 1)
                    next_image_enc = tprelu(next_image_enc, 0.2, shared_axes=(1, 2))

                for k in range(num_lstm_layers):
                    with tf.variable_scope('conv_lstm_%d_%d' % (k + 1, i + 1)):
                        # hidden, lstm_states[lstm_state_k] = lstm_func(next_image_enc, lstm_states[lstm_state_k], unit_dim)
                        # NEW
                        hidden, conv_lstm_states[conv_lstm_state_k] = lstm_func(next_image_enc, conv_lstm_states[conv_lstm_state_k], unit_dim, filter_size=4)
                        conv_lstm_state_k += 1
                    # with tf.variable_scope('layer_norm_%d_%d' % (k + 1, i + 1)):
                    #     hidden = tf.contrib.layers.layer_norm(hidden)
                    if interleave_lstm:
                        with tf.variable_scope('fc_%d_%d' % (k + 1, i + 1)):
                            next_image_enc = conv2d(hidden, unit_dim, (1, 1), 1)
                            next_image_enc = tprelu(next_image_enc, 0.2, shared_axes=(1, 2))
                    else:
                        next_image_enc = hidden
                next_image_encs.append(next_image_enc)

            num_encoder_layers = len(next_image_encs)
            next_image_decs = []
            for decoder_layer, (out_channels, stride, dropout) in enumerate(dec_layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = next_image_encs[-1]
                    else:
                        input = tf.concat([next_image_decs[-1], next_image_encs[skip_layer]], axis=3)

                    rectified = tprelu(input, 0.2, shared_axes=(1, 2))
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    output = deconv2d(rectified, out_channels, (4, 4), stride)

                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    next_image_decs.append(output)

            with tf.variable_scope("decoder_1"):
                input = tf.concat([next_image_decs[-1], next_image_encs[0]], axis=3)
                rectified = tprelu(input, 0.2, shared_axes=(1, 2))
                output = deconv2d(rectified, 3, (4, 4), 1)
                output = affine(output)
                output = tf.tanh(output)
                next_image_decs.append(output)

            gen_images.append(next_image_decs[-1])

            state_enc, lstm_states = stacked_lstm(state_action_enc, lstm_states)
            with tf.variable_scope('state_fc'):
                current_state = dense(state_enc, int(states[0].get_shape()[1]))
                current_state = affine(current_state)
            gen_states.append(current_state)

            # state_enc = state_action
            # if include_image_enc:
            #     pooled_image_encs = []
            #     for i, next_image_enc in enumerate(next_image_encs):
            #         filters = max(next_image_enc.shape[-1], ngf)
            #         pooled_image_enc = next_image_enc
            #         for k in range(2):
            #             with tf.variable_scope('pooled_image_enc_%d_%d' % (k + 1, i + 1)):
            #                 pooled_image_enc = tf.layers.conv2d(pooled_image_enc, filters, (3, 3), strides=1, padding='same',
            #                                                     activation=tf.nn.relu,
            #                                                     kernel_initializer=tf.random_normal_initializer(0, 0.02))
            #             if pooled_image_enc.shape[1] != 1 and pooled_image_enc.shape[2] != 1:
            #                 pooled_image_enc = tf.layers.max_pooling2d(pooled_image_enc, 2, strides=2, padding='same')
            #         pooled_image_enc = tf.reduce_max(pooled_image_enc, axis=(1, 2))
            #         pooled_image_encs.append(pooled_image_enc)
            #     state_enc = tf.concat([state_enc] + pooled_image_encs, axis=-1)
            # if use_nonlinear_dynamics:
            #     for k, unit_dim in enumerate(state_action_fc_layer_specs):
            #         with tf.variable_scope('state_fc_%d' % (k + 1)):
            #             state_enc = tf.layers.dense(state_enc, unit_dim, activation=tf.nn.relu)
            # with tf.variable_scope('state_fc'):
            #     current_state = tf.layers.dense(state_enc, int(states[0].get_shape()[1]), activation=None)
            # gen_states.append(current_state)

    return gen_images, None, gen_states


def create_acp2p_discriminator(image, state, action, gen_image, ndf=None, **kwargs):
    n_layers = 3
    layers = []

    state_action_fc_layer_specs = [
        256,
        256
    ]

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    image_pair = tf.concat([image, gen_image], axis=3)

    state_action = tf.concat([state, action], axis=-1)
    state_action_enc = state_action
    for k, unit_dim in enumerate(state_action_fc_layer_specs):
        with tf.variable_scope('state_action_fc_%d' % (k + 1)):
            state_action_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(image_pair, ndf, stride=1)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            # stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            stride = 1 if i >= n_layers - 2 else 2  # last two layers here have stride 1  # TODO: stride

            enc = layers[-1]
            unit_dim = int(enc.shape[-1])
            with tf.variable_scope('state_action_fc_last'):
                state_action_last_enc = tf.layers.dense(state_action_enc, unit_dim, activation=tf.nn.relu)
            tile_pattern = [1, int(enc.shape[1]), int(enc.shape[2]), 1]
            enc = tf.concat([tf.tile(tf.expand_dims(tf.expand_dims(state_action_last_enc, 1), 1), tile_pattern), enc], axis=-1)
            # enc = tf.layers.conv2d(enc, unit_dim, (1, 1), strides=1, padding='same', activation=tf.nn.relu,  # activation?
            #                        kernel_initializer=tf.random_normal_initializer(0, 0.02))

            convolved = conv(enc, out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


# def create_generator(generator_inputs, generator_outputs_channels, ngf=None, **kwargs):
#     layers = []
#
#     # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
#     with tf.variable_scope("encoder_1"):
#         output = conv(generator_inputs, ngf, stride=1)
#         layers.append(output)
#
#     layer_specs = [
#         (ngf * 2, 1),  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
#         (ngf * 4, 2),  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
#         (ngf * 8, 2),  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
#         (ngf * 8, 2),  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
#         (ngf * 8, 2),  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
#         (ngf * 8, 2),  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
#         (ngf * 8, 2),  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
#     ]
#
#     for out_channels, stride in layer_specs:
#         with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
#             rectified = lrelu(layers[-1], 0.2)
#             # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
#             convolved = conv(rectified, out_channels, stride=stride)
#             output = batchnorm(convolved)
#             layers.append(output)
#
#     layer_specs = [
#         (ngf * 8, 2, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
#         (ngf * 8, 2, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
#         (ngf * 8, 2, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
#         (ngf * 8, 2, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
#         (ngf * 4, 2, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
#         (ngf * 2, 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
#         (ngf, 1, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
#     ]
#
#     num_encoder_layers = len(layers)
#     for decoder_layer, (out_channels, stride, dropout) in enumerate(layer_specs):
#         skip_layer = num_encoder_layers - decoder_layer - 1
#         with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
#             if decoder_layer == 0:
#                 # first decoder layer doesn't have skip connections
#                 # since it is directly connected to the skip_layer
#                 input = layers[-1]
#             else:
#                 input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
#
#             rectified = tf.nn.relu(input)
#             # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
#             output = deconv(rectified, out_channels, stride=stride)
#             output = batchnorm(output)
#
#             if dropout > 0.0:
#                 output = tf.nn.dropout(output, keep_prob=1 - dropout)
#
#             layers.append(output)
#
#     # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
#     with tf.variable_scope("decoder_1"):
#         input = tf.concat([layers[-1], layers[0]], axis=3)
#         rectified = tf.nn.relu(input)
#         output = deconv(rectified, generator_outputs_channels, stride=1)
#         output = tf.tanh(output)
#         layers.append(output)
#
#     return layers[-1]


def create_discriminator(discrim_inputs, discrim_targets, ndf=None, **kwargs):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # TODO: use stride depending on the original image size
    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=1)  # TODO: stride
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            # stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            stride = 1 if i >= n_layers - 2 else 2  # last two layers here have stride 1  # TODO: stride
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


# def create_model_orig(inputs, targets, gan_weight=None, l1_weight=None, lr=None, beta1=None, **kwargs):
#     with tf.variable_scope("generator") as scope:
#         out_channels = int(targets.get_shape()[-1])
#         outputs = create_generator(inputs, out_channels, **kwargs)
#
#     # create two copies of discriminator, one for real pairs and one for fake pairs
#     # they share the same underlying variables
#     with tf.name_scope("real_discriminator"):
#         with tf.variable_scope("discriminator"):
#             # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#             predict_real = create_discriminator(inputs, targets, **kwargs)
#
#     with tf.name_scope("fake_discriminator"):
#         with tf.variable_scope("discriminator", reuse=True):
#             # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#             predict_fake = create_discriminator(inputs, outputs, **kwargs)
#
#     with tf.name_scope("discriminator_loss"):
#         # minimizing -tf.log will try to get inputs to 1
#         # predict_real => 1
#         # predict_fake => 0
#         discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
#
#     with tf.name_scope("generator_loss"):
#         # predict_fake => 1
#         # abs(targets - outputs) => 0
#         gen_GAN_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
#         gen_L1_loss = tf.reduce_mean(tf.abs(targets - outputs))
#         gen_loss = gen_GAN_loss * gan_weight + gen_L1_loss * l1_weight
#
#     with tf.name_scope("discriminator_train"):
#         discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
#         discrim_optim = tf.train.AdamOptimizer(lr, beta1)
#         discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
#         discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
#
#     with tf.name_scope("generator_train"):
#         with tf.control_dependencies([discrim_train]):
#             gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#             gen_optim = tf.train.AdamOptimizer(lr, beta1)
#             gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
#             gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
#
#     ema = tf.train.ExponentialMovingAverage(decay=0.99)
#     update_losses = ema.apply([discrim_loss, gen_GAN_loss, gen_L1_loss])
#
#     global_step = tf.contrib.framework.get_or_create_global_step()
#     incr_global_step = tf.assign(global_step, global_step+1)
#
#     return Model(
#         predict_real=predict_real,
#         predict_fake=predict_fake,
#         discrim_loss=ema.average(discrim_loss),
#         discrim_grads_and_vars=discrim_grads_and_vars,
#         gen_GAN_loss=ema.average(gen_GAN_loss),
#         gen_L1_loss=ema.average(gen_L1_loss),
#         gen_grads_and_vars=gen_grads_and_vars,
#         outputs=outputs,
#         train=tf.group(update_losses, incr_global_step, gen_train),
#     )


def create_orig_generator(generator_inputs, generator_outputs_channels, ngf=None, **kwargs):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, ngf, stride=1)  # TODO: stride
        layers.append(output)

    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=1 if len(layers) == 1 else 2)  # TODO: stride
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels, stride=1 if decoder_layer == len(layer_specs) - 1 else 2)  # TODO: stride
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels, stride=1)  # TODO: stride
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_orig_discriminator(discrim_inputs, discrim_targets, ndf=None, **kwargs):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=1)  # TODO: stride
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            # stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            stride = 1 if i >= n_layers - 2 else 2  # last two layers here have stride 1  # TODO: stride
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


def create_model_orig(inputs, targets, gan_weight=None, l1_weight=None, l2_weight=None, lr=None, beta1=None, preprocess_image=None, **kwargs):
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator_orig(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator_orig(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator_orig(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss = 0.0
        if preprocess_image:
            # deprocessing is done so that the loss has the right scale
            image = deprocess(image)
            gen_image = deprocess(gen_image)
        if l1_weight:
            gen_L1_loss = tf.reduce_mean(tf.abs(image - gen_image))
            gen_loss += gen_L1_loss * l1_weight
        if l2_weight:
            gen_L2_loss = tf.reduce_mean(tf.square(image - gen_image))
            gen_loss += gen_L2_loss * l2_weight
        if gan_weight:
            gen_GAN_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss += gen_GAN_loss * gan_weight
        gen_psnr_loss = peak_signal_to_noise_ratio(image, gen_image)

    # (num_samples, time_steps, height, width, channels)

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )
    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=discrim_loss,
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_GAN_loss=gen_GAN_loss,
        gen_L1_loss=gen_L1_loss,
        gen_L2_loss=gen_L2_loss,
        gen_state_loss=None,
        gen_psnr_loss=gen_psnr_loss,
        gen_grads_and_vars=gen_grads_and_vars,
        gen_images=outputs,
        gen_masks=None,
        gen_states=None,
        train=tf.group(incr_global_step, gen_train),
    )


def create_model(images, masks, states, actions, generator=None, discriminator=None,
                 gan_weight=None, l1_weight=None, l2_weight=None, mask_weight=None, state_weight=None, reg_weight=None,
                 lr=None, beta1=None, beta2=None, preprocess_image=None, include_context=None, **kwargs):
    context_frames = kwargs.get('context_frames')
    images = tf.unstack(images, axis=1)
    if masks is not None:
        masks = tf.unstack(masks, axis=1)
    else:
        masks = [None] * len(images)
    states = tf.unstack(states, axis=1)
    actions = tf.unstack(actions, axis=1)

    gen_transformed_images = None
    preactivation_feature_maps = None
    feature_maps = None

    with tf.variable_scope("generator") as scope:
        global_step = tf.contrib.framework.get_or_create_global_step()
        if generator == 'orig':
            assert state_weight == 0.0
            gen_images = [create_orig_generator(images[0], 3, **kwargs)]
            gen_masks = None
            gen_states = states[1:]
            assert len(gen_images) == len(gen_states)
        elif generator == 'mask':
            assert state_weight == 0.0
            gen_images, gen_masks, regularization_losses = create_mask_generator(images, masks, states, actions, iter_num=global_step, **kwargs)
            gen_states = states[1:]
        elif generator == 'stp_mask':
            gen_images, gen_masks, gen_states, regularization_losses = create_stp_mask_generator(images, masks, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'cdna_mask':
            gen_images, gen_masks, gen_states = create_cdna_mask_generator(images, masks, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'cdna_accum_tf':
            gen_images, gen_masks, gen_states, gen_transformed_images = \
                create_accum_tf_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'cdna_accum_tf_factorized':
            gen_images, gen_masks, gen_states, gen_transformed_images = \
                create_accum_tf_factorized_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'cdna_accum_tf_factorized_unet':
            gen_images, gen_masks, gen_states, gen_transformed_images = \
                create_accum_tf_factorized_unet_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'afn':
            gen_images, gen_masks, gen_states = create_afn_generator(images, states, actions, iter_num=global_step, **kwargs)
            # gen_states = [0.0] * len(gen_images)
        elif generator == 'vgg_afn':
            gen_images, gen_masks, gen_states, gen_transformed_images = create_vgg_afn_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'vgg_accum_tf_factorized':
            gen_images, gen_masks, gen_states, gen_transformed_images, preactivation_feature_maps, feature_maps = \
                create_vgg_accum_tf_factorized_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'rafn':
            gen_images, gen_masks, gen_states = create_rafn_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'p2p':
            gen_images, gen_masks, gen_states = create_p2p_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'acp2p':
            gen_images, gen_masks, gen_states = create_acp2p_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'acp2pln':
            gen_images, gen_masks, gen_states = create_acp2pln_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator == 'acp2pwn':
            gen_images, gen_masks, gen_states = create_acp2pwn_generator(images, states, actions, iter_num=global_step, **kwargs)
        elif generator in ('cdna', 'dna', 'stp'):
            construct_kwargs = dict(k=kwargs['schedule_sampling_k'],
                                    num_masks=kwargs['num_masks'],
                                    context_frames=kwargs['context_frames'],
                                    cdna=False, dna=False, stp=False,
                                    conf=dict())
            construct_kwargs[generator] = True
            from video_prediction.prediction_model_downsized_lesslayer import construct_model
            iter_num = tf.to_float(global_step)
            gen_images, gen_states, gen_masks, gen_transformed_images = construct_model(images, states=states, actions=actions, iter_num=iter_num, **construct_kwargs)
        else:
            raise ValueError("Invalid generator %r" % generator)

    predict_reals = []
    predict_fakes = []
    discrim_losses = []
    gen_GAN_losses = []
    gen_L1_losses = []
    gen_L2_losses = []
    gen_mask_losses = []
    gen_state_losses = []
    gen_losses = []
    gen_psnr_losses = []
    if include_context:
        tuples = zip(images[:-1], images[1:], gen_images,
                     masks[:-1], masks[1:], gen_masks if gen_masks is not None else [None] * len(gen_images),
                     states[:-1], states[1:], gen_states,
                     actions[:-1], actions[1:])
    else:
        tuples = zip(images[context_frames - 1:-1],
                     images[context_frames:],
                     gen_images[context_frames - 1:],
                     masks[context_frames - 1:-1],
                     masks[context_frames:],
                     (gen_masks if gen_masks is not None else [None] * len(gen_images))[context_frames - 1:],
                     states[context_frames - 1:-1],
                     states[context_frames:],
                     gen_states[context_frames - 1:],
                     actions[context_frames - 1:-1],
                     actions[context_frames:])
    for i, (prev_image, image, gen_image,
            prev_mask, mask, gen_mask,
            prev_state, state, gen_state,
            prev_action, action) in enumerate(tuples):
        if gan_weight:
            # create two copies of discriminator, one for real pairs and one for fake pairs
            # they share the same underlying variables
            with tf.name_scope("real_discriminator_%d" % i):
                with tf.variable_scope("discriminator", reuse=len(discrim_losses) > 0):
                    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                    if discriminator == 'p2p':
                        # predict_real = create_orig_discriminator(prev_image, image, **kwargs)
                        predict_real = create_discriminator(prev_image, image, **kwargs)
                    elif discriminator == 'acp2p':
                        predict_real = create_acp2p_discriminator(prev_image, prev_state, prev_action, image, **kwargs)
                    else:
                        raise ValueError("Invalid discriminator %r" % discriminator)

            with tf.name_scope("fake_discriminator_%d" % i):
                with tf.variable_scope("discriminator", reuse=True):
                    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                    if discriminator == 'p2p':
                        # predict_fake = create_orig_discriminator(prev_image, gen_image, **kwargs)
                        predict_fake = create_discriminator(prev_image, gen_image, **kwargs)
                    elif discriminator == 'acp2p':
                        predict_fake = create_acp2p_discriminator(prev_image, prev_state, prev_action, gen_image, **kwargs)
                    else:
                        raise ValueError("Invalid discriminator %r" % discriminator)

            with tf.name_scope("discriminator_loss_%d" % i):
                # minimizing -tf.log will try to get inputs to 1
                # predict_real => 1
                # predict_fake => 0
                discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss_%d" % i):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss = 0.0
            if preprocess_image:
                # deprocessing is done so that the loss has the right scale
                image = deprocess(image)
                gen_image = deprocess(gen_image)
            if l1_weight:
                gen_L1_loss = tf.reduce_mean(tf.abs(image - gen_image))
                gen_loss += gen_L1_loss * l1_weight
            if l2_weight:
                gen_L2_loss = tf.reduce_mean(tf.square(image - gen_image))
                gen_loss += gen_L2_loss * l2_weight
            if gan_weight:
                gen_GAN_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
                gen_loss += gen_GAN_loss * gan_weight
            if mask_weight:
                gen_mask_loss = tf.reduce_mean(tf.square(mask - gen_mask))
                gen_loss += gen_mask_loss * mask_weight
            if state_weight:
                gen_state_loss = tf.reduce_mean(tf.square(state - gen_state))
                gen_loss += gen_state_loss * state_weight
            gen_psnr_loss = peak_signal_to_noise_ratio(image, gen_image)

        if gan_weight:
            predict_reals.append(predict_real)
            predict_fakes.append(predict_fake)
            discrim_losses.append(discrim_loss)
            gen_GAN_losses.append(gen_GAN_loss)
        if l1_weight:
            gen_L1_losses.append(gen_L1_loss)
        if l2_weight:
            gen_L2_losses.append(gen_L2_loss)
        if mask_weight:
            gen_mask_losses.append(gen_mask_loss)
        if state_weight:
            gen_state_losses.append(gen_state_loss)
        gen_losses.append(gen_loss)
        gen_psnr_losses.append(gen_psnr_loss)

    if gan_weight:
        predict_real = tf.stack(predict_reals, axis=1)
        predict_fake = tf.stack(predict_fakes, axis=1)
        discrim_loss = tf.reduce_mean(discrim_losses)
        gen_GAN_loss = tf.reduce_mean(gen_GAN_losses)
    else:
        predict_real = None
        predict_fake = None
        discrim_loss = None
        gen_GAN_loss = None
    gen_images = tf.stack(gen_images, axis=1)
    if gen_masks:
        gen_masks = [tf.concat(gen_mask, axis=-1) for gen_mask in gen_masks]
        gen_masks = tf.stack(gen_masks, axis=1)
    gen_states = tf.stack(gen_states, axis=1)
    if gen_transformed_images:
        gen_transformed_images = [tf.stack(gen_transformed_images_, axis=-1) for gen_transformed_images_ in gen_transformed_images]
        gen_transformed_images = tf.stack(gen_transformed_images, axis=1)
    if preactivation_feature_maps:
        preactivation_feature_maps = [tf.stack(feature_map, axis=1) for feature_map in zip(*preactivation_feature_maps)]
    if feature_maps:
        feature_maps = [tf.stack(feature_map, axis=1) for feature_map in zip(*feature_maps)]
    if l1_weight:
        gen_L1_loss = tf.reduce_mean(gen_L1_losses)
    else:
        gen_L1_loss = None
    if l2_weight:
        gen_L2_loss = tf.reduce_mean(gen_L2_losses)
    else:
        gen_L2_loss = None
    if mask_weight:
        gen_mask_loss = tf.reduce_mean(gen_mask_losses)
    else:
        gen_mask_loss = None
    if state_weight:
        gen_state_loss = tf.reduce_mean(gen_state_losses)
    else:
        gen_state_loss = None
    gen_loss = tf.reduce_mean(gen_losses)
    if reg_weight:
        gen_loss += tf.reduce_mean(regularization_losses)
    gen_psnr_loss = tf.reduce_mean(gen_psnr_losses)

    if gan_weight:
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(lr, beta1, beta2)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
            control_inputs = [discrim_train]
    else:
        discrim_grads_and_vars = []
        control_inputs = []

    with tf.name_scope("generator_train"):
        with tf.control_dependencies(control_inputs):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1, beta2)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=discrim_loss,
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_GAN_loss=gen_GAN_loss,
        gen_L1_loss=gen_L1_loss,
        gen_L2_loss=gen_L2_loss,
        gen_mask_loss=gen_mask_loss,
        gen_state_loss=gen_state_loss,
        gen_psnr_loss=gen_psnr_loss,
        gen_grads_and_vars=gen_grads_and_vars,
        gen_images=gen_images,
        gen_masks=gen_masks,
        gen_states=gen_states,
        gen_transformed_images=gen_transformed_images,
        preactivation_feature_maps=preactivation_feature_maps,
        feature_maps=feature_maps,
        train=tf.group(incr_global_step, gen_train),
    )


def save_images(fetches, step=None, output_dir=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False, output_dir=None):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to folder containing images")
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--train_val_split", default=0.95)
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default=None,
                        help="directory with checkpoint to resume training from or use for testing")

    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
    parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
    parser.add_argument("--display_freq", type=int, default=0,
                        help="write current training images every display_freq steps")
    parser.add_argument("--gif_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

    parser.add_argument('--niter', type=int, default=None, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=None, help='# of iter to linearly decay learning rate to zero')

    parser.add_argument("--batch_size", type=int, default=32, help="number of images in batch")
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")

    parser.add_argument("--num_masks", type=int, default=10)
    parser.add_argument("--scale_size", type=int, default=64,
                        help="scale images to this size a   fter cropping to a square image")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.9, help="momentum term of adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=1.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--l2_weight", type=float, default=0.0, help="weight on L2 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=0.0, help="weight on GAN term for generator gradient")
    parser.add_argument("--mask_weight", type=float, default=0.0, help="weight on mask term for generator gradient")
    parser.add_argument("--state_weight", type=float, default=1e-4, help="weight on state term for generator gradient")
    parser.add_argument("--reg_weight", type=float, default=0.0, help="weight on transformation  regularization term for generator gradient")

    parser.add_argument("--generator", choices=["orig", "mask", "stp_mask", "cdna_mask",
                                                "cdna_accum_tf", "cdna_accum_tf_factorized", "cdna_accum_tf_factorized_unet",
                                                "afn", "vgg_afn", "rafn",
                                                "vgg_accum_tf_factorized",
                                                "p2p", "acp2p", "acp2pln", "acp2pwn",
                                                "cdna", "dna", "stp"], default="afn")
    parser.add_argument("--discriminator", choices=["p2p", "acp2p"], default="acp2p")
    parser.add_argument("--schedule_sampling_k", type=float, default=1200.0, help='The k hyperparameter for scheduled sampling, -1 for no scheduled sampling.')
    parser.add_argument("--state_schedule_sampling_k", type=float, default=1200.0, help='The k hyperparameter for scheduled sampling of the state, -1 for no scheduled sampling.')
    parser.add_argument("--sequence_length", type=int, default=15, help='sequence length, including context frames.')
    parser.add_argument("--frame_skip", type=int, default=0, help='number of frames to skip between frames. 0 means no skipping.')
    parser.add_argument("--context_frames", type=int, default=2, help='# of frames before predictions.')

    parser.add_argument("--preprocess_image", type=int, default=1, help='whether the images need to be rescaled to the interval [-1, 1].')
    parser.add_argument("--remove_background", type=int, default=0)
    parser.add_argument("--include_context", type=int, default=0, help='whether to include the context frames in the loss.')
    parser.add_argument("--gpu_mem_frac", type=float, default=None, help="fraction of gpu memory to use")

    parser.add_argument("--num_layers", type=int, default=8, help='number of encoding/decoding layers.')
    parser.add_argument("--num_lstm_layers", type=int, default=2, help='number of lstm layers.')
    parser.add_argument("--interleave_lstm", type=int, default=0, help='whether to use 1x1 convolutions between each lstm.')
    parser.add_argument("--use_kernel_initializer", type=int, default=0, help='whether to use Gaussian 0.02 kernel initializer for smearing convolution.')
    parser.add_argument("--use_nonlinear_dynamics", type=int, default=0)
    parser.add_argument("--include_image_enc", type=int, default=0, help='whether to include image encoding in the dynamics.')
    parser.add_argument("--use_gt_mask", type=int, default=0, help='whether to use ground truth mask.')
    parser.add_argument("--dependent_mask", type=int, default=0, help='whether the generated masks should also depend on the transformed images.')
    parser.add_argument("--truncate_hidden_decs", type=int, default=1, help='whether to truncate hidden_decs in cdna_accum_tf_factorized_unet model')
    parser.add_argument("--use_layer_norm", type=int, default=0, help='whether to use layer norm in decoding of vgg models.')
    parser.add_argument("--use_lstm", type=int, default=0, help='whether to use lstm at output of decoder of vgg models.')

    args = parser.parse_args()

    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == "test":
        if args.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"ngf", "ndf"}
        with open(os.path.join(args.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(args, key, val)

    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    decay_lr = args.niter is not None and args.niter_decay is not None
    if decay_lr:
        global_step = tf.contrib.framework.get_or_create_global_step()
        args.lr = tf.clip_by_value(args.lr * tf.to_float(args.niter + args.niter_decay - global_step) / tf.to_float(args.niter_decay), 0, args.lr)

    args_kwargs = args._get_kwargs()
    for k, v in args_kwargs:
        print(k, "=", v)

    examples = load_examples(**dict(args_kwargs))
    print("examples count = %d" % examples.count)
    val_examples = load_examples(**dict(args_kwargs + [('mode', 'val'), ('batch_size', 8)]))
    print("validation examples count = %d" % examples.count)
    if 'vgg' in args.generator:
        stats_examples = load_examples(**dict(args_kwargs + [('num_epochs', 1), ('mode', 'val'), ('batch_size', 32)]))

    with tf.variable_scope('') as training_scope:
        model = create_model(examples.images, examples.masks, examples.states, examples.actions, **dict(args_kwargs))
    with tf.variable_scope(training_scope, reuse=True):
        val_model = create_model(val_examples.images, val_examples.masks, val_examples.states, val_examples.actions, **dict(args_kwargs))
    if 'vgg' in args.generator:
        with tf.variable_scope(training_scope, reuse=True):
            stats_model = create_model(stats_examples.images, stats_examples.masks, stats_examples.states, stats_examples.actions, **dict(args_kwargs))

    # combine frames of video horizontally
    images = tf.concat(tf.unstack(examples.images, axis=1), axis=2)
    gen_images = tf.concat(tf.unstack(model.gen_images, axis=1), axis=2)
    # keep validation frames unchanged
    val_images = val_examples.images
    val_gen_images = val_model.gen_images

    if args.preprocess_image:
        images = deprocess(images)
        gen_images = deprocess(gen_images)
        val_images = deprocess(val_images)
        val_gen_images = deprocess(val_gen_images)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_images"):
        converted_images = tf.image.convert_image_dtype(images, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_gen_images"):
        converted_gen_images = tf.image.convert_image_dtype(gen_images, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_val_images"):
        converted_val_images = tf.image.convert_image_dtype(val_images, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_val_gen_images"):
        converted_val_gen_images = tf.image.convert_image_dtype(val_gen_images, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_val_vis_images"):
        val_vis_images = [val_images[:, 1:, ..., None], val_model.gen_images[..., None]]
        if val_model.gen_transformed_images is not None:
            val_vis_images.append(val_model.gen_transformed_images)
        if val_model.gen_masks is not None:
            val_vis_images.append(tf.tile(val_model.gen_masks[..., None, :], [1, 1, 1, 1, 3, 1]))
        if val_model.gen_transformed_images is not None and val_model.gen_masks is not None:
            val_vis_images.append(val_model.gen_transformed_images * val_model.gen_masks[..., None, :])
        val_vis_images = tf.concat(val_vis_images, axis=-1)
        converted_val_vis_images = tf.image.convert_image_dtype(val_vis_images, dtype=tf.uint8, saturate=True)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "images": tf.map_fn(tf.image.encode_png, converted_images, dtype=tf.string, name="image_pngs"),
            "gen_images": tf.map_fn(tf.image.encode_png, converted_gen_images, dtype=tf.string, name="gen_image_pngs")
        }

    # summaries
    with tf.name_scope("images_summary"):
        tf.summary.image("images", converted_images)

    with tf.name_scope("gen_images_summary"):
        tf.summary.image("gen_images", converted_gen_images)

    if model.gen_masks is not None:
        with tf.name_scope("gen_masks_summary"):
            gen_masks = tf.expand_dims(tf.concat(tf.unstack(tf.concat(tf.unstack(model.gen_masks, axis=-1), axis=-2), axis=1), axis=2), axis=-1)
            tf.summary.image("gen_masks", tf.image.convert_image_dtype(gen_masks, dtype=tf.uint8))

    if model.gen_transformed_images is not None:
        with tf.name_scope("gen_transformed_images_summary"):
            gen_transformed_images = tf.concat(tf.unstack(tf.concat(tf.unstack(model.gen_transformed_images, axis=-1), axis=-3), axis=1), axis=2)
            tf.summary.image("gen_transformed_images", tf.image.convert_image_dtype(gen_transformed_images, dtype=tf.uint8))

    if model.predict_real is not None:
        with tf.name_scope("predict_real_summary"):
            predict_real = tf.concat(tf.unstack(model.predict_real, axis=1), axis=2)
            tf.summary.image("predict_real", tf.image.convert_image_dtype(predict_real, dtype=tf.uint8))

    if model.predict_fake is not None:
        with tf.name_scope("predict_fake_summary"):
            predict_fake = tf.concat(tf.unstack(model.predict_fake, axis=1), axis=2)
            tf.summary.image("predict_fake", tf.image.convert_image_dtype(predict_fake, dtype=tf.uint8))

    if model.discrim_loss is not None:
        tf.summary.scalar("discriminator_loss", model.discrim_loss)
    if model.gen_GAN_loss is not None:
        tf.summary.scalar("generator_GAN_loss", model.gen_GAN_loss)
    if model.gen_L1_loss is not None:
        tf.summary.scalar("generator_L1_loss", model.gen_L1_loss)
    if model.gen_L2_loss is not None:
        tf.summary.scalar("generator_L2_loss", model.gen_L2_loss)
    if model.gen_mask_loss is not None:
        tf.summary.scalar("generator_mask_loss", model.gen_mask_loss)
    if model.gen_state_loss is not None:
        tf.summary.scalar("generator_state_loss", model.gen_state_loss)
    tf.summary.scalar("generator_psnr_loss", model.gen_psnr_loss)

    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name + "/values", var)

    # for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
    #     tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # with tf.name_scope("set_global_step"):
    #     global_step = tf.contrib.framework.get_or_create_global_step()
    #     set_global_step = tf.assign(global_step, 7800)

    if 'vgg' in args.generator:
        loading_callback = vgg_assign_from_values_fn()
        loading_stats_callback = vgg_stats_assign_from_values_fn()
    else:
        loading_callback = None
        loading_stats_callback = None

    saver = tf.train.Saver(max_to_keep=1)

    # if args.checkpoint is not None:
    #     print("loading metagraph from checkpoint")
    #     checkpoint = tf.train.latest_checkpoint(args.checkpoint)
    #     saver = tf.train.import_meta_graph(os.path.join(checkpoint, 'meta'))
    # else:
    #     saver = tf.train.Saver(max_to_keep=1)

    # restorable_variables = [variable for variable in tf.global_variables() if variable.name.startswith('generator')]
    # restorable_saver = tf.train.Saver(restorable_variables, max_to_keep=1)

    logdir = args.output_dir if (args.trace_freq > 0 or args.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    if args.gpu_mem_frac is None:
        config = None
    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem_frac
    with sv.managed_session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        # sess.run(set_global_step)

        if loading_callback:
            loading_callback(sess)
        if loading_stats_callback:
            print("computing statistics of vgg features")
            start_time = time.time()
            online_stats = [OnlineStatistics(axis=(0, 1, 2, 3)) for _ in stats_model.preactivation_feature_maps]
            while True:
                try:
                    preactivation_feature_maps = sess.run(stats_model.preactivation_feature_maps)
                except tf.errors.OutOfRangeError:
                    break
                for online_stat, preactivation_feature_map in zip(online_stats, preactivation_feature_maps):
                    online_stat.add_data(preactivation_feature_map)
            print("done in %ds" % (time.time() - start_time))
            stat_values = []
            for block_id, online_stat in enumerate(online_stats):
                # stat_values.append(np.zeros_like(online_stat.mean))  # don't set offset
                stat_values.append(online_stat.mean)
                stat_values.append(online_stat.std)
            loading_stats_callback(sess, stat_values)

        if args.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
            # import IPython as ipy; ipy.embed()
            # restorable_saver.restore(sess, checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if args.max_epochs is not None:
            max_steps = examples.steps_per_epoch * args.max_epochs
        if args.max_steps is not None:
            max_steps = args.max_steps
        if decay_lr:
            max_steps = args.niter + args.niter_decay

        if args.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(args.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(args.progress_freq):
                    if model.discrim_loss is not None:
                        fetches["discrim_loss"] = model.discrim_loss
                    if model.gen_GAN_loss is not None:
                        fetches["gen_GAN_loss"] = model.gen_GAN_loss
                    if model.gen_L1_loss is not None:
                        fetches["gen_L1_loss"] = model.gen_L1_loss
                    if model.gen_L2_loss is not None:
                        fetches["gen_L2_loss"] = model.gen_L2_loss
                    if model.gen_mask_loss is not None:
                        fetches["gen_mask_loss"] = model.gen_mask_loss
                    if model.gen_state_loss is not None:
                        fetches["gen_state_loss"] = model.gen_state_loss
                    if decay_lr:
                        fetches["lr"] = args.lr

                if should(args.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(args.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(args.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(args.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(args.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(args.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * args.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * args.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    if model.discrim_loss is not None:
                        print("discrim_loss", results["discrim_loss"])
                    if model.gen_GAN_loss is not None:
                        print("gen_GAN_loss", results["gen_GAN_loss"])
                    if model.gen_L1_loss is not None:
                        print("gen_L1_loss", results["gen_L1_loss"])
                    if model.gen_L2_loss is not None:
                        print("gen_L2_loss", results["gen_L2_loss"])
                    if model.gen_mask_loss is not None:
                        print("gen_mask_loss", results["gen_mask_loss"])
                    if model.gen_state_loss is not None:
                        print("gen_state_loss", results["gen_state_loss"])
                    if decay_lr:
                        print("lr", results["lr"])

                if should(args.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(args.output_dir, "model"), global_step=sv.global_step)

                # import IPython as ipy; ipy.embed()
                # import matplotlib
                # import matplotlib.cm
                # import cv2
                #
                # images, *feature_maps = sess.run([converted_val_images] + val_model.feature_maps)
                #
                # output_dir = 'feature_maps'
                # feature_name = ['x1', 'x2', 'x3', 'x4', 'x5']
                # feature = [feature_map[0][0].transpose((2, 0, 1)) for feature_map in feature_maps]
                # feature_limits = [None] * len(feature)
                #
                # for i, (y, feature_limit) in enumerate(zip(feature, feature_limits)):
                #     if feature_limit is None:
                #         feature_limit = [y.min(axis=(1, 2)), y.max(axis=(1, 2))]
                #     else:
                #         feature_limit = [np.minimum(feature_limit[0], y.min(axis=(1, 2))),
                #                          np.maximum(feature_limit[1], y.max(axis=(1, 2)))]
                #     feature_limits[i] = feature_limit
                #
                # os.makedirs(os.path.join(output_dir, 'y'), exist_ok=True)
                # for y_name, y, y_limit in zip(feature_name, feature, feature_limits):
                #     y = (y - y_limit[0][:, None, None]) / (y_limit[1] - y_limit[0])[:, None, None]
                #     y = matplotlib.cm.viridis(y)
                #     y = (y * 255.0).astype(np.uint8)
                #     for i_slice, y_slice in enumerate(y):
                #         y_slice_fname = os.path.join(output_dir, 'y', '%s_%03d.jpg' % (y_name, i_slice))
                #         y_slice = cv2.cvtColor(y_slice, cv2.COLOR_RGBA2BGR)
                #         cv2.imwrite(y_slice_fname, y_slice, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                if should(args.gif_freq):
                    # (num_samples, time_steps, height, width, channels)
                    # (num_samples, time_steps - 1, height, width, channels)
                    # (num_samples, time_steps - 1, state_dim)
                    # images, gen_images, gen_states = sess.run([converted_val_images, converted_val_gen_images, val_model.gen_states])
                    images, gen_images, vis_images = sess.run([converted_val_images, converted_val_gen_images, converted_val_vis_images])

                    # if args.state_weight:
                    #     gen_state_images = []
                    #     for gen_states_ in gen_states:
                    #         gen_state_images_ = np.empty_like(gen_images[0])
                    #         for t, gen_state in enumerate(gen_states_):
                    #             gen_state_images_[t] = state_to_image(gen_state)
                    #         gen_state_images.append(gen_state_images_)

                    # (num_samples, time_steps, height, width, channels)
                    gen_images = np.concatenate([images[:, 0:1], gen_images], axis=1)
                    # if args.state_weight:
                    #     gen_state_images = np.concatenate([images[:, 0:1], gen_state_images], axis=1)

                    import moviepy.editor as mpy
                    image_dir = os.path.join(args.output_dir, "images")
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)

                    names = ['images', 'gen_images', 'vis_images']
                    images = [images, gen_images, vis_images]
                    # if args.state_weight:
                    #     names += ['gen_state_images']
                    #     images += [gen_state_images]

                    for name, images in zip(names, images):
                        num_samples, time_steps, height, width, channels, *other_channels = images.shape
                        if other_channels:
                            other_channels, = other_channels
                            images = images.transpose((1, 5, 2, 0, 3, 4))
                            images = images.reshape(
                                (time_steps, other_channels * height, num_samples * width, channels))
                        else:
                            images = images.transpose((1, 2, 0, 3, 4))
                            images = images.reshape((time_steps, height, num_samples * width, channels))
                        clip = mpy.ImageSequenceClip(list(images), fps=4)
                        filename = "%08d-%s.gif" % (results["global_step"], name)
                        clip.write_gif(os.path.join(image_dir, filename))

                if sv.should_stop():
                    break


if __name__ == '__main__':
    main()

