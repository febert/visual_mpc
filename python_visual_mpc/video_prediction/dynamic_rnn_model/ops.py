import numpy as np
import tensorflow as tf


def get_coords(img_shape):
    """
    returns coordinate grid corresponding to identity appearance flow
    :param img_shape:
    :return:
    """
    y = tf.cast(tf.range(img_shape[1]), tf.float32)
    x = tf.cast(tf.range(img_shape[2]), tf.float32)
    batch_size = img_shape[0]

    X,Y = tf.meshgrid(x,y)
    coords = tf.expand_dims(tf.stack((X, Y), axis=2), axis=0)
    coords = tf.tile(coords, [batch_size, 1,1,1])
    return coords

def resample_layer(src_img, warp_pts):
    return tf.contrib.resampler.resampler(src_img, warp_pts)

def warp_pts_layer(flow_field):
    img_shape = flow_field.get_shape().as_list()
    return flow_field + get_coords(img_shape)

def apply_warp(I0, flow_field):
    warp_pts = warp_pts_layer(flow_field)
    return resample_layer(I0, warp_pts)

def dense(inputs, units):
    with tf.variable_scope('dense'):
        input_shape = inputs.get_shape().as_list()
        kernel_shape = [input_shape[1], units]
        kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [units], dtype=tf.float32, initializer=tf.zeros_initializer())
        outputs = tf.matmul(inputs, kernel) + bias
        return outputs


def pad1d(inputs, size, strides=(1,), padding='SAME', mode='CONSTANT'):
    size = list(size) if isinstance(size, (tuple, list)) else [size]
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides]
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    in_width = input_shape[1]
    if padding in ('SAME', 'FULL'):
        if in_width % strides[0] == 0:
            pad_along_width = max(size[0] - strides[0], 0)
        else:
            pad_along_width = max(size[0] - (in_width % strides[0]), 0)
        if padding == 'SAME':
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
        else:
            pad_left = pad_along_width
            pad_right = pad_along_width
        padding_pattern = [[0, 0],
                           [pad_left, pad_right],
                           [0, 0]]
        outputs = tf.pad(inputs, padding_pattern, mode=mode)
    elif padding == 'VALID':
        outputs = inputs
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    return outputs


def conv1d(inputs, filters, kernel_size, strides=(1,), padding='SAME', kernel=None, use_bias=True):
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size]
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides]
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('conv1d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" % (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        inputs = pad1d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    stride, = strides
    outputs = tf.nn.conv1d(inputs, kernel, stride, padding=padding)
    if use_bias:
        with tf.variable_scope('conv1d'):
            bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def pad2d_paddings(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME'):
    """
    Computes the paddings for a 4-D tensor according to the convolution padding algorithm.

    See pad2d.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
    """
    size = list(size) if isinstance(size, (tuple, list)) else [size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    rate = list(rate) if isinstance(rate, (tuple, list)) else [rate] * 2
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 4
    in_height, in_width = input_shape[1:3]
    if padding in ('SAME', 'FULL'):
        if in_height % strides[0] == 0:
            pad_along_height = max(size[0] - strides[0], 0)
        else:
            pad_along_height = max(size[0] - (in_height % strides[0]), 0)
        if in_width % strides[1] == 0:
            pad_along_width = max(size[1] - strides[1], 0)
        else:
            pad_along_width = max(size[1] - (in_width % strides[1]), 0)
        if padding == 'SAME':
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
        else:
            pad_top = pad_along_height
            pad_bottom = pad_along_height
            pad_left = pad_along_width
            pad_right = pad_along_width
        # TODO: not sure if paddings are simply multiplied by rate
        paddings = [[0, 0],
                    [pad_top * rate[0], pad_bottom * rate[0]],
                    [pad_left * rate[1], pad_right * rate[1]],
                    [0, 0]]
    elif padding == 'VALID':
        paddings = [[0, 0]] * 4
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    return paddings


def pad2d(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME', mode='CONSTANT'):
    """
    Pads a 4-D tensor according to the convolution padding algorithm.

    Convolution with a padding scheme
        conv2d(..., padding=padding)
    is equivalent to zero-padding of the input with such scheme, followed by
    convolution with 'VALID' padding
        padded = pad2d(..., padding=padding, mode='CONSTANT')
        conv2d(padded, ..., padding='VALID')

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        padding: A string, either 'VALID', 'SAME', or 'FULL'. The padding algorithm.
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).

    Returns:
        A 4-D tensor.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
    """
    paddings = pad2d_paddings(inputs, size, strides=strides, rate=rate, padding=padding)
    if paddings == [[0, 0]] * 4:
        outputs = inputs
    else:
        outputs = tf.pad(inputs, paddings, mode=mode)
    return outputs


def local2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
            kernel=None, flip_filters=False,
            use_bias=True, channelwise=False):
    """
    2-D locally connected operation.

    Works similarly to 2-D convolution except that the weights are unshared, that is, a different set of filters is
    applied at each different patch of the input.

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernel: A 6-D or 7-D tensor of shape
            `[in_height, in_width, kernel_size[0], kernel_size[1], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], in_channels, filters]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if strides != [1, 1]:
        raise NotImplementedError
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    input_shape = inputs.get_shape().as_list()
    if padding == 'SAME':
        output_shape = input_shape[:3] + [filters]
    elif padding == 'VALID':
        output_shape = [input_shape[0], input_shape[1] - kernel_size[0] + 1, input_shape[2] - kernel_size[1] + 1, filters]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)

    if channelwise:
        if filters not in (input_shape[-1], 1):
            raise ValueError("Number of filters should match the number of input channels or be 1 when channelwise "
                             "is true, but got filters=%r and %d input channels" % (filters, input_shape[-1]))
        kernel_shape = output_shape[1:3] + kernel_size + [filters]
    else:
        kernel_shape = output_shape[1:3] + kernel_size + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('local2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel.get_shape().as_list() not in (kernel_shape, [input_shape[0], kernel_shape]):   ###
            raise ValueError("Expecting kernel with shape %s or %s but instead got kernel with shape %s"
                             % (tuple(kernel_shape), tuple([input_shape[0], kernel_shape]), tuple(kernel.get_shape().as_list())))  ###

    outputs = []
    for i in range(kernel_size[0]):
        filter_h_ind = -i-1 if flip_filters else i
        if padding == 'VALID':
            ii = i
        else:
            ii = i - (kernel_size[0] // 2)
        input_h_slice = slice(max(ii, 0), min(ii + output_shape[1], input_shape[1]))
        output_h_slice = slice(input_h_slice.start - ii, input_h_slice.stop - ii)
        assert 0 <= output_h_slice.start < output_shape[1]
        assert 0 < output_h_slice.stop <= output_shape[1]

        for j in range(kernel_size[1]):
            filter_w_ind = -j-1 if flip_filters else j
            if padding == 'VALID':
                jj = j
            else:
                jj = j - (kernel_size[1] // 2)
            input_w_slice = slice(max(jj, 0), min(jj + output_shape[2], input_shape[2]))
            output_w_slice = slice(input_w_slice.start - jj, input_w_slice.stop - jj)
            assert 0 <= output_w_slice.start < output_shape[2]
            assert 0 < output_w_slice.stop <= output_shape[2]
            if channelwise:
                inc = inputs[:, input_h_slice, input_w_slice, :] * \
                      kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :]
            else:
                inc = tf.reduce_sum(inputs[:, input_h_slice, input_w_slice, :, None] *
                                    kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :, :], axis=-2)
            # equivalent to this
            # outputs[:, output_h_slice, output_w_slice, :] += inc
            paddings = [[0, 0], [output_h_slice.start, output_shape[1] - output_h_slice.stop],
                        [output_w_slice.start, output_shape[2] - output_w_slice.stop], [0, 0]]
            outputs.append(tf.pad(inc, paddings))
    outputs = tf.add_n(outputs)
    if use_bias:
        with tf.variable_scope('local2d'):
            bias = tf.get_variable('bias', output_shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def separable_local2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
                      vertical_kernel=None, horizontal_kernel=None, flip_filters=False,
                      use_bias=True, channelwise=False):
    """
    2-D locally connected operation with separable filters.

    Note that, unlike tf.nn.separable_conv2d, this is spatial separability between dimensions 1 and 2.

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        vertical_kernel: A 5-D or 6-D tensor of shape
            `[in_height, in_width, kernel_size[0], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[0], in_channels, filters]`.
        horizontal_kernel: A 5-D or 6-D tensor of shape
            `[in_height, in_width, kernel_size[1], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[1], in_channels, filters]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if strides != [1, 1]:
        raise NotImplementedError
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    input_shape = inputs.get_shape().as_list()
    if padding == 'SAME':
        output_shape = input_shape[:3] + [filters]
    elif padding == 'VALID':
        output_shape = [input_shape[0], input_shape[1] - kernel_size[0] + 1, input_shape[2] - kernel_size[1] + 1, filters]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)

    kernels = [vertical_kernel, horizontal_kernel]
    for i, (kernel_type, kernel_length, kernel) in enumerate(zip(['vertical', 'horizontal'], kernel_size, kernels)):
        if channelwise:
            if filters not in (input_shape[-1], 1):
                raise ValueError("Number of filters should match the number of input channels or be 1 when channelwise "
                                 "is true, but got filters=%r and %d input channels" % (filters, input_shape[-1]))
            kernel_shape = output_shape[1:3] + [kernel_length, filters]
        else:
            kernel_shape = output_shape[1:3] + [kernel_length, input_shape[-1], filters]
        if kernel is None:
            with tf.variable_scope('separable_local2d'):
                kernel = tf.get_variable('%s_kernel' % kernel_type, kernel_shape, dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
                kernels[i] = kernel
        else:
            if kernel.get_shape().as_list() not in (kernel_shape, [input_shape[0], kernel_shape]):  ##
                raise ValueError("Expecting %s kernel with shape %s or %s but instead got kernel with shape %s"
                                 % (kernel_type,
                                    tuple(kernel_shape), tuple([input_shape[0], kernel_shape]), ##
                                    tuple(kernel.get_shape().as_list())))

    outputs = []
    for i in range(kernel_size[0]):
        filter_h_ind = -i-1 if flip_filters else i
        if padding == 'VALID':
            ii = i
        else:
            ii = i - (kernel_size[0] // 2)
        input_h_slice = slice(max(ii, 0), min(ii + output_shape[1], input_shape[1]))
        output_h_slice = slice(input_h_slice.start - ii, input_h_slice.stop - ii)
        assert 0 <= output_h_slice.start < output_shape[1]
        assert 0 < output_h_slice.stop <= output_shape[1]

        for j in range(kernel_size[1]):
            filter_w_ind = -j-1 if flip_filters else j
            if padding == 'VALID':
                jj = j
            else:
                jj = j - (kernel_size[1] // 2)
            input_w_slice = slice(max(jj, 0), min(jj + output_shape[2], input_shape[2]))
            output_w_slice = slice(input_w_slice.start - jj, input_w_slice.stop - jj)
            assert 0 <= output_w_slice.start < output_shape[2]
            assert 0 < output_w_slice.stop <= output_shape[2]
            if channelwise:
                inc = inputs[:, input_h_slice, input_w_slice, :] * \
                      kernels[0][..., output_h_slice, output_w_slice, filter_h_ind, :] * \
                      kernels[1][..., output_h_slice, output_w_slice, filter_w_ind, :]
            else:
                inc = tf.reduce_sum(inputs[:, input_h_slice, input_w_slice, :, None] *
                                    kernels[0][..., output_h_slice, output_w_slice, filter_h_ind, :, :] *
                                    kernels[1][..., output_h_slice, output_w_slice, filter_w_ind, :, :],
                                    axis=-2)
            # equivalent to this
            # outputs[:, output_h_slice, output_w_slice, :] += inc
            paddings = [[0, 0], [output_h_slice.start, output_shape[1] - output_h_slice.stop],
                        [output_w_slice.start, output_shape[2] - output_w_slice.stop], [0, 0]]
            outputs.append(tf.pad(inc, paddings))
    outputs = tf.add_n(outputs)
    if use_bias:
        with tf.variable_scope('separable_local2d'):
            bias = tf.get_variable('bias', output_shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def kronecker_local2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
                      kernels=None, flip_filters=False, use_bias=True, channelwise=False):
    """
    2-D locally connected operation with filters represented as a kronecker product of smaller filters

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernel: A list of 6-D or 7-D tensors of shape
            `[in_height, in_width, kernel_size[i][0], kernel_size[i][1], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[i][0], kernel_size[i][1], in_channels, filters]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if strides != [1, 1]:
        raise NotImplementedError
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    input_shape = inputs.get_shape().as_list()
    if padding == 'SAME':
        output_shape = input_shape[:3] + [filters]
    elif padding == 'VALID':
        output_shape = [input_shape[0], input_shape[1] - kernel_size[0] + 1, input_shape[2] - kernel_size[1] + 1, filters]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)

    if channelwise:
        if filters not in (input_shape[-1], 1):
            raise ValueError("Number of filters should match the number of input channels or be 1 when channelwise "
                             "is true, but got filters=%r and %d input channels" % (filters, input_shape[-1]))
        kernel_shape = output_shape[1:3] + kernel_size + [filters]
        factor_kernel_shape = output_shape[1:3] + [None, None, filters]
    else:
        kernel_shape = output_shape[1:3] + kernel_size + [input_shape[-1], filters]
        factor_kernel_shape = output_shape[1:3] + [None, None, input_shape[-1], filters]
    if kernels is None:
        with tf.variable_scope('kronecker_local2d'):
            kernels = [tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))]
        filter_h_lengths = [kernel_size[0]]
        filter_w_lengths = [kernel_size[1]]
    else:
        for kernel in kernels:
            if not ((len(kernel.shape) == len(factor_kernel_shape) and
                    all(((k == f) or f is None) for k, f in zip(kernel.get_shape().as_list(), factor_kernel_shape))) or
                    (len(kernel.shape) == (len(factor_kernel_shape) + 1) and
                    all(((k == f) or f is None) for k, f in zip(kernel.get_shape().as_list(), [input_shape[0], factor_kernel_shape])))):
                raise ValueError("Expecting kernel with shape %s or %s but instead got kernel with shape %s"
                                 % (tuple(factor_kernel_shape), tuple([input_shape[0], factor_kernel_shape]),
                                    tuple(kernel.get_shape().as_list())))
        if channelwise:
            filter_h_lengths, filter_w_lengths = list(zip(*[kernel.get_shape().as_list()[-3:-1] for kernel in kernels]))
        else:
            filter_h_lengths, filter_w_lengths = list(zip(*[kernel.get_shape().as_list()[-4:-2] for kernel in kernels]))
        if [np.prod(filter_h_lengths), np.prod(filter_w_lengths)] != kernel_size:
            raise ValueError("Expecting kernel size %s but instead got kernel size %s"
                             % (tuple(kernel_size), tuple([np.prod(filter_h_lengths), np.prod(filter_w_lengths)])))

    def get_inds(ind, lengths):
        inds = []
        for i in range(len(lengths)):
            curr_ind = int(ind)
            for j in range(len(lengths) - 1, i, -1):
                curr_ind //= lengths[j]
            curr_ind %= lengths[i]
            inds.append(curr_ind)
        return inds

    outputs = []
    for i in range(kernel_size[0]):
        if padding == 'VALID':
            ii = i
        else:
            ii = i - (kernel_size[0] // 2)
        input_h_slice = slice(max(ii, 0), min(ii + output_shape[1], input_shape[1]))
        output_h_slice = slice(input_h_slice.start - ii, input_h_slice.stop - ii)
        assert 0 <= output_h_slice.start < output_shape[1]
        assert 0 < output_h_slice.stop <= output_shape[1]

        for j in range(kernel_size[1]):
            if padding == 'VALID':
                jj = j
            else:
                jj = j - (kernel_size[1] // 2)
            input_w_slice = slice(max(jj, 0), min(jj + output_shape[2], input_shape[2]))
            output_w_slice = slice(input_w_slice.start - jj, input_w_slice.stop - jj)
            assert 0 <= output_w_slice.start < output_shape[2]
            assert 0 < output_w_slice.stop <= output_shape[2]
            kernel_slice = 1.0
            for filter_h_ind, filter_w_ind, kernel in zip(get_inds(i, filter_h_lengths), get_inds(j, filter_w_lengths), kernels):
                if flip_filters:
                    filter_h_ind = -filter_h_ind-1
                    filter_w_ind = -filter_w_ind-1
                if channelwise:
                    kernel_slice *= kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :]
                else:
                    kernel_slice *= kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :, :]
            if channelwise:
                inc = inputs[:, input_h_slice, input_w_slice, :] * kernel_slice
            else:
                inc = tf.reduce_sum(inputs[:, input_h_slice, input_w_slice, :, None] * kernel_slice, axis=-2)
            # equivalent to this
            # outputs[:, output_h_slice, output_w_slice, :] += inc
            paddings = [[0, 0], [output_h_slice.start, output_shape[1] - output_h_slice.stop],
                        [output_w_slice.start, output_shape[2] - output_w_slice.stop], [0, 0]]
            outputs.append(tf.pad(inc, paddings))
    outputs = tf.add_n(outputs)
    if use_bias:
        with tf.variable_scope('kronecker_local2d'):
            bias = tf.get_variable('bias', output_shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def depthwise_conv2d(inputs, channel_multiplier, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True):
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    input_shape = inputs.get_shape().as_list()
    kernel_shape = kernel_size + [input_shape[-1], channel_multiplier]
    if kernel is None:
        with tf.variable_scope('depthwise_conv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" % (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    outputs = tf.nn.depthwise_conv2d(inputs, kernel, [1] + strides + [1], padding=padding)
    if use_bias:
        with tf.variable_scope('depthwise_conv2d'):
            bias = tf.get_variable('bias', [input_shape[-1] * channel_multiplier], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True, bias=None, dtype = tf.float32):

    """
    2-D convolution.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('conv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" % (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    outputs = tf.nn.conv2d(inputs, kernel, [1] + strides + [1], padding=padding)
    if use_bias:
        if bias is None:
            with tf.variable_scope('conv2d'):
                bias = tf.get_variable('bias', [filters], dtype=dtype, initializer=tf.zeros_initializer())
                outputs = tf.nn.bias_add(outputs, bias)
        else:
            bias_shape = [filters]
            if bias_shape != bias.get_shape().as_list():
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s" %
                                 (tuple(bias_shape), tuple(bias.get_shape().as_list())))
    return outputs


def deconv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True):
    """
    2-D transposed convolution.

    Notes on padding:
       The equivalent of transposed convolution with full padding is a convolution with valid padding, and
       the equivalent of transposed convolution with valid padding is a convolution with full padding.

    Reference:
        http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [filters, input_shape[-1]]
    if kernel is None:
        with tf.variable_scope('deconv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" % (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        output_h, output_w = [s * (i + 1) - k for (i, k, s) in zip(input_shape[1:3], kernel_size, strides)]
    elif padding == 'SAME':
        output_h, output_w = [s * i for (i, s) in zip(input_shape[1:3], strides)]
    elif padding == 'VALID':
        output_h, output_w = [s * (i - 1) + k for (i, k, s) in zip(input_shape[1:3], kernel_size, strides)]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    output_shape = [input_shape[0], output_h, output_w, filters]
    outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape, [1] + strides + [1], padding=padding)
    if use_bias:
        with tf.variable_scope('deconv2d'):
            bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def get_bilinear_kernel(strides):
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    strides = np.array(strides)
    kernel_size = 2 * strides - strides % 2
    center = strides - (kernel_size % 2 == 1) - 0.5 * (kernel_size % 2 != 1)
    vertical_kernel = 1 - abs(np.arange(kernel_size[0]) - center[0]) / strides[0]
    horizontal_kernel = 1 - abs(np.arange(kernel_size[1]) - center[1]) / strides[1]
    kernel = vertical_kernel[:, None] * horizontal_kernel[None, :]
    return kernel


def upsample2d(inputs, strides, padding='SAME'):
    single_bilinear_kernel = get_bilinear_kernel(strides).astype(np.float32)
    input_shape = inputs.get_shape().as_list()
    bilinear_kernel = tf.matrix_diag(tf.tile(tf.constant(single_bilinear_kernel)[..., None], (1, 1, input_shape[-1])))
    outputs = deconv2d(inputs, input_shape[-1], kernel_size=single_bilinear_kernel.shape,
                       strides=strides, kernel=bilinear_kernel, padding=padding, use_bias=False)
    return outputs


def upsample_conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True, bias=None):
    """
    Upsamples the inputs by a factor using bilinear interpolation and the performs conv2d on the upsampled input. This
    function is more computationally and memory efficient than a naive implementation. Unlike a naive implementation
    that would upsample the input first, this implementation first convolves the bilinear kernel with the given kernel,
    and then performs the convolution (actually a deconv2d) with the combined kernel. As opposed to just using deconv2d
    directly, this function is less prone to checkerboard artifacts thanks to the implicit bilinear upsampling.

    Example:
        # >>> import numpy as np
        # >>> import tensorflow as tf
        # >>> from docile.ops import upsample_conv2d, upsample2d, conv2d, pad2d_paddings
        # >>> inputs_shape = [4, 8, 8, 64]
        # >>> kernel_size = [3, 3]  # for convolution
        # >>> filters = 32  # for convolution
        # >>> strides = [2, 2]  # for upsampling
        # >>> inputs = tf.get_variable("inputs", inputs_shape)
        # >>> kernel = tf.get_variable("kernel", (kernel_size[0], kernel_size[1], inputs_shape[-1], filters))
        # >>> bias = tf.get_variable("bias", (filters,))
        # >>> outputs = upsample_conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, \
        #                               kernel=kernel, bias=bias)
        # >>> # upsample with bilinear interpolation
        # >>> inputs_up = upsample2d(inputs, strides=strides, padding='VALID')
        # >>> # convolve upsampled input with kernel
        # >>> outputs_up = conv2d(inputs_up, filters, kernel_size=kernel_size, strides=(1, 1), \
        #                         kernel=kernel, bias=bias, padding='FULL')
        # >>> # crop appropriately
        # >>> same_paddings = pad2d_paddings(inputs, kernel_size, strides=(1, 1), padding='SAME')
        # >>> full_paddings = pad2d_paddings(inputs, kernel_size, strides=(1, 1), padding='FULL')
        # >>> crop_top = (strides[0] - strides[0] % 2) // 2 + full_paddings[1][1] - same_paddings[1][1]
        # >>> crop_left = (strides[1] - strides[1] % 2) // 2 + full_paddings[2][1] - same_paddings[2][1]
        # >>> outputs_up = outputs_up[:, crop_top:crop_top + strides[0] * inputs_shape[1], \
        #                             crop_left:crop_left + strides[1] * inputs_shape[2], :]
        # >>> sess = tf.Session()
        # >>> sess.run(tf.global_variables_initializer())
        # >>> assert np.allclose(*sess.run([outputs, outputs_up]), atol=1e-5)

    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding != 'SAME':
        raise NotImplementedError
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('upsample_conv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" %
                             (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))

    # convolve bilinear kernel with kernel
    single_bilinear_kernel = get_bilinear_kernel(strides).astype(np.float32)
    kernel_transposed = tf.transpose(kernel, (0, 1, 3, 2))
    kernel_reshaped = tf.reshape(kernel_transposed, kernel_size + [1, input_shape[-1] * filters])
    kernel_up_reshaped = conv2d(tf.constant(single_bilinear_kernel)[None, :, :, None], input_shape[-1] * filters,
                                kernel_size=kernel_size, kernel=kernel_reshaped, padding='FULL', use_bias=False)
    kernel_up = tf.reshape(kernel_up_reshaped,
                           kernel_up_reshaped.get_shape().as_list()[1:3] + [filters, input_shape[-1]])

    # deconvolve with the bilinearly convolved kernel
    outputs = deconv2d(inputs, filters, kernel_size=kernel_up.get_shape().as_list()[:2], strides=strides,
                       kernel=kernel_up, padding='SAME', use_bias=False)
    if use_bias:
        if bias is None:
            with tf.variable_scope('upsample_conv2d'):
                bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
                outputs = tf.nn.bias_add(outputs, bias)
        else:
            bias_shape = [filters]
            if bias_shape != bias.get_shape().as_list():
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s" %
                                 (tuple(bias_shape), tuple(bias.get_shape().as_list())))
    return outputs


def conv3d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME'):
    with tf.variable_scope('conv3d'):
        kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 3
        strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 3
        input_shape = inputs.get_shape().as_list()
        kernel_shape = list(kernel_size) + [input_shape[-1], filters]
        kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        outputs = tf.nn.conv3d(inputs, kernel, [1] + strides + [1], padding=padding)
        bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
        outputs = tf.nn.bias_add(outputs, bias)
        return outputs


def pool2d(inputs, pool_size, strides=(1, 1), padding='SAME', pool_mode='max'):
    pool_size = list(pool_size) if isinstance(pool_size, (tuple, list)) else [pool_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding == 'FULL':
        inputs = pad2d(inputs, pool_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    if pool_mode == 'max':
        outputs = tf.nn.max_pool(inputs, [1] + pool_size + [1], [1] + strides + [1], padding=padding)
    elif pool_mode == 'avg':
        outputs = tf.nn.avg_pool(inputs, [1] + pool_size + [1], [1] + strides + [1], padding=padding)
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)
    return outputs


def conv_pool2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True, bias=None, pool_mode='avg'):
    """
    Similar optimization as in upsample_conv2d

    Example:
        # >>> import numpy as np
        # >>> import tensorflow as tf
        # >>> from docile.ops import conv_pool2d, conv2d, pool2d
        # >>> inputs_shape = [4, 16, 16, 32]
        # >>> kernel_size = [3, 3]  # for convolution
        # >>> filters = 64  # for convolution
        # >>> strides = [2, 2]  # for pooling
        # >>> inputs = tf.get_variable("inputs", inputs_shape)
        # >>> kernel = tf.get_variable("kernel", (kernel_size[0], kernel_size[1], inputs_shape[-1], filters))
        # >>> bias = tf.get_variable("bias", (filters,))
        # >>> outputs = conv_pool2d(inputs, filters, kernel_size=kernel_size, strides=strides, \
        #                           kernel=kernel, bias=bias, pool_mode='avg')
        # >>> inputs_conv = conv2d(inputs, filters, kernel_size=kernel_size, strides=(1, 1), \
        #                          kernel=kernel, bias=bias)
        # >>> outputs_pool = pool2d(inputs_conv, pool_size=strides, strides=strides, pool_mode='avg')
        # >>> sess = tf.Session()
        # >>> sess.run(tf.global_variables_initializer())
        # >>> assert np.allclose(*sess.run([outputs, outputs_pool]), atol=1e-5)

    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding != 'SAME' or pool_mode != 'avg':
        raise NotImplementedError
    input_shape = inputs.get_shape().as_list()
    if input_shape[1] % strides[0] or input_shape[2] % strides[1]:
        raise NotImplementedError("The height and width of the input should be "
                                  "an integer multiple of the respective stride.")
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('conv_pool2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" %
                             (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))

    # pool kernel
    kernel_reshaped = tf.reshape(kernel, [1] + kernel_size + [input_shape[-1] * filters])
    kernel_pool_reshaped = pool2d(kernel_reshaped, strides, padding='FULL', pool_mode='avg')
    kernel_pool = tf.reshape(kernel_pool_reshaped,
                             kernel_pool_reshaped.get_shape().as_list()[1:3] + [input_shape[-1], filters])

    outputs = conv2d(inputs, filters, kernel_size=kernel_pool.get_shape().as_list()[:2], strides=strides,
                     kernel=kernel_pool, padding='SAME', use_bias=False)
    if use_bias:
        if bias is None:
            with tf.variable_scope('conv_pool2d'):
                bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
                outputs = tf.nn.bias_add(outputs, bias)
        else:
            bias_shape = [filters]
            if bias_shape != bias.get_shape().as_list():
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s" %
                                 (tuple(bias_shape), tuple(bias.get_shape().as_list())))
    return outputs


def lrelu(x, alpha):
    """
    Leaky ReLU activation function

    Reference:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py
    """
    with tf.name_scope("lrelu"):
        return tf.maximum(alpha * x, x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[-1]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=list(range(len(input.get_shape()) - 1)), keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def instancenorm(input):
    with tf.variable_scope("instancenorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[-1]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=list(range(1, len(input.get_shape()) - 1)), keep_dims=True)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                               variance_epsilon=variance_epsilon)
        return normalized


def flatten(input, axis=1, end_axis=-1):
    """
    Caffe-style flatten.

    Args:
        inputs: An N-D tensor.
        axis: The first axis to flatten: all preceding axes are retained in the output.
            May be negative to index from the end (e.g., -1 for the last axis).
        end_axis: The last axis to flatten: all following axes are retained in the output.
            May be negative to index from the end (e.g., the default -1 for the last
            axis)

    Returns:
        A M-D tensor where M = N - (end_axis - axis)
    """
    input_shape = input.shape
    input_rank = input_shape.ndims
    if axis < 0:
        axis = input_rank + axis
    if end_axis < 0:
        end_axis = input_rank + end_axis
    output_shape = []
    if axis != 0:
        output_shape.append(input_shape[:axis])
    output_shape.append([tf.reduce_prod(input_shape[axis:end_axis + 1])])
    if end_axis + 1 != input_rank:
        output_shape.append(input_shape[end_axis + 1:])
    output_shape = tf.concat(output_shape, axis=0)
    output = tf.reshape(input, output_shape)
    return output


def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy
