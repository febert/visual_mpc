import tensorflow as tf

def local2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, flip_filters=False):
    """
    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernel: A 6-D or 7-D of shape
            `[in_height, in_width, kernel_size[0], kernel_size[1], filters, in_channels]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], filters, in_channels]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if strides != [1, 1]:
        raise NotImplementedError
    if padding != 'SAME':
        raise NotImplementedError
    input_shape = inputs.get_shape().as_list()  # N H W C
    output_shape = input_shape[:3] + [filters]  # N H W F
    kernel_shape = output_shape[1:3] + kernel_size + [filters, input_shape[-1]]  # H W K K F C
    if kernel is None:
        with tf.variable_scope('local2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        assert kernel_shape == kernel.get_shape().as_list() or kernel_shape == kernel.get_shape().as_list()[1:]

    # start with ii == jj == 0 case to initialize tensor
    i = kernel_size[0] // 2
    j = kernel_size[1] // 2
    filter_h_ind = -i-1 if flip_filters else i
    filter_w_ind = -j-1 if flip_filters else j
    outputs = tf.reduce_sum(inputs[:, :, :, None, :] * kernel[..., filter_h_ind, filter_w_ind, :, :], axis=-1)

    for i in range(kernel_size[0]):
        filter_h_ind = -i-1 if flip_filters else i
        ii = i - (kernel_size[0] // 2)
        input_h_slice = slice(
            max(ii, 0), min(ii + output_shape[1], output_shape[1]))
        output_h_slice = slice(
            max(-ii, 0), min(-ii + output_shape[1], output_shape[1]))

        for j in range(kernel_size[1]):
            filter_w_ind = -j-1 if flip_filters else j
            jj = j - (kernel_size[1] // 2)
            input_w_slice = slice(
                max(jj, 0), min(jj + output_shape[2], output_shape[2]))
            output_w_slice = slice(
                max(-jj, 0), min(-jj + output_shape[2], output_shape[2]))
            # skip this case since it was done at the beginning
            if ii == jj == 0:
                continue
            inc = tf.reduce_sum(inputs[:, input_h_slice, input_w_slice, None, :] *
                                kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :, :], axis=-1)
            # equivalent to this
            # outputs[:, output_h_slice, output_w_slice, :] += inc
            paddings = [[0, 0], [output_h_slice.start, output_shape[1] - output_h_slice.stop],
                        [output_w_slice.start, output_shape[2] - output_w_slice.stop], [0, 0]]
            outputs += tf.pad(inc, paddings)
    return outputs