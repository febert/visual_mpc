import tensorflow as tf
import numpy as np


def get_coords(img_shape):
    """
    returns coordinate grid corresponding to identity appearance flow
    :param img_shape:
    :return:
    """
    y = tf.cast(tf.range(img_shape[1]), tf.float32)
    x = tf.cast(tf.range(img_shape[2]), tf.float32)
    batch_size = img_shape[0]

    X, Y = tf.meshgrid(x, y)
    coords = tf.expand_dims(tf.stack((X, Y), axis=2), axis=0)
    coords = tf.tile(coords, [batch_size, 1, 1, 1])
    return coords

def resample_layer(src_img, warp_pts, name="tgt_img"):
    with tf.variable_scope(name):
        return tf.contrib.resampler.resampler(src_img, warp_pts)

def warp_pts_layer(flow_field, name="warp_pts"):
    with tf.variable_scope(name):
        img_shape = flow_field.get_shape().as_list()
        return flow_field + get_coords(img_shape)

def apply_warp(I0, flow_field):
    warp_pts = warp_pts_layer(flow_field)
    return resample_layer(I0, warp_pts), warp_pts

def conv2d(nout, kernel_size=np.array([3,3]), activation=tf.nn.relu):
    func = tf.keras.layers.Conv2D(nout, kernel_size, activation=activation, padding='same')
    return func


def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

def mean_square(x):
    return tf.reduce_mean(tf.square(x))



def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)

        return tf.reduce_sum(error) / (normalization + 1e-6)


def flow_smooth_cost(flow, norm, mode, mask):
    """
    computes the norms of the derivatives and averages over the image
    :param flow_field:

    :return:
    """
    if mode == '2nd':  # compute 2nd derivative
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

    elif mode == 'sobel':
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # sobel filter
        filter_y = np.transpose(filter_x)
        weight_array = np.ones([3, 3, 1, 2])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y

    weights = tf.constant(weight_array, dtype=tf.float32)

    flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
    delta_u =  tf.nn.conv2d(flow_u, weights, strides=[1, 1, 1, 1], padding='SAME')
    delta_v =  tf.nn.conv2d(flow_v, weights, strides=[1, 1, 1, 1], padding='SAME')

    deltas = tf.concat([delta_u, delta_v], axis=3)

    return norm(deltas, mask)
