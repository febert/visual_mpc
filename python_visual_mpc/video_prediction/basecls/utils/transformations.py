import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

RELU_SHIFT = 1e-12

def dna_transformation(conf, prev_image, dna_input):
    """Apply dynamic neural advection to previous image.

    Args:
      prev_image: previous image to be transformed.
      dna_input: hidden lyaer to be used for computing DNA transformation.
    Returns:
      List of images transformed by the predicted CDNA kernels.
    """
    # Construct translated images.
    DNA_KERN_SIZE = conf['kern_size']

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
    inputs = tf.concat(axis=3, values=inputs)

    # Normalize channels to 1.
    kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
    kernel = tf.expand_dims(
        kernel / tf.reduce_sum(
            kernel, [3], keep_dims=True), [4])

    return tf.reduce_sum(kernel * inputs, [3], keep_dims=False), kernel


def cdna_transformation(conf, prev_image, cdna_input, reuse_sc=None, scope=None):
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
    height = int(prev_image.get_shape()[1])
    width = int(prev_image.get_shape()[2])

    DNA_KERN_SIZE = conf['kern_size']
    num_masks = conf['num_masks']
    color_channels = int(prev_image.get_shape()[3])

    if scope == None:
        scope = 'cdna_params'
    # Predict kernels using linear function of last hidden layer.
    cdna_kerns = slim.layers.fully_connected(
        cdna_input,
        DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
        scope=scope,
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

    return transformed, cdna_kerns


## Utility functions
def stp_transformation(conf, prev_image, stp_input, num_masks, reuse= None, suffix = None):
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


from python_visual_mpc.video_prediction.basecls.ops.ops import local2d
def apply_kernel(image, kernel):
    """
    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernel: A 5-D of shape
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1]]`.

    Returns:
        A 4-D tensor.
    """
    output = image
    output = tf.split(output, output.get_shape().as_list()[-1], axis=3)
    kernel_size = kernel.get_shape().as_list()[-2:]
    kernel = kernel[..., None, None]
    # TODO: 'SYMMETRIC' padding
    output = [local2d(o, 1, kernel_size=kernel_size, kernel=kernel) for o in output]
    output = tf.concat(output, axis=3)
    return output

def sep_dna_transformation(conf, prev_image, dna_input):
    """
    :param conf:
    :param prev_image:
    :param dna_input: batchsize, H, W, kernsize*2
    :return:
    """
    h_kernels, v_kernels = tf.split(dna_input, 2, axis=3)
    h_kernels = normalize_kernels(h_kernels)
    v_kernels = normalize_kernels(v_kernels)

    output0 = apply_kernel(prev_image, h_kernels[..., :, None])
    output1 = apply_kernel(output0, v_kernels[..., None, :])

    return output1


def normalize_kernels(kernels, kernel_normalization="softmax"):

    if kernel_normalization == "softmax":
        kernels = tf.nn.softmax(kernels, dim=3)
    elif kernel_normalization == "relu":
        kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
        kernels = kernels / tf.reduce_sum(kernels, axis=3, keep_dims=True)
    else:
        raise ValueError
    return kernels

def sep_cdna_transformation(conf, prev_image, cdna_input, reuse_sc=None, scope=None):
    ##TODO: sep cdna
    pass
    # return transformed, cdna_kerns