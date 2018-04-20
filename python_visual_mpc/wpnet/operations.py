import tensorflow as tf

def lrelu(x, alpha):
    """
    Leaky ReLU activation function

    Reference:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py
    """
    with tf.name_scope("lrelu"):
        return tf.maximum(alpha * x, x)
