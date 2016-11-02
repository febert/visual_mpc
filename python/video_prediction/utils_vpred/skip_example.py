import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def skip_example(*args):
    print 'skipping every second example in every batch !!'
    res = []
    for arg in args:

        indices = np.zeros((FLAGS.batch_size, 15, 2))
        for i in range(32):
            for j in range(15):
                indices[i, j] = np.array([i, j * 2])

        indices = np.int64(indices)
        arg = tf.gather_nd(arg, indices)

        res.append(arg)

    return res