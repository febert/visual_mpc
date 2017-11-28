import numpy as np
import tensorflow as tf


def compute_motion_vector_cdna(conf, cdna_kerns):
    """
    :param conf:
    :param cdna_kerns:  (H,W,N,C)
    :return:
    """
    range = conf['kern_size'] / 2
    dc = np.linspace(-range, range, num=conf['kern_size'])
    dc = np.expand_dims(dc, axis=0)
    dc = np.repeat(dc, conf['kern_size'], axis=0)
    dr = np.transpose(dc)
    dr = tf.constant(dr, dtype=cdna_kerns.dtype)
    dc = tf.constant(dc, dtype=cdna_kerns.dtype)

    cdna_kerns = tf.transpose(cdna_kerns, [2, 3, 0, 1])
    cdna_kerns = tf.split(cdna_kerns, conf['num_masks'], axis=1)
    cdna_kerns = [tf.squeeze(k) for k in cdna_kerns]

    vecs = []
    for kern in cdna_kerns:
        vec_r = tf.multiply(dr, kern)
        vec_r = tf.reduce_sum(vec_r, axis=[1, 2])
        vec_c = tf.multiply(dc, kern)
        vec_c = tf.reduce_sum(vec_c, axis=[1, 2])

        vecs.append(tf.stack([vec_r, vec_c], axis=1))
    return vecs



def compute_motion_vector_dna(conf, dna_kerns):

    range = conf['kern_size'] / 2
    dc = np.linspace(-range, range, num= conf['kern_size'])
    dc = np.expand_dims(dc, axis=0)
    dc = np.repeat(dc, conf['kern_size'], axis=0)

    dc = dc.reshape([1,1,conf['kern_size'],conf['kern_size']])
    dc = np.repeat(dc, 64, axis=0)
    dc = np.repeat(dc, 64, axis=1)

    dr = np.transpose(dc, [0,1,3,2])

    dr = tf.constant(dr, dtype=tf.float32)
    dc = tf.constant(dc, dtype=tf.float32)

    dna_kerns = tf.reshape(dna_kerns, [conf['batch_size'], 64,64,conf['kern_size'],conf['kern_size']])

    dr = tf.expand_dims(dr, axis=0)
    dc = tf.expand_dims(dc, axis=0)

    vec_r = tf.multiply(dr, dna_kerns)
    vec_r = tf.reduce_sum(vec_r, axis=[3,4])
    vec_c = tf.multiply(dc, dna_kerns)
    vec_c = tf.reduce_sum(vec_c, axis=[3,4])

    vec_c = tf.expand_dims(vec_c, axis=-1)
    vec_r = tf.expand_dims(vec_r, axis=-1)

    flow = tf.concat([vec_r, vec_c], axis=-1)  # size: [conf['batch_size'], 64, 64, 2]

    return [flow]