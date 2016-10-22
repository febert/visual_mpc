import os
import tensorflow as tf
import numpy as np


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def save_tf_record(X_full, Xdot_full, U, sample_images, dir, filename):
    """
    saves data from one sample trajectory into one tf-record file
    """

    X_Xdot = np.concatenate((X_full, Xdot_full), axis=1)
    sequence_length = X_Xdot.shape[0]

    if sample_images.shape[0] != sequence_length:
        raise ValueError('Number of Images %d does not match number of labels %d.' %
                         (sample_images.shape[0], sequence_length))

    U = U.astype(np.float32)
    X_Xdot = X_Xdot.astype(np.float32)


    filename = os.path.join(dir, filename + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    feature = {}
    for index in range(sequence_length):
        image_raw = sample_images[index].tostring()

        feature['move/' + str(index) + '/action']= _float_feature(U[index,:].tolist())
        feature['move/' + str(index) + '/state'] = _float_feature(X_Xdot[index,:].tolist())
        feature['move/' + str(index) + '/image/encoded'] = _bytes_feature(image_raw)

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()


    # filename = os.path.join(dir, filename + '.tfrecords')
    # print('Writing', filename)
    # writer = tf.python_io.TFRecordWriter(filename)
    # for index in range(sequence_length):
    #     image_raw = sample_images[index].tostring()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         # 'move/' + str(index) + '/action': _float_feature(U[index,:].tolist()),
    #         # 'move/' + str(index) + '/action': _float_feature([1.0]),
    #         'action0': _float_feature(U[index,:].tolist()),
    #         'action1': _float_feature(U[index,:].tolist())
    #         # 'move/' + str(index) + '/state': _float_feature(X_Xdot[index,:].tolist()),
    #         # 'move/' + str(index) + '/image/encoded': _bytes_feature(image_raw)
    #     }))
    #     print 'index:', index
    #     writer.write(example.SerializeToString())
    # writer.close()