import os
import tensorflow as tf
import numpy as np


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def save_tf_record(dir, filename, trajectory_list):
    """
    saves data_files from one sample trajectory into one tf-record file
    """

    filename = os.path.join(dir, filename + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for traj in range(len(trajectory_list)):

        [X_Xdot, U, sample_images] = trajectory_list[traj]
        sequence_length = sample_images.shape[0]

        for index in range(sequence_length):
            image_raw = sample_images[index].tostring()

            feature['move/' + str(index) + '/action']= _float_feature(U[index,:].tolist())
            feature['move/' + str(index) + '/state'] = _float_feature(X_Xdot[index,:].tolist())
            feature['move/' + str(index) + '/image/encoded'] = _bytes_feature(image_raw)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()