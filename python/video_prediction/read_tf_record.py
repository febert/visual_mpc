import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from PIL import Image
import imp

# Original image dimensions
ORIGINAL_WIDTH = 64
ORIGINAL_HEIGHT = 64
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Dimension of the state and action.
STATE_DIM = 4
ACION_DIM = 2


def build_tfrecord_input(conf, training=True):
    """Create input tfrecord tensors.

    Args:
      training: training or validation data.
      conf: A dictionary containing the configuration for the experiment
    Returns:
      list of tensors corresponding to images, actions, and states. The images
      tensor is 5D, batch x time x height x width x channels. The state and
      action tensors are 3D, batch x time x dimension.
    Raises:
      RuntimeError: if no files found.
    """
    filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    index = int(np.floor(conf['train_val_split'] * len(filenames)))
    if training:
        filenames = filenames[:index]
    else:
        filenames = filenames[index:]

    if conf['visualize']:
        filenames = [conf['visual_file']]
        print 'using input file', filenames
        shuffle = False
    else: shuffle = True

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq, state_seq, action_seq = [], [], []

    load_indx = range(0, 29, conf['skip_frame'])
    load_indx = load_indx[:conf['sequence_length']]
    print 'using frame sequence: ', load_indx

    for i in load_indx:
        image_name = 'move/' + str(i) + '/image/encoded'
        action_name = 'move/' + str(i) + '/action'
        state_name = 'move/' + str(i) + '/state'
        # print 'reading index', i
        if conf['use_state']:
            features = {
                        image_name: tf.FixedLenFeature([1], tf.string),
                        action_name: tf.FixedLenFeature([ACION_DIM], tf.float32),
                        state_name: tf.FixedLenFeature([STATE_DIM], tf.float32)
            }
        else:
            features = {image_name: tf.FixedLenFeature([1], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)

        image = tf.decode_raw(features[image_name], tf.uint8)
        image = tf.reshape(image, shape=[1,ORIGINAL_HEIGHT*ORIGINAL_WIDTH*COLOR_CHAN])
        image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

        if IMG_HEIGHT != IMG_WIDTH:
            raise ValueError('Unequal height and width unsupported')

        crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
        image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.cast(image, tf.float32) / 255.0
        image_seq.append(image)

        if conf['use_state']:
            state = tf.reshape(features[state_name], shape=[1, STATE_DIM])
            state_seq.append(state)
            action = tf.reshape(features[action_name], shape=[1, ACION_DIM])
            action_seq.append(action)

    image_seq = tf.concat(0, image_seq)

    if conf['visualize']: num_threads = 1
    else: num_threads = conf['batch_size']

    if conf['use_state']:
        state_seq = tf.concat(0, state_seq)
        action_seq = tf.concat(0, action_seq)
        [image_batch, action_batch, state_batch] = tf.train.batch(
            [image_seq, action_seq, state_seq],
            conf['batch_size'],
            num_threads= num_threads,
            capacity=100 * conf['batch_size'])
        return image_batch, action_batch, state_batch
    else:
        image_batch = tf.train.batch(
            [image_seq],
            conf['batch_size'],
            num_threads=num_threads,
            capacity=100 * conf['batch_size'])
        zeros_batch_action = tf.zeros([conf['batch_size'], conf['sequence_length'], ACION_DIM])
        zeros_batch_state = tf.zeros([conf['batch_size'], conf['sequence_length'], STATE_DIM])
        return image_batch, zeros_batch_action, zeros_batch_state


if __name__ == '__main__':
    # for debugging only:

    confpath = '/home/frederik/Documents/lsdc/tensorflow_data/downsized_model_skip2/conf.py'
    hyperparams = imp.load_source('hyperparams', confpath)
    conf = hyperparams.configuration

    DATA_DIR = '/home/frederik/Documents/pushing_data/test'
    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data.' ,
    conf['visualize'] = 'True'
    conf['visual_file'] = DATA_DIR + '/traj_66304_to_66559.tfrecords'
    conf['skip_frame'] = 1
    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'testing the reader'
    image_batch, action_batch, state_batch  = build_tfrecord_input(conf, training=True)
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())


    for i in range(1):
        print 'run number ', i
        image_data, action_data, state_data = sess.run([image_batch, action_batch, state_batch])

        print action_data.shape
        print image_data.shape
        print state_data.shape


        for i in range(2):
            img = np.uint8(255. *image_data[i,0])
            img = Image.fromarray(img, 'RGB')
            img.show()
