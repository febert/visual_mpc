import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_integer('sequence_length', 30, 'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1, 'Whether or not to give the state+action to the model')
flags.DEFINE_integer('batch_size', 4, 'batch size for training')
flags.DEFINE_float('train_val_split', 1,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

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

# DATA_DIR = '/home/frederik/Documents/pushing_data/tfrecords'
DATA_DIR = '/media/frederik/UBUNTU 14_0/pushing_data/tfrecords'

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')


def build_tfrecord_input(training=True):
    """Create input tfrecord tensors.

    Args:
      training: training or validation data.
    Returns:
      list of tensors corresponding to images, actions, and states. The images
      tensor is 5D, batch x time x height x width x channels. The state and
      action tensors are 3D, batch x time x dimension.
    Raises:
      RuntimeError: if no files found.
    """
    filenames = gfile.Glob(os.path.join(FLAGS.data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')
    index = int(np.floor(FLAGS.train_val_split * len(filenames)))
    if training:
        filenames = filenames[:index]
    else:
        filenames = filenames[index:]

    # import pdb; pdb.set_trace()

    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq, state_seq, action_seq = [], [], []

    for i in range(FLAGS.sequence_length):
        image_name = 'move/' + str(i) + '/image/encoded'
        action_name = 'move/' + str(i) + '/action'
        state_name = 'move/' + str(i) + '/state'
        # print 'reading index', i
        if FLAGS.use_state:
            features = {
                        image_name: tf.FixedLenFeature([1], tf.string),
                        action_name: tf.FixedLenFeature([ACION_DIM], tf.float32),
                        state_name: tf.FixedLenFeature([STATE_DIM], tf.float32)
            }
        else:
            features = {image_name: tf.FixedLenFeature([1], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)


        # image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)

        # print features[image_name].getshape()
        # features[image_name].getshape()
        image = tf.decode_raw(features[image_name], tf.uint8)
        image = tf.reshape(image, shape=[1,ORIGINAL_HEIGHT*ORIGINAL_WIDTH*COLOR_CHAN])
        image = tf.reshape(image, shape=[1,ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

        if IMG_HEIGHT != IMG_WIDTH:
            raise ValueError('Unequal height and width unsupported')

        crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
        # image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        # image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
        # image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.cast(image, tf.float32) / 255.0
        image_seq.append(image)

        if FLAGS.use_state:
            state = tf.reshape(features[state_name], shape=[1, STATE_DIM])
            state_seq.append(state)
            action = tf.reshape(features[action_name], shape=[1, ACION_DIM])
            action_seq.append(action)

    image_seq = tf.concat(0, image_seq)


    if FLAGS.use_state:
        state_seq = tf.concat(0, state_seq)
        action_seq = tf.concat(0, action_seq)
        [image_batch, action_batch, state_batch] = tf.train.batch(
            [image_seq, action_seq, state_seq],
            FLAGS.batch_size,
            num_threads=FLAGS.batch_size,
            capacity=100 * FLAGS.batch_size)
        return image_batch, action_batch, state_batch
    else:
        image_batch = tf.train.batch(
            [image_seq],
            FLAGS.batch_size,
            num_threads=FLAGS.batch_size,
            capacity=100 * FLAGS.batch_size)
        zeros_batch_action = tf.zeros([FLAGS.batch_size, FLAGS.sequence_length, ACION_DIM])
        zeros_batch_state = tf.zeros([FLAGS.batch_size, FLAGS.sequence_length, STATE_DIM])
        return image_batch, zeros_batch_action, zeros_batch_state


if __name__ == '__main__':

    print 'testing the reader'
    image_batch, action_batch, state_batch  = build_tfrecord_input(training=True)
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    for i in range(1):
        print 'run number ', i
        image_data, action_data, state_data = sess.run([image_batch, action_batch, state_batch])

        print action_data.shape
        print image_data.shape
        print state_data.shape

        # print image_data[0,0]
        for i in range(4):
            for j in range(1):
                print action_data[i]
                img = image_data[i,j]*255.0
                # print img
                img = img.astype(np.uint8)
                # print img

                img = Image.fromarray(img, 'RGB')
                img.show()