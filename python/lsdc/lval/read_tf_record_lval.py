import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import pdb

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
OBJECT_POS_DIM = 8


def build_tfrecord_input(conf, training=True):
    """Create input tfrecord tensors.

    Args:
      training: training or validation data_files.
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
        raise RuntimeError('No data_files files found.')

    if conf['visualize']:
        shuffle = False
        print 'visualizenig, using data form:  ', conf['data_dir']
    else: shuffle = True

    index = int(np.floor(conf['train_val_split'] * len(filenames)))
    if training:
        filenames = filenames[:index]
    else:
        filenames = filenames[index:]



    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_name = 'img'
    score_name = 'score'
    goalpos_name = 'goalpos'
    desig_pos_name = 'desig_pos'
    # init_state_name = 'init_state'

    features = {
                image_name: tf.FixedLenFeature([1], tf.string),
                score_name: tf.FixedLenFeature([1], tf.float32),
                goalpos_name: tf.FixedLenFeature([2], tf.float32),
                desig_pos_name: tf.FixedLenFeature([2], tf.float32),
                # init_state_name: tf.FixedLenFeature([2], tf.float32)
    }

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

    if conf['visualize']:
        num_threads = 1
    else:
        num_threads = np.min((conf['batch_size'], 32))

    score = features[score_name]
    goalpos = features[goalpos_name]
    desig_pos = features[desig_pos_name]
    # init_state = features[init_state_name]


    [image_batch, score_batch, goalpos_batch, desig_pos_batch] = tf.train.batch(
                                                    [image, score, goalpos, desig_pos],
                                                    conf['batch_size'],
                                                    num_threads=num_threads,
                                                    capacity=100 * conf['batch_size'])
    # [image_batch, score_batch, goalpos_batch, desig_pos_batch, init_state_batch] = tf.train.batch(
    #                                                 [image, score, goalpos, desig_pos, init_state],
    #                                                 conf['batch_size'],
    #                                                 num_threads=num_threads,
    #                                                 capacity=100 * conf['batch_size'])

    return image_batch, score_batch, goalpos_batch, desig_pos_batch



##### code below is used for debugging

def mujoco_to_imagespace(mujoco_coord, numpix=64):
    viewer_distance = .75  # distance from camera to the viewing plane
    window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
    pixelheight = window_height / numpix  # height of one pixel
    pixelwidth = pixelheight
    window_width = pixelwidth * numpix
    middle_pixel = numpix / 2
    pixel_coord = np.array([-mujoco_coord[1], mujoco_coord[0]])/pixelwidth + \
                  np.array([middle_pixel, middle_pixel])
    return pixel_coord

if __name__ == '__main__':
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    DATA_DIR = '/home/frederik/Documents/lsdc/experiments/val_exp/dna_mpc_states/train'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['batch_size']= 13
    conf['visualize']=False
    conf['use_object_pos'] = True

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'testing the reader'

    image_batch, score_batch, goalpos_batch, desig_pos_batch, init_states_batch  = build_tfrecord_input(conf, training=True)
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())


    for i in range(2):
        print '-------------------'
        print 'run number ', i

        image, score, goalpos, desig_pos, init_states = sess.run([image_batch,
                                                                  score_batch,
                                                                  goalpos_batch,
                                                                  desig_pos_batch,
                                                                  init_states_batch])

        print 'image shape', image.shape
        print 'score shape', score.shape
        print 'scores', score
        print 'goalpos shape', goalpos.shape
        print 'goalpos', goalpos
        print 'desigpos shape', desig_pos.shape
        print 'desigpos', desig_pos
        print 'init_states shape', init_states.shape
        print 'init_states', init_states

        image = image[i].squeeze()
        desig_pos = desig_pos[i].squeeze()
        desig_pos_pix = mujoco_to_imagespace(desig_pos.squeeze())
        image[int(desig_pos_pix[0]), int(desig_pos_pix[1])] = 1
        image = Image.fromarray(np.uint8(image * 255)).show()