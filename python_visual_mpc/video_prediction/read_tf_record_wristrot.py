import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import pdb

from PIL import Image
import imp

import cPickle


def build_tfrecord_input(conf, training=True, input_file=None):
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

    if 'sdim' in conf:
        sdim = conf['sdim']
    else: sdim = 3
    if 'adim' in conf:
        adim = conf['adim']
    else: adim = 4
    print 'adim', adim
    print 'sdim', sdim

    if input_file is not None:
        filenames = [input_file]
        shuffle = False
    else:
        filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
        if not filenames:
            raise RuntimeError('No data_files files found.')

        index = int(np.floor(conf['train_val_split'] * len(filenames)))
        if training:
            filenames = filenames[:index]
        else:
            filenames = filenames[index:]

        if conf['visualize']:
            filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
            shuffle = False
        else:
            shuffle = True

    print 'using shuffle files: ', shuffle

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq, image_main_seq, endeffector_pos_seq, action_seq, object_pos_seq, robot_pos_seq = [], [], [], [], [], []

    load_indx = range(0, conf['sequence_length'], conf['skip_frame'])
    print 'using frame sequence: ', load_indx

    rand_h = tf.random_uniform([1], minval=-0.2, maxval=0.2)
    rand_s = tf.random_uniform([1], minval=-0.2, maxval=0.2)
    rand_v = tf.random_uniform([1], minval=-0.2, maxval=0.2)

    for i in load_indx:

        image_name = str(i) + '/image_view0/encoded'
        action_name = str(i) + '/action'
        endeffector_pos_name = str(i) + '/endeffector_pos'

        features = {

                    image_name: tf.FixedLenFeature([1], tf.string),
                    action_name: tf.FixedLenFeature([adim], tf.float32),
                    endeffector_pos_name: tf.FixedLenFeature([sdim], tf.float32),
        }

        if 'test_metric' in conf:
            robot_pos_name = str(i) + '/robot_pos'
            object_pos_name = str(i) + '/object_pos'
            features[robot_pos_name] = tf.FixedLenFeature([conf['test_metric']['robot_pos']*2], tf.int64)
            features[object_pos_name] = tf.FixedLenFeature([conf['test_metric']['object_pos']*2], tf.int64)

        features = tf.parse_single_example(serialized_example, features=features)

        COLOR_CHAN = 3
        if '128x128' in conf:
            ORIGINAL_WIDTH = 128
            ORIGINAL_HEIGHT = 128
        else:
            ORIGINAL_WIDTH = 64
            ORIGINAL_HEIGHT = 64

        if 'im_height' in conf:
            IMG_HEIGHT = conf['im_height']
        else: IMG_HEIGHT = 64

        if 'im_width' in conf:
            IMG_WIDTH = conf['im_width']
        else: IMG_WIDTH = 64

        image = tf.decode_raw(features[image_name], tf.uint8)
        image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
        image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
        image = image[0:IMG_HEIGHT]
        image = tf.reshape(image, [1, IMG_HEIGHT, IMG_WIDTH, COLOR_CHAN])
        image = tf.cast(image, tf.float32) / 255.0

        if 'color_augmentation' in conf:
            # print 'performing color augmentation'
            image_hsv = tf.image.rgb_to_hsv(image)
            img_stack = [tf.unstack(imag, axis=2) for imag in tf.unstack(image_hsv, axis=0)]
            stack_mod = [tf.stack([x[0] + rand_h,
                                   x[1] + rand_s,
                                   x[2] + rand_v]
                                  , axis=2) for x in img_stack]

            image_rgb = tf.image.hsv_to_rgb(tf.stack(stack_mod))
            image = tf.clip_by_value(image_rgb, 0.0, 1.0)

        image_seq.append(image)

        endeffector_pos = tf.reshape(features[endeffector_pos_name], shape=[1, sdim])
        endeffector_pos_seq.append(endeffector_pos)
        action = tf.reshape(features[action_name], shape=[1, adim])
        action_seq.append(action)

        if 'test_metric' in conf:
            robot_pos = tf.reshape(features[robot_pos_name], shape=[1, 2])
            robot_pos_seq.append(robot_pos)

            object_pos = tf.reshape(features[object_pos_name], shape=[1, conf['test_metric']['object_pos'],2])
            object_pos_seq.append(object_pos)

    image_seq = tf.concat(values=image_seq, axis=0)

    if conf['visualize']: num_threads = 1
    else: num_threads = np.min((conf['batch_size'], 32))

    endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
    action_seq = tf.concat(action_seq, 0)

    if 'test_metric' in conf:
        robot_pos_seq = tf.concat(robot_pos_seq, 0)
        object_pos_seq = tf.concat(object_pos_seq, 0)

    if 'test_metric' in conf:
        [image_batch, action_batch, endeffector_pos_batch, robot_pos_batch, object_pos_batch] = tf.train.batch(
            [image_seq, action_seq, endeffector_pos_seq, robot_pos_seq, object_pos_seq],
            conf['batch_size'],
            num_threads=num_threads,
            capacity=100 * conf['batch_size'])
        return image_batch, action_batch, endeffector_pos_batch, robot_pos_batch, object_pos_batch
    else:
        [image_batch, action_batch, endeffector_pos_batch] = tf.train.batch(
            [image_seq, action_seq, endeffector_pos_seq],
            conf['batch_size'],
            num_threads=num_threads,
            capacity=100 * conf['batch_size'])
        return image_batch, action_batch, endeffector_pos_batch



##### code below is used for debugging


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_tf_records(images, actions, states, filepath, init_pix_distrib, init_pix_pos):
    filename = os.path.join(dir, filepath + '/tfrecords/canon_examples.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for ex in range(images.shape[0]):
        sequence_length = 15

        for tind in range(sequence_length):
            image_raw = (images[ex,tind]*255.).astype(np.uint8)
            image_raw = image_raw.tostring()

            feature[str(tind) + '/action'] = _float_feature(actions[ex, tind].tolist())
            feature[str(tind) + '/endeffector_pos'] = _float_feature(states[ex,tind].tolist())
            feature[str(tind) + '/image_aux1/encoded'] = _bytes_feature(image_raw)

            pix_raw = (init_pix_distrib[ex][tind]*255.).astype(np.uint8)
            pix_raw = pix_raw.tostring()
            feature[str(tind) + '/init_pix_distrib'] = _bytes_feature(pix_raw)

            if tind == 0:
                feature['/init_pix_pos'] = _float_feature(init_pix_pos[ex].tolist())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def visualize_annotation(conf, images, robot_pos, object_pos):

    for t in range(conf['sequence_length']):
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.set_xlim(0, 63)
        ax.set_ylim(63, 0)

        plt.imshow(images[t])

        ax.scatter(robot_pos[t][1], robot_pos[t][0], s=20, marker="D", facecolors='r')

        # object_pos is t, i_object, row_col
        ax.scatter(object_pos[t][0][1], object_pos[t][0][0], s=20, marker="D", facecolors='b')
        ax.scatter(object_pos[t][1][1], object_pos[t][1][0], s=20, marker="D", facecolors='b')
        plt.show()


def main():
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # DATA_DIR = '/'.join(str.split(current_dir, '/')[:-2]) + '/pushing_data/softmotion30_v1/train'
    DATA_DIR = '/'.join(str.split(current_dir, '/')[:-2]) + '/pushing_data/weiss_gripper/train'
    # DATA_DIR = '/'.join(str.split(current_dir, '/')[:-2]) + '/pushing_data/wristrot_test_newobj/test_annotations'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 48      # 'sequence length, including context frames.'
    conf['batch_size']= 10
    conf['visualize']= True
    conf['context_frames'] = 2

    conf['im_height'] = 54
    conf['im_width'] = 64
    conf['sdim'] = 4
    conf['adim'] = 5

    conf['color_augmentation'] = ''

    # conf['test_metric'] = {'robot_pos': 1, 'object_pos': 2}

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'testing the reader'

    image_batch, action_batch, endeff_pos_batch = build_tfrecord_input(conf, training=True)
    # image_batch, action_batch, endeff_pos_batch, robot_pos_batch, object_pos_batch = build_tfrecord_input(conf, training=True)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video

    for i_run in range(1):
        print 'run number ', i_run

        images, actions, endeff = sess.run([image_batch, action_batch, endeff_pos_batch])
        # images, actions, endeff, robot_pos, object_pos = sess.run([image_batch, action_batch, endeff_pos_batch, robot_pos_batch, object_pos_batch])

        file_path = '/'.join(str.split(DATA_DIR, '/')[:-1]+['preview'])
        comp_single_video(file_path, images, num_exp=conf['batch_size'])

        # show some frames
        for b in range(conf['batch_size']):

            print 'actions'
            print actions[b]

            print 'endeff'
            print endeff[b]

            print 'video mean brightness', np.mean(images[b])
            if np.mean(images[b]) < 0.25:
                print b
                plt.imshow(images[b,0])
                plt.show()

            # print 'robot_pos'
            # print robot_pos
            #
            # print 'object_pos'
            # print object_pos

            # visualize_annotation(conf, images[b], robot_pos[b], object_pos[b])

if __name__ == '__main__':
    main()