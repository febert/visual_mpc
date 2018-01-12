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

    print 'using shuffle: ', shuffle

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(serialized_example):
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
                features[robot_pos_name] = tf.FixedLenFeature([conf['test_metric']['robot_pos'] * 2], tf.int64)
                features[object_pos_name] = tf.FixedLenFeature([conf['test_metric']['object_pos'] * 2], tf.int64)

            features = tf.parse_single_example(serialized_example, features=features)

            COLOR_CHAN = 3
            if '128x128' in conf:
                ORIGINAL_WIDTH = 128
                ORIGINAL_HEIGHT = 128
            else:
                ORIGINAL_WIDTH = 64
                ORIGINAL_HEIGHT = 64

            if 'row_start' in conf:
                IMG_HEIGHT = conf['row_end'] - conf['row_start']
            else:
                IMG_HEIGHT = 64

            if 'img_width' in conf:
                IMG_WIDTH = conf['img_width']
            else:
                IMG_WIDTH = 64

            image = tf.decode_raw(features[image_name], tf.uint8)
            image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
            image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
            image = image[conf['row_start']:conf['row_end']]
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

                object_pos = tf.reshape(features[object_pos_name], shape=[1, conf['test_metric']['object_pos'], 2])
                object_pos_seq.append(object_pos)

        image_seq = tf.concat(values=image_seq, axis=0)

        endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
        action_seq = tf.concat(action_seq, 0)

        return {"images": image_seq, "endeffector_pos": endeffector_pos_seq, "actions": action_seq}

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(conf['batch_size'])
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    return next_element['images'], next_element['actions'], next_element['endeffector_pos']


def main():
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = '/'.join(str.split(current_dir, '/')[:-2]) + '/pushing_data/cartgripper/train'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 15 #48      # 'sequence length, including context frames.'
    conf['batch_size']= 10
    conf['visualize']= True
    conf['context_frames'] = 2

    conf['row_start'] = 15
    conf['row_end'] = 63
    conf['img_width'] = 64
    conf['sdim'] = 6
    conf['adim'] = 3

    # conf['color_augmentation'] = ''

    # conf['test_metric'] = {'robot_pos': 1, 'object_pos': 2}

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'testing the reader'

    dict = build_tfrecord_input(conf, training=True)
    # image_batch, action_batch, endeff_pos_batch, robot_pos_batch, object_pos_batch = build_tfrecord_input(conf, training=True)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video

    for i_run in range(10):
        print 'run number ', i_run

        images, actions, endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos']])
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