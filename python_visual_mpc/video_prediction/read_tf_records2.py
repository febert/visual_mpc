import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from python_visual_mpc.utils.txt_in_image import draw_text_image
import pdb
import time
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import assemble_gif
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import npy_to_gif
from PIL import Image
import imp

import pickle
from random import shuffle as shuffle_list
from python_visual_mpc.misc.zip_equal import zip_equal
import copy
COLOR_CHAN = 3
def decode_im(conf, features, image_name):

    if 'orig_size' in conf:
        ORIGINAL_HEIGHT = conf['orig_size'][0]
        ORIGINAL_WIDTH = conf['orig_size'][1]
    else:
        ORIGINAL_WIDTH = 64
        ORIGINAL_HEIGHT = 64
    if 'row_start' in conf:
        IMG_HEIGHT = conf['row_end'] - conf['row_start']
    else:
        IMG_HEIGHT = ORIGINAL_HEIGHT
    if 'img_width' in conf:
        IMG_WIDTH = conf['img_width']
    else:
        IMG_WIDTH = ORIGINAL_WIDTH
    image = tf.decode_raw(features[image_name], tf.uint8)
    image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
    image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
    if 'row_start' in conf:
        image = image[conf['row_start']:conf['row_end']]
    image = tf.reshape(image, [1, IMG_HEIGHT, IMG_WIDTH, COLOR_CHAN])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def mix_datasets(dataset0, dataset1, ratio_01):
    """Sample batch with specified mix of ground truth and generated data_files points.

    Args:
      ground_truth_x: tensor of ground-truth data_files points.
      generated_x: tensor of generated data_files points.
      batch_size: batch size
      ratio_01: ratio between examples taken from the first dataset and the batchsize
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    batch_size = dataset0['images'].get_shape().as_list()[0]
    num_set0 = tf.cast(int(batch_size)*ratio_01, tf.int64)
    idx = tf.range(int(batch_size))
    set0_idx = tf.gather(idx, tf.range(num_set0))
    set1_idx = tf.gather(idx, tf.range(num_set0, int(batch_size)))

    output = {}
    for key in dataset0.keys():
        ten0 = dataset0[key]
        ten1 = dataset1[key]
        dataset0_examps = tf.gather(ten0, set0_idx)
        dataset1_examps = tf.gather(ten1, set1_idx)
        output[key] = tf.reshape(tf.dynamic_stitch([set0_idx, set1_idx],
                         [dataset0_examps, dataset1_examps]), [batch_size] + ten0.get_shape().as_list()[1:])
    return output


def build_tfrecord_input(conf, training=True, input_file=None, shuffle=True):
    if isinstance(conf['data_dir'], (list, tuple)):
        data_set = []
        for dir in conf['data_dir']:
            conf_ = copy.deepcopy(conf)
            conf_['data_dir'] = dir
            data_set.append(build_tfrecord_single(conf_, training, None, shuffle))

        comb_dataset = mix_datasets(data_set[0], data_set[1], 0.5)
        return comb_dataset
    else:
        return build_tfrecord_single(conf, training, input_file, shuffle)


def build_tfrecord_single(conf, training=True, input_file=None, shuffle=True):
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
    print('adim', adim)
    print('sdim', sdim)

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

        if conf['visualize']:  #if visualize do not perform train val split
            filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
            shuffle = False

    print('using shuffle: ', shuffle)
    if shuffle:
        shuffle_list(filenames)
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(serialized_example):
        image_seq, image_main_seq, endeffector_pos_seq, gen_images_seq, gen_states_seq,\
        action_seq, object_pos_seq, robot_pos_seq, goal_image = [], [], [], [], [], [], [], [], []

        load_indx = list(range(0, conf['sequence_length'], conf['skip_frame']))
        print('using frame sequence: ', load_indx)

        rand_h = tf.random_uniform([1], minval=-0.2, maxval=0.2)
        rand_s = tf.random_uniform([1], minval=-0.2, maxval=0.2)
        rand_v = tf.random_uniform([1], minval=-0.2, maxval=0.2)

        features_name = {}

        for i in load_indx:

            image_names = []
            if 'ncam' in conf:
                ncam = conf['ncam']
            else: ncam = 1

            for icam in range(ncam):
                image_names.append(str(i) + '/image_view{}/encoded'.format(icam))
                features_name[image_names[-1]] = tf.FixedLenFeature([1], tf.string)

            if 'image_only' not in conf:
                action_name = str(i) + '/action'
                endeffector_pos_name = str(i) + '/endeffector_pos'


            if 'image_only' not in conf:
                features_name[action_name] = tf.FixedLenFeature([adim], tf.float32)
                features_name[endeffector_pos_name] = tf.FixedLenFeature([sdim], tf.float32)

            if 'test_metric' in conf:
                robot_pos_name = str(i) + '/robot_pos'
                object_pos_name = str(i) + '/object_pos'
                features_name[robot_pos_name] = tf.FixedLenFeature([conf['test_metric']['robot_pos'] * 2], tf.int64)
                features_name[object_pos_name] = tf.FixedLenFeature([conf['test_metric']['object_pos'] * 2], tf.int64)

            if 'load_vidpred_data' in conf:
                gen_image_name = str(i) + '/gen_images'
                gen_states_name = str(i) + '/gen_states'
                features_name[gen_image_name] = tf.FixedLenFeature([1], tf.string)
                features_name[gen_states_name] = tf.FixedLenFeature([sdim], tf.float32)


            features = tf.parse_single_example(serialized_example, features=features_name)

            images_t = []
            for image_name in image_names:
                image = decode_im(conf, features, image_name)

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
                images_t.append(image)

            image_seq.append(tf.stack(images_t, axis=1))

            if 'image_only' not in conf:
                endeffector_pos = tf.reshape(features[endeffector_pos_name], shape=[1, sdim])
                endeffector_pos_seq.append(endeffector_pos)
                action = tf.reshape(features[action_name], shape=[1, adim])
                action_seq.append(action)

            if 'test_metric' in conf:
                robot_pos = tf.reshape(features[robot_pos_name], shape=[1, 2])
                robot_pos_seq.append(robot_pos)

                object_pos = tf.reshape(features[object_pos_name], shape=[1, conf['test_metric']['object_pos'], 2])
                object_pos_seq.append(object_pos)

            if 'load_vidpred_data' in conf:
                gen_images_seq.append(decode_im(gen_image_name))
                gen_states = tf.reshape(features[gen_states_name], shape=[1, sdim])
                gen_states_seq.append(gen_states)

        return_dict = {}
        image_seq = tf.concat(values=image_seq, axis=0)
        return_dict['images'] = tf.squeeze(image_seq)

        if 'goal_image' in conf:
            features_name = {}
            features_name['/goal_image'] = tf.FixedLenFeature([1], tf.string)
            features = tf.parse_single_example(serialized_example, features=features_name)
            goal_image = tf.squeeze(decode_im(conf, features, '/goal_image'))
            return_dict['goal_image'] = goal_image

        if 'image_only' not in conf:
            return_dict['endeffector_pos'] = tf.concat(endeffector_pos_seq, 0)
            return_dict['actions'] = tf.concat(action_seq, 0)

        if 'load_vidpred_data' in conf:
            return_dict['gen_images'] = gen_images_seq
            return_dict['gen_states'] = gen_states_seq

        return return_dict


    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)

    if 'max_epoch' in conf:
        dataset = dataset.repeat(conf['max_epoch'])
    else: dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.batch(conf['batch_size'])
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    output_element = {}
    for k in list(next_element.keys()):
        output_element[k] = tf.reshape(next_element[k], [conf['batch_size']] + next_element[k].get_shape().as_list()[1:])

    return output_element


def main():
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))
    # DATA_DIR = '/mnt/sda1/pushing_data/cartgripper_sact_2view/train'
    # DATA_DIR = '/mnt/sda1/pushing_data/weiss_gripper_20k/test'
    DATA_DIR = os.environ['VMPC_DATA_DIR']
    # DATA_DIR = [DATA_DIR + '/cartgripper_updown_sact/train', DATA_DIR + '/onpolicy/updown_sact_bounded_disc/train']
    DATA_DIR = DATA_DIR + '/cartgripper_updown_sact/train'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 15 #48      # 'sequence length, including context frames.'
    conf['batch_size']= 10
    conf['visualize']= True
    conf['context_frames'] = 2
    # conf['ncam'] = 2

    # conf['row_start'] = 15
    # conf['row_end'] = 63
    conf['sdim'] = 6
    conf['adim'] = 3
    # conf['image_only'] = ''
    # conf['goal_image'] = ""

    conf['orig_size'] = [48, 64]
    # conf['orig_size'] = [64, 64]
    # conf['orig_size'] = [96, 128]
    # conf['load_vidpred_data'] = ''
    # conf['color_augmentation'] = ''
    # conf['test_metric'] = {'robot_pos': 1, 'object_pos': 2}

    print('-------------------------------------------------------------------')
    print('verify current settings!! ')
    for key in list(conf.keys()):
        print(key, ': ', conf[key])
    print('-------------------------------------------------------------------')

    print('testing the reader')

    dict = build_tfrecord_input(conf, training=True)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video
    deltat = []
    end = time.time()
    for i_run in range(10000):
        # print 'run number ', i_run

        # images, actions, endeff, gen_images, gen_endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos'], dict['gen_images'], dict['gen_states']])
        # images, actions, endeff = sess.run([dict['gen_images'], dict['actions'], dict['endeffector_pos']])
        images, actions, endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos']])
        # [images] = sess.run([dict['images']])

        file_path = '/'.join(str.split(DATA_DIR[0], '/')[:-1]+['preview'])

        if 'ncam' in conf:
            vidlist = []
            for i in range(images.shape[2]):
                video = [v.squeeze() for v in np.split(images[:,:,i],images.shape[1], 1)]
                vidlist.append(video)
            npy_to_gif(assemble_gif(vidlist, num_exp=conf['batch_size']), file_path)
        else:
            images = [v.squeeze() for v in np.split(images,images.shape[1], 1)]
            numbers = create_numbers(conf['sequence_length'], conf['batch_size'])
            npy_to_gif(assemble_gif([images, numbers], num_exp=conf['batch_size']), file_path)

        # comp_single_video(file_path, images, num_exp=conf['batch_size'])

        # deltat.append(time.time() - end)
        # if i_run % 10 == 0:
        #     print('tload{}'.format(time.time() - end))
        #     print('average time:', np.average(np.array(deltat)))
        # end = time.time()


        for b in range(10):
            print('actions {}'.format(b))
            print(actions[b])

            print('endeff {}'.format(b))
            print(endeff[b])

            # print 'gen_endeff'
            # print gen_endeff[b]

            # print 'gen_endeff'
            # print gen_endeff[b]

            # print 'video mean brightness', np.mean(images[b])
            # if np.mean(images[b]) < 0.25:
            #     print b
            #     plt.imshow(images[b,0])
            #     plt.show()
            # plt.imshow(images[0, 0])
            # plt.show()
            #
            # pdb.set_trace()

            # print 'robot_pos'
            # print robot_pos
            #
            # print 'object_pos'
            # print object_pos

            # visualize_annotation(conf, images[b], robot_pos[b], object_pos[b])
        import sys
        sys.exit()


def create_numbers(t, size):
    nums = [draw_text_image('im{}'.format(i), (255, 255, 255)) for i in range(size)]
    nums = np.stack(nums, 0)
    nums = [nums for _ in range(t)]
    return nums


if __name__ == '__main__':
    main()
