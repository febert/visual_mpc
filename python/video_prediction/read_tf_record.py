import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import pdb

from PIL import Image
import imp



# Dimension of the state and action.
STATE_DIM = 4
ACION_DIM = 2
OBJECT_POS_DIM = 8


def build_tfrecord_input(conf, training=True, gtruth_pred = False):
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

    index = int(np.ceil(conf['train_val_split'] * len(filenames)))
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

    gtruthimage_seq, predimage_seq, image_seq, state_seq, action_seq, object_pos_seq, touch_seq = [], [], [], [], [], [], []


    load_indx = range(0, 30, conf['skip_frame'])
    load_indx = load_indx[:conf['sequence_length']]
    print 'using frame sequence: ', load_indx

    for i in load_indx:
        if gtruth_pred:
            image_pred_name = 'move/' + str(i) + '/image_pred/encoded'
            image_gtruth_name = 'move/' + str(i) + '/image_gtruth/encoded'

            features = {
                image_pred_name: tf.FixedLenFeature([1], tf.string),
                image_gtruth_name: tf.FixedLenFeature([1], tf.string),
            }
        else:
            image_name = 'move/' + str(i) + '/image/encoded'
            action_name = 'move/' + str(i) + '/action'
            state_name = 'move/' + str(i) + '/state'
            object_pos_name = 'move/' + str(i) + '/object_pos'

            features = {
                        image_name: tf.FixedLenFeature([1], tf.string),
                        action_name: tf.FixedLenFeature([ACION_DIM], tf.float32),
                        state_name: tf.FixedLenFeature([STATE_DIM], tf.float32)
            }
        if 'use_object_pos' in conf.keys():
            if conf['use_object_pos']:
                features[object_pos_name] = tf.FixedLenFeature([OBJECT_POS_DIM], tf.float32)

        if 'touch' in conf:
            touchdata_name = 'touchdata/' + str(i)
            TOUCH_DIM = 20
            features[touchdata_name] =  tf.FixedLenFeature([TOUCH_DIM], tf.float32)

        features = tf.parse_single_example(serialized_example, features=features)

        if gtruth_pred:
            predimage_seq.append(resize_im( features, image_pred_name))
            gtruthimage_seq.append(resize_im( features, image_gtruth_name))

        else:

            image_seq.append(resize_im( features, image_name))

            state = tf.reshape(features[state_name], shape=[1, STATE_DIM])
            state_seq.append(state)
            action = tf.reshape(features[action_name], shape=[1, ACION_DIM])
            action_seq.append(action)

            if 'touch' in conf:
                touchdata = tf.reshape(features[touchdata_name], shape=[1, TOUCH_DIM])
                touch_seq.append(touchdata)


            if 'use_object_pos' in conf.keys():
                if conf['use_object_pos']:
                    object_pos = tf.reshape(features[object_pos_name], shape=[1, OBJECT_POS_DIM])
                    object_pos_seq.append(object_pos)

    if gtruth_pred:
        gtruthimage_seq = tf.concat(0, gtruthimage_seq)
        predimage_seq = tf.concat(0, predimage_seq)

        if conf['visualize']:
            num_threads = 1
        else:
            num_threads = np.min((conf['batch_size'], 32))

        [pred_image_batch, gtruth_image_batch] = tf.train.batch(
            [predimage_seq, gtruthimage_seq],
            conf['batch_size'],
            num_threads=num_threads,
            capacity=100 * conf['batch_size'])
        return gtruth_image_batch, pred_image_batch
    else:
        image_seq = tf.concat(0, image_seq)

        if conf['visualize']: num_threads = 1
        else: num_threads = np.min((conf['batch_size'], 32))

        state_seq = tf.concat(0, state_seq)
        action_seq = tf.concat(0, action_seq)
        touch_seq = tf.concat(0, touch_seq)

        if 'use_object_pos' in conf.keys():
            if conf['use_object_pos']:
                [image_batch, action_batch, state_batch, object_pos_batch] = tf.train.batch(
                [image_seq, action_seq, state_seq, object_pos_seq],
                conf['batch_size'],
                num_threads=num_threads,
                capacity=100 * conf['batch_size'])

                return image_batch, action_batch, state_batch, object_pos_batch
        elif 'touch' in conf:
            [image_batch, action_batch, state_batch, touch_batch] = tf.train.batch(
                [image_seq, action_seq, state_seq, touch_seq],
                conf['batch_size'],
                num_threads=num_threads,
                capacity=100 * conf['batch_size'])
            return image_batch, action_batch, state_batch, touch_batch
        else:
            [image_batch, action_batch, state_batch] = tf.train.batch(
                [image_seq, action_seq, state_seq],
                conf['batch_size'],
                num_threads=num_threads,
                capacity=100 * conf['batch_size'])
            return image_batch, action_batch, state_batch


def resize_im(features, image_pred_name):
    COLOR_CHAN = 3
    if '128x128' in conf:
        ORIGINAL_WIDTH = 128
        ORIGINAL_HEIGHT = 128
        IMG_WIDTH = 128
        IMG_HEIGHT = 128
    else:
        ORIGINAL_WIDTH = 64
        ORIGINAL_HEIGHT = 64
        IMG_WIDTH = 64
        IMG_HEIGHT = 64

    image = tf.decode_raw(features[image_pred_name], tf.uint8)
    image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
    image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
    if IMG_HEIGHT != IMG_WIDTH:
        raise ValueError('Unequal height and width unsupported')
    crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
    image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    return image


##### code below is used for debugging

def add_visuals_to_batch(image_data, action_data, state_data, action_pos = False):
    batchsize, sequence_length = state_data.shape[0], state_data.shape[1]

    img = np.uint8(255. * image_data)

    image__with_visuals = np.zeros((32, 15, 64, 64, 3), dtype=np.uint8)

    for b in range(batchsize):
        for t in range(sequence_length):
            actions = action_data[b, t]
            state = state_data[b, t, :2]
            sel_img = img[b,t]
            image__with_visuals[b, t] = get_frame_with_visual(sel_img, actions, state, action_pos= action_pos)

    return image__with_visuals.astype(np.float32) / 255.0


def get_frame_with_posdata(img, pos):
    """
    visualizes the actions in the frame
    :param img:
    :param action:
    :param state:
    :param action_pos:
    :return:
    """
    pos = pos.squeeze().reshape(4,2)

    fig = plt.figure(figsize=(1, 1), dpi=64)
    fig.add_subplot(111)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    axes = plt.gca()
    plt.cla()
    axes.axis('off')
    plt.imshow(img, zorder=0)
    axes.autoscale(False)

    for i in range(4):
        pos_img = mujoco_to_imagespace(pos[i])
        plt.plot(pos_img[1], pos_img[0], zorder=1, marker='o', color='b')

    fig.canvas.draw()  # draw the canvas, cache the renderer

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.show()
    Image.fromarray(data).show()
    pdb.set_trace()

    return data


def get_frame_with_visual(img, action, state, action_pos= False):
    """
    visualizes the actions in the frame
    :param img:
    :param action:
    :param state:
    :param action_pos:
    :return:
    """
    fig = plt.figure(figsize=(1, 1), dpi=64)
    fig.add_subplot(111)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    axes = plt.gca()
    plt.cla()
    axes.axis('off')
    plt.imshow(img, zorder=0)
    axes.autoscale(False)

    if action_pos:
        p = mujoco_to_imagespace(action)
    else:
        p = mujoco_to_imagespace(state + .05 * action)

    state = mujoco_to_imagespace(state)

    plt.plot(state[1], state[0], zorder=1, marker='o', color='r')

    yaction = np.array([state[0], p[0]])
    xaction = np.array([state[1], p[1]])
    plt.plot(xaction, yaction, zorder=1, color='y', linewidth=3)

    fig.canvas.draw()  # draw the canvas, cache the renderer

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.show()
    # Image.fromarray(data).show()
    # pdb.set_trace()

    return data

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

    # DATA_DIR = '/home/frederik/Documents/lsdc/experiments/cem_exp/benchmarks_goalimage/pixelerror_store_wholepred/tfrecords/train'
    DATA_DIR = '/home/frederik/Documents/lsdc/pushing_data/random_action_var10_touch/train'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 15      # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['batch_size']= 5
    conf['visualize']=False


    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    # both ground truth and predicted images in data:
    gtruth_pred = False
    touch = True

    print 'testing the reader'
    if gtruth_pred:
        gtruth_image_batch, pred_image_batch  = build_tfrecord_input(conf, training=True, gtruth_pred= gtruth_pred)
    elif touch:
        conf['touch'] = ''
        image_batch, action_batch, state_batch, touch_batch = build_tfrecord_input(conf, training=True)
    else:
        image_batch, action_batch, state_batch = build_tfrecord_input(conf, training=True, gtruth_pred=True)
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())


    for i in range(1):
        print 'run number ', i
        if gtruth_pred:
            gtruth_data, pred_data = sess.run([gtruth_image_batch, pred_image_batch])
        elif touch:
            image_data, action_data, state_data, touch_data = sess.run([image_batch,
                                                                        action_batch,
                                                                        state_batch,
                                                                        touch_batch])
        else:
            image_data, action_data, state_data = sess.run([image_batch, action_batch, state_batch])



        # print 'action:', action_data.shape
        # print 'action: batch ind 0', action_data[0]
        # print 'action: batch ind 1', action_data[1]
        # print 'images:', image_data.shape
        #
        # print 'states:', state_data.shape
        # print 'states: batch ind 0', state_data[0]
        # print 'states: batch ind 1', state_data[1]
        # print 'average speed in dir1:', np.average(state_data[:,:,3])
        # print 'average speed in dir2:', np.average(state_data[:,:,2])

        print 'touchdata:', touch_data.shape
        print 'touch_data: batch ind 0', touch_data[0]
        print 'touch_data: batch ind 1', touch_data[1]


        # import pdb;pdb.set_trace()
        # pos = state_data[:,:,:2]
        # action_var = np.cov(action_data.reshape(action_data.shape[0]*action_data.shape[1], -1), rowvar= False)
        # pos_var = np.cov(pos.reshape(pos.shape[0] * pos.shape[1], -1), rowvar=False)
        #
        # print 'action variance of single batch'
        # print action_var
        # print 'state variance of single batch'
        # print pos_var


        from utils_vpred.create_gif import comp_single_video

        # make video preview video
        gif_preview = '/'.join(str.split(__file__, '/')[:-1] + ['preview'])
        if gtruth_pred:
            comp_single_video(gif_preview, gtruth_data, predicted=pred_data, num_exp=conf['batch_size'])
        else:
            comp_single_video(gif_preview, image_data, num_exp=conf['batch_size'])


        # make video preview video with annotated forces
        # gif_preview = '/'.join(str.split(__file__, '/')[:-1] + ['preview_visuals'])
        # comp_single_video(gif_preview, add_visuals_to_batch(image_data, action_data, state_data, action_pos=True))

        # show some frames
        # for i in range(10):
        #     # print 'object pos', object_pos.shape
        #     img = np.uint8(255. *image_data[0, i])
        #     img = Image.fromarray(img, 'RGB')
        #     img.show()
            # get_frame_with_posdata(img, object_pos[0, i])
