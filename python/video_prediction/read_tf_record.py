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

    image_seq, state_seq, action_seq = [], [], []


    load_indx = range(0, 30, conf['skip_frame'])
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


##### code below is used for debugging

def add_visuals_to_batch(image_data, action_data, state_data):
    batchsize, sequence_length = state_data.shape[0], state_data.shape[1]

    img = np.uint8(255. * image_data)

    image__with_visuals = np.zeros((32, 15, 64, 64, 3), dtype=np.uint8)

    for b in range(batchsize):
        for t in range(sequence_length):
            actions = action_data[b, t]
            state = state_data[b, t, :2]
            sel_img = img[b,t]
            image__with_visuals[b, t] = get_frame_with_visual(sel_img, actions, state)

    return image__with_visuals.astype(np.float32) / 255.0


def get_frame_with_visual(img, action, state):
    fig = plt.figure(figsize=(1, 1), dpi=64)
    fig.add_subplot(111)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    axes = plt.gca()
    plt.cla()
    axes.axis('off')
    plt.imshow(img, zorder=0)
    axes.autoscale(False)

    p = mujoco_to_imagespace(state + .05 * action)
    state = mujoco_to_imagespace(state)

    plt.scatter(state[1], state[0], zorder=1, marker='o', color='r')

    yaction = np.array([state[0], p[0]])
    xaction = np.array([state[1], p[1]])
    plt.plot(xaction, yaction, zorder=1, marker='o', color='y')

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
    pixel_coord = np.rint(np.array([-mujoco_coord[1], mujoco_coord[0]]) /
                          pixelwidth + np.array([middle_pixel, middle_pixel]))
    pixel_coord = pixel_coord.astype(int)
    return pixel_coord

if __name__ == '__main__':
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    # DATA_DIR = '/home/frederik/Documents/pushing_data/settled_scene_rnd3/train'
    # DATA_DIR = '/home/frederik/Documents/pushing_data/random_action/train'
    DATA_DIR = '/home/frederik/Documents/pushing_data/old/train'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 2
    conf['train_val_split']= 0.95
    conf['sequence_length']= 15      # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['batch_size']= 32
    conf['visualize']=False

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

        print 'action:', action_data.shape
        print 'action: batch ind 0', action_data[0]
        print 'action: batch ind 1', action_data[1]
        print 'images:', image_data.shape

        print 'states:', state_data.shape
        print 'states: batch ind 0', state_data[0]
        print 'states: batch ind 1', state_data[1]
        print 'average speed in dir1:', np.average(state_data[:,:,3])
        print 'average speed in dir2:', np.average(state_data[:,:,2])

        from utils_vpred.create_gif import comp_single_video

        gif_preview = '/'.join(str.split(__file__, '/')[:-1] + ['preview'])
        comp_single_video(gif_preview, image_data)
        gif_preview = '/'.join(str.split(__file__, '/')[:-1] + ['preview_visuals'])
        comp_single_video(gif_preview, add_visuals_to_batch(image_data, action_data, state_data))

        for i in range(2):
            img = np.uint8(255. *image_data[i,0])
            img = Image.fromarray(img, 'RGB')
            img.show()


