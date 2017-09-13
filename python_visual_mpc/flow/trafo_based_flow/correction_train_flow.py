# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for training the prediction model."""
import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import pdb

import matplotlib.pyplot as plt

from python_visual_mpc.video_prediction.utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from python_visual_mpc.video_prediction.read_tf_record_sawyer12 import build_tfrecord_input

from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000



## Helper functions
def peak_signal_to_noise_ratio(true, pred):
    """Image quality metric based on maximal signal power vs. power of the noise.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      peak signal to noise ratio (PSNR)
    """
    return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Getdesig(object):
    def __init__(self,img,conf,img_namesuffix):
        self.suf = img_namesuffix
        self.conf = conf
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        self.coords = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        self.coords = np.array([event.ydata, event.xdata])
        self.ax.scatter(self.coords[1], self.coords[0], marker= "o", s=70, facecolors='b', edgecolors='b')
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.draw()
        plt.savefig(self.conf['output_dir']+'/img_desigpix'+self.suf)


class CorrectorModel(object):
    def __init__(self,
                 conf,
                 images=None,
                 reuse_scope=None,
                 pix_distrib=None,
                 train = True
                 ):

        self.conf = conf

        from correction import construct_correction

        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        if train:
            rand_ind = tf.random_uniform([1], 0, self.conf['sequence_length']-1, dtype=tf.int64)
            self.rand_ind = rand_ind
            images = [images[:,tf.squeeze(rand_ind)], images[:,tf.squeeze(rand_ind+1)]]
        else:
            images = tf.split(images,2,1)
            images = [tf.reshape(im, (1, 64,64, 3)) for im in images]

        if reuse_scope is None:
            gen_images, gen_masks, gen_distrib, flow = construct_correction(
                conf,
                images,
                num_masks=conf['num_masks'],
                cdna=conf['model'] == 'CDNA',
                dna=conf['model'] == 'DNA',
                stp=conf['model'] == 'STP',
                pix_distrib_input= pix_distrib)
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images, gen_masks, gen_distrib, flow = construct_correction(
                    conf,
                    images,
                    num_masks=conf['num_masks'],
                    cdna=conf['model'] == 'CDNA',
                    dna=conf['model'] == 'DNA',
                    stp=conf['model'] == 'STP',
                    pix_distrib_input=pix_distrib)

        # L2 loss, PSNR for eval.

        loss = mean_squared_error(images[1], gen_images)
        summaries.append(tf.summary.scalar('recon_cost', loss))

        self.loss = loss

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.summary.merge(summaries)

        self.gen_images= gen_images
        self.gen_masks = gen_masks
        self.gen_distrib = gen_distrib

        self.flow = flow

def mujoco_to_imagespace(mujoco_coord, numpix=64):
    """
    convert form Mujoco-Coord to numpix x numpix image space:
    :param numpix: number of pixels of square image
    :param mujoco_coord:
    :return: pixel_coord
    """
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


def visualize(conf):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.

    conf['batch_size'] = 1

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    tf.train.start_queue_runners(sess)

    input_distrib = tf.placeholder(tf.float32, shape=(conf['batch_size'], 64, 64, 1))

    images = tf.placeholder(tf.float32, name='images',
                            shape=(conf['batch_size'], 2, 64, 64, 3))

    with tf.variable_scope('model', reuse=None):
        val_images, _, _   = build_tfrecord_input(conf, training=False)
        model = CorrectorModel(conf, images, pix_distrib= input_distrib, train=False)

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    import re
    itr_vis = re.match('.*?([0-9]+)$', conf['visualize']).group(1)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    saver.restore(sess, conf['output_dir'] + '/' + FLAGS.visualize)
    print 'restore done.'

    [ground_truth] = sess.run([val_images])

    b_exp = 0
    initial_img = ground_truth[b_exp][0]
    c = Getdesig(initial_img, conf, 'b{}'.format(b_exp))
    desig_pos_aux1 = c.coords.astype(np.int32)
    # desig_pos_aux1 = np.array([31, 29])
    start_one_hot = np.zeros([conf['batch_size'], 64, 64, 1])

    start_one_hot[0, desig_pos_aux1[0], desig_pos_aux1[1]] = 1

    output_distrib_list, gen_masks_list, gen_image_list, flow_list = [], [], [], []

    for t in range(conf['sequence_length']-1):

        if t == 0:
            next_input_distrib = start_one_hot
        else:
            next_input_distrib = output_distrib

        feed_dict = {
                     model.lr: 0,
                     images: ground_truth[:,t:t+2],  #could alternatively feed in gen_image
                     input_distrib: next_input_distrib
                     }

        gen_image, gen_masks, output_distrib, flow = sess.run([model.gen_images,
                                                             model.gen_masks,
                                                             model.gen_distrib,
                                                             model.flow
                                                             ],
                                                             feed_dict)

        output_distrib_list.append(output_distrib)
        gen_image_list.append(gen_image)
        gen_masks_list.append(gen_masks)
        flow_list.append(flow)

    import collections

    dict = collections.OrderedDict()
    dict['ground_truth'] = ground_truth
    dict['gen_images'] = gen_image_list
    dict['gen_masks'] = gen_masks_list
    dict['gen_distrib'] = output_distrib_list
    dict['flow'] = flow_list

    dict['iternum'] = itr_vis

    cPickle.dump(dict, open(conf['output_dir'] + '/pred.pkl', 'wb'))
    print 'written files to:' + conf['output_dir']

    from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
    v = Visualizer_tkinter(dict, numex=1, append_masks=True,
                           gif_savepath=conf['output_dir'],
                           suf='flow{}_l{}'.format(b_exp, conf['sequence_length']))
    v.build_figure()


def main(unused_argv, conf_script= None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    if conf_script == None: conf_file = FLAGS.hyper
    else: conf_file = conf_script

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)
    conf = hyperparams.configuration
    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf = adapt_params_visualize(conf, FLAGS.visualize)
    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'


    if conf['visualize']:
        print 'visualizing'
        visualize(conf)

    print 'Constructing models and inputs'
    with tf.variable_scope('model', reuse=None) as training_scope:
        images,_ , _ = build_tfrecord_input(conf, training=True)
        model = CorrectorModel(conf, images)

    with tf.variable_scope('val_model', reuse=None):
        val_images,_ , _ = build_tfrecord_input(conf, training=False)
        val_model = CorrectorModel(conf, val_images, reuse_scope=training_scope)

    print 'Constructing saver.'
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())


    itr_0 =0
    if conf['pretrained_model']:    # is the order of initialize_all_variables() and restore() important?!?
        saver.restore(sess, conf['pretrained_model'])
        # resume training at iteration step of the loaded model:
        import re
        itr_0 = re.match('.*?([0-9]+)$', conf['pretrained_model']).group(1)
        itr_0 = int(itr_0)
        print 'resuming training at iteration:  ', itr_0

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter = []
    # Run training.
    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()
        # Generate new batch of data_files.
        feed_dict = {
                     model.iter_num: np.float32(itr),
                     model.lr: conf['learning_rate']}
        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        # Print info: iteration #, cost.
        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {val_model.lr: 0.0,
                         val_model.iter_num: np.float32(itr)}
            _, val_summary_str = sess.run([val_model.train_op, val_model.summ_op],
                                          feed_dict)
            summary_writer.add_summary(val_summary_str, itr)


        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model to' + conf['output_dir'])
            saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        t_iter.append((datetime.now() - t_startiter).seconds * 1e6 +  (datetime.now() - t_startiter).microseconds )

        if itr % 100 == 1:
            hours = (datetime.now() -starttime).seconds/3600
            tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                (datetime.now() - starttime).days,
                hours,
                (datetime.now() - starttime).seconds/60 - hours*60))
            avg_t_iter = np.sum(np.asarray(t_iter))/len(t_iter)
            tf.logging.info('time per iteration: {0}'.format(avg_t_iter/1e6))
            tf.logging.info('expected for complete training: {0}h '.format(avg_t_iter /1e6/3600 * conf['num_iterations']))

        if (itr) % SUMMARY_INTERVAL:
            summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, conf['output_dir'] + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0, 'the value for CUDA_VISIBLE_DEVICES variable')

    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
