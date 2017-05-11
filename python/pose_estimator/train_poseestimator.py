import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import lsdc
from copy import deepcopy

from video_prediction.utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
import video_prediction.utils_vpred.create_gif

import matplotlib.pyplot as plt
from poseestimator import construct_model
from PIL import Image
import pdb

from video_prediction.read_tf_record import build_tfrecord_input

from video_prediction.utils_vpred.skip_example import skip_example

from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')



class Model(object):
    def __init__(self,
                 conf,
                 video = None,
                 poses = None,
                 reuse_scope = None,
                 ):
        """
        :param conf:
        :param video:
        :param actions:
        :param states:
        :param lt_states: latent states
        :param test:
        :param ltprop:   whether to porpagate laten state forward
        """
        poses = tf.squeeze(poses)

        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        first_row = tf.reshape(np.arange(conf['batch_size']),shape=[conf['batch_size'],1])
        rand_ind = np.random.randint(0, conf['sequence_length'], size=[conf['batch_size'],1])

        self.num_ind_0 = num_ind_0 = tf.concat(1, [first_row, rand_ind])
        self.image = image = tf.gather_nd(video, num_ind_0)
        self.true_pose = true_pose = tf.gather_nd(poses, num_ind_0)

        if reuse_scope is None:
            is_training = True
        else:
            is_training = False

        if reuse_scope is None:
            inferred_pose  = construct_model(conf, image, is_training= is_training)
        else:
            # If it's a validation or test model.
            if 'nomoving_average' in conf:
                is_training = True
                print 'valmodel with is_training: ', is_training

            with tf.variable_scope(reuse_scope, reuse=True):
                inferred_pose = construct_model(conf, image,is_training=is_training)

        self.inferred_pose = inferred_pose

        inferred_pos = tf.slice(inferred_pose, [0,0], [-1, 2])
        true_pos = tf.slice(true_pose, [0, 0], [-1, 2])
        pos_cost = tf.reduce_sum(tf.square(inferred_pos - true_pos))

        inferred_ori = tf.slice(inferred_pose, [0, 2], [-1, 1])
        true_ori = tf.slice(true_pose, [0, 2], [-1, 1])

        c1 = tf.cos(inferred_ori)
        s1 = tf.sin(inferred_ori)
        c2 = tf.cos(true_ori)
        s2 = tf.sin(true_ori)
        ori_cost = tf.reduce_sum(tf.square(c1 -c2) + tf.square(s1 -s2))

        total_cost = pos_cost + ori_cost

        self.prefix = prefix = tf.placeholder(tf.string, [])
        summaries.append(tf.scalar_summary(prefix + 'pos_cost', pos_cost))
        summaries.append(tf.scalar_summary(prefix + 'ori_cost', ori_cost))
        summaries.append(tf.scalar_summary(prefix + 'total_cost', total_cost))
        self.loss = loss = total_cost
        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        else:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        self.summ_op = tf.merge_summary(summaries)


def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    conf_file = FLAGS.hyper
    if not os.path.exists(conf_file):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    conf['event_log_dir'] = conf['output_dir']
    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
        conf['visualize'] = conf['output_dir'] + '/' + FLAGS.visualize
        conf['event_log_dir'] = '/tmp'
        filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
        conf['visual_file'] = filenames
        conf['batch_size'] = 18

    print 'Constructing models and inputs.'
    with tf.variable_scope('trainmodel') as training_scope:
        images, actions, states, poses  = build_tfrecord_input(conf, training=True)
        model = Model(conf, images, poses)

    with tf.variable_scope('val_model', reuse=None):
        images_val, actions_val, states_val, poses_val = build_tfrecord_input(conf, training=False)
        val_model = Model(conf, images_val, poses_val, reuse_scope= training_scope)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.train.SummaryWriter(
        conf['event_log_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'
    if FLAGS.visualize:
        visualize(conf, sess, saver, val_model)
        return

    itr_0 =0

    if FLAGS.pretrained != None:


        conf['pretrained_model'] = FLAGS.pretrained

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
        feed_dict = {model.prefix: 'train',
                     model.iter_num: np.float32(itr),
                     model.lr: conf['learning_rate'],
                     }
        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        # Print info: iteration #, cost.
        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {val_model.lr: 0.0,
                         val_model.prefix: 'val',
                         val_model.iter_num: np.float32(itr),
                         }
            _, val_summary_str = sess.run([val_model.train_op, val_model.summ_op],
                                          feed_dict)
            summary_writer.add_summary(val_summary_str, itr)


        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model to' + conf['output_dir'])
            oldfile = conf['output_dir'] + '/model' + str(itr - SAVE_INTERVAL)
            if os.path.isfile(oldfile):
                os.system("rm {}".format(oldfile))
                os.system("rm {}".format(oldfile + '.meta'))
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


def visualize(conf, sess, saver, model):
    print 'creating visualizations ...'
    saver.restore(sess,  conf['visualize'])

    feed_dict = {model.lr: 0.0,
                 model.prefix: 'val',
                 }

    input_image, ground_truth_pose, inferred_pose = sess.run([ model.image, model.true_pose, model.inferred_pose
                                                                ],
                                                                feed_dict)
    n_examples = 8
    fig = plt.figure(figsize=(12, 6), dpi=80)

    for ind in range(n_examples):
        ax = fig.add_subplot(3, n_examples, ind+1)
        ax.imshow((input_image[ind]*255).astype(np.uint8))

        plot_arrow(inferred_pose[ind], color= 'r')
        plot_arrow(ground_truth_pose[ind], color='b')

    # plt.tight_layout(pad=0., w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(left=.1, bottom=.1, right=.95, top=.9, wspace=.05, hspace=.05)
    plt.savefig(conf['output_dir'] + '/fig.png')
    plt.show()

def plot_arrow(pose, color = 'r'):
    arrow_start = mujoco_to_imagespace(pose[:2])
    arrow_end = pose[:2] + np.array([np.cos(pose[2]), np.sin(pose[2])]) * .15
    arrow_end = mujoco_to_imagespace(arrow_end)

    plt.plot(arrow_start[1], arrow_start[0], zorder=1, marker='o', color=color)

    yarrow = np.array([arrow_start[0], arrow_end[0]])
    xarrow = np.array([arrow_start[1], arrow_end[1]])
    plt.plot(xarrow, yarrow, zorder=1, color=color, linewidth=3)

    plt.axis('off')

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
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
