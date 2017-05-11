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

from matplotlib import gridspec

import matplotlib.pyplot as plt
from featurepoints import construct_model
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


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model(object):
    def __init__(self,
                 conf,
                 video = None,
                 reuse_scope = None,
                 ):

        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        first_row = tf.reshape(np.arange(conf['batch_size']),shape=[conf['batch_size'],1])
        rand_ind = np.random.randint(0, conf['sequence_length'], size=[conf['batch_size'],1])

        self.num_ind_0 = num_ind_0 = tf.concat(1, [first_row, rand_ind])
        self.input_images = input_images = tf.gather_nd(video, num_ind_0)

        if reuse_scope is None:
            is_training = True
        else:
            is_training = False

        if reuse_scope is None:
            images_rec, feature_points  = construct_model(conf, input_images, is_training= is_training)
        else:
            # If it's a validation or test model.
            if 'nomoving_average' in conf:
                is_training = True
                print 'valmodel with is_training: ', is_training

            with tf.variable_scope(reuse_scope, reuse=True):
                images_rec, feature_points = construct_model(conf, input_images,is_training=is_training)

        self.feature_points = feature_points
        self.images_rec = images_rec
        self.loss = loss = mean_squared_error(input_images, images_rec)
        self.prefix = prefix = tf.placeholder(tf.string, [])
        summaries.append(tf.scalar_summary(prefix + 'reconstr_loss', loss))
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


    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
        conf['visualize'] = conf['output_dir'] + '/' + FLAGS.visualize
        conf['event_log_dir'] = '/tmp'
        filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
        conf['visual_file'] = filenames

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'


    print 'Constructing models and inputs.'
    with tf.variable_scope('trainmodel') as training_scope:
        images, actions, states  = build_tfrecord_input(conf, training=True)
        model = Model(conf, images)

    with tf.variable_scope('val_model', reuse=None):
        images_val, actions_val, states_val = build_tfrecord_input(conf, training=False)
        val_model = Model(conf, images_val, reuse_scope= training_scope)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

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

    inp_images, rec_images, fp = sess.run([ model.input_images,
                                            model.images_rec,
                                            model.feature_points
                                            ],
                                            feed_dict)

    n_examples = 8
    fig = plt.figure(figsize=(n_examples*2+4, 5), dpi=80)

    for ind in range(n_examples):
        # ax = fig.add_subplot(3, n_examples, ind+1)
        ax = plt.subplot2grid((2, n_examples), (0, ind))

        ax.imshow((inp_images[ind]*255).astype(np.uint8))

        plt.axis('off')

        # ax = fig.add_subplot(3, n_examples, n_examples+ind + 1)
        ax = plt.subplot2grid((2, n_examples), (1, ind))
        ax.imshow((rec_images[ind] * 255).astype(np.uint8))

        # plt.plot(0., 0., marker='o', color='b')
        # plt.plot(64., 64., marker='o', color='r')
        for i_p in range(fp.shape[1]):
            plt.plot(fp[ind,i_p,0]*64.+32, fp[ind,i_p,1]*64.+32, marker='o', color='b')

        # if ind == 1:
        print 'feature points, shape:',fp.shape
        print fp[ind]*64. + 32

        plt.axis('off')

    plt.tight_layout()
    plt.savefig(conf['output_dir'] + '/fig.png')
    plt.show()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
