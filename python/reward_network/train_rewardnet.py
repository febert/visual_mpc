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
from reward_model import construct_model
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
                 pred_video = None,
                 gtruth_video = None,
                 currentimages=None,
                 goalimage = None,
                 states = None,
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


        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        if goalimage is not None: # when testing
            # images is shape: batchsize x 2 x 64 x 64 x 3;
            #  the first image is the current image, the second is the goal image

            self.image_0 = image_0 = currentimages
            goalimage = tf.expand_dims(goalimage, dim=0)
            self.image_1 = image_1 = tf.tile(goalimage, [conf['batch_size'], 1, 1, 1])

            inference = True

        else: # when training pick random pairs of images:

            inference = False
            first_row = tf.reshape(np.arange(conf['batch_size']),shape=[conf['batch_size'],1])
            rand_pair = np.random.randint(0, conf['sequence_length'] - 1, size=[conf['batch_size'],2])

            ind_0 = tf.reshape(tf.reduce_min(rand_pair, reduction_indices=1), shape=[conf['batch_size'],1])
            ind_1 = tf.reshape(tf.reduce_max(rand_pair, reduction_indices=1), shape=[conf['batch_size'],1])

            num_ind_0 = tf.concat(1, [first_row, ind_0])
            num_ind_1 = tf.concat(1, [first_row, ind_1])

            if pred_video is not None:
                self.image_0 = image_0 = tf.gather_nd(pred_video, num_ind_0)
                self.image_1 = image_1 = tf.gather_nd(gtruth_video, num_ind_1)
            else:
                self.image_0 = image_0 = tf.gather_nd(video, num_ind_0)
                self.image_1 = image_1 = tf.gather_nd(video, num_ind_1)

        if reuse_scope is None:
            is_training = True
        else:
            is_training = False

        if 'dropout' in conf:
            if is_training:
                conf['dropout'] = 0.5
            else:
                conf['dropout'] = 1

        if reuse_scope is None:
            softmax_output  = construct_model(conf, images_0=image_0,
                                            images_1=image_1,
                                            is_training= is_training)
        else: # If it's a validation or test model.
            if 'nomoving_average' in conf:
                is_training = True
                print 'valmodel with is_training: ', is_training

            with tf.variable_scope(reuse_scope, reuse=True):
                softmax_output = construct_model(conf,  images_0=image_0,
                                                images_1=image_1,
                                                is_training=is_training)

        self.softmax_output = tf.nn.softmax(softmax_output)


        # mult = tf.mul(softmax_output, tf.cast(tf.range(0, conf['sequence_length']-1), tf.float32))
        # expected_timesteps = tf.reduce_sum(mult,1)

        # da_dx0 = []
        # # for el in range(conf['batch_size']):
        # for el in range(8):
        #     exp_t = tf.slice(expected_timesteps, [el], [1])
        #     image_0_ex = tf.slice(image_0, [el, 0, 0, 0,], [1, -1, -1, -1])
        #     [grad] = tf.gradients(exp_t, image_0_ex)
        #     pdb.set_trace()
        #     grad = tf.expand_dims(grad, 0)
        #     da_dx0.append(grad)
        # self.da_dx0 = tf.concat(0, da_dx0)
        #
        # da_dx1 = []
        # # for el in range(conf['batch_size']):
        # for el in range(8):
        #     exp_t = tf.slice(expected_timesteps, [el], [1])
        #     image_1_ex = tf.slice(image_0, [el, 0, 0, 0,], [1, -1, -1, -1])
        #     [grad] = tf.gradients(exp_t, image_1_ex)
        #     grad = tf.expand_dims(grad, 0)
        #     da_dx1.append(grad)
        # self.da_dx1 = tf.concat(0, da_dx1)


        if inference == False:
            self.hard_labels = hard_labels = tf.squeeze(ind_1 - ind_0)


            rows = []
            for i in range(conf['batch_size']):
                tstep = tf.slice(self.hard_labels, [i], [1])
                zeros = tf.zeros(tf.to_int32(tstep))
                ones = tf.ones(tf.to_int32(conf['sequence_length']-1 - tstep))
                ones = ones / tf.reduce_sum(ones)
                row = tf.expand_dims(tf.concat(0, [zeros, ones]),0)
                rows.append(row)

            self.soft_labels = tf.concat(0, rows)

            if 'soft_labels' in conf:
                self.cross_entropy = cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.soft_labels, logits=softmax_output, name='cross_entropy_per_example')
            else:
                self.cross_entropy = cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=hard_labels, logits=softmax_output, name='cross_entropy_per_example')

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            self.prefix = prefix = tf.placeholder(tf.string, [])
            summaries.append(tf.scalar_summary(prefix + 'cross_entropy_mean', cross_entropy_mean))
            self.loss = loss = cross_entropy_mean
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
        conf['visual_file'] = filenames[0]
        conf['batch_size'] = 8

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'


    print 'Constructing models and inputs.'
    with tf.variable_scope('trainmodel') as training_scope:
        if 'pred_gtruth' in conf:
            gtruth_video, pred_video = build_tfrecord_input(conf, training=True)
            model = Model(conf, pred_video= pred_video, gtruth_video= gtruth_video)
        else:
            images, actions, states = build_tfrecord_input(conf, training=True)
            model = Model(conf, images)

    with tf.variable_scope('val_model', reuse=None):
        if 'pred_gtruth' in conf:
            gtruth_video_val, pred_video_val = build_tfrecord_input(conf, training=True)
            val_model = Model(conf, pred_video= pred_video_val, gtruth_video= gtruth_video_val)
        else:
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
    im0, im1, softout, c_entr, gtruth, soft_labels, dadx0, dadx1 = sess.run([  model.image_0,
                                                    model.image_1,
                                                    model.softmax_output,
                                                    model.cross_entropy,
                                                    model.hard_labels,
                                                    model.soft_labels,
                                                    model.da_dx0,
                                                    model.da_dx1
                                                                ],
                                                    feed_dict)

    fig = plt.figure(figsize=(20, 10), dpi=80)
    n_examples = 8

    for ind in range(n_examples):
        ax = fig.add_subplot(3, n_examples, ind+1)
        ax.imshow((im0[ind]*255).astype(np.uint8))
        plt.axis('off')

        ax = fig.add_subplot(3, n_examples, n_examples+1+ind)
        ax.imshow((im1[ind]*255).astype(np.uint8))
        plt.axis('off')


        # visualize dadx0
        ax = fig.add_subplot(3, n_examples, n_examples*2 + 1 + ind)
        plt.imshow(dadx0[ind], zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
        plt.axis('off')
        # visualize dadx1
        ax = fig.add_subplot(3, n_examples, n_examples*3 + 1 + ind)
        plt.imshow(dadx1[ind], zorder=0, cmap=plt.get_cmap('jet'), interpolation='none')
        plt.axis('off')

        ax = fig.add_subplot(3, n_examples, n_examples*4 +ind +1)

        N = conf['sequence_length'] -1
        values = softout[ind]

        loc = np.arange(N)  # the x locations for the groups
        width = 0.3  # the width of the bars

        rects1 = ax.bar(loc, values, width)

        # add some text for labels, title and axes ticks
        ax.set_title('softmax')
        ax.set_xticks(loc + width / 2)
        ax.set_xticklabels([str(j+1) for j in range(N)])

        centr = 0.
        for i in range(N):
            if gtruth[ind] == i:
                l = 1
            else:
                l = 0
            centr += np.log(softout[ind,i])*l + (1-l)* np.log(1- softout[ind,i])
        centr = -centr

        if 'soft_labels' in conf:
            print 'softlabel {0}, gtrut {1}'.format(soft_labels[ind], gtruth[ind])


        ax.set_xlabel('true temp distance: {0} \n  cross-entropy: {1}\n self-calc centr: {2}'
                      .format(gtruth[ind], round(c_entr[ind], 3), round(centr, 3)))

    # plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
    plt.savefig(conf['output_dir'] + '/fig.png')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
