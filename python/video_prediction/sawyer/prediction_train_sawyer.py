import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import pdb

import imp

from video_prediction.utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import video_prediction.utils_vpred.create_gif
from create_gif import create_gif

from read_tf_record_sawyer import build_tfrecord_input

from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

FLAGS = flags.FLAGS
flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')

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




class Model(object):
    def __init__(self,
                 conf,
                 images=None,
                 actions=None,
                 states=None,
                 reuse_scope=None,
                 pix_distrib=None):

        self.conf = conf
        from prediction_model_sawyer import construct_model

        if 'use_length' in conf:
            #randomly shift videos for data augmentation
            images, actions, states = self.random_shift(images, actions, states)

        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        # Split into timesteps.
        if actions != None:
            actions = tf.split(1, actions.get_shape()[1], actions)
            actions = [tf.squeeze(act) for act in actions]
        if states != None:
            states = tf.split(1, states.get_shape()[1], states)
            states = [tf.squeeze(st) for st in states]
        images = tf.split(1, images.get_shape()[1], images)
        images = [tf.squeeze(img) for img in images]
        if pix_distrib != None:
            pix_distrib = tf.split(1, pix_distrib.get_shape()[1], pix_distrib)
            pix_distrib = [tf.squeeze(pix) for pix in pix_distrib]

        if reuse_scope is None:
            gen_images, gen_states, gen_masks, gen_distrib = construct_model(
                images,
                actions,
                states,
                iter_num=self.iter_num,
                k=conf['schedsamp_k'],
                num_masks=conf['num_masks'],
                context_frames=conf['context_frames'],
                pix_distributions= pix_distrib,
                conf=conf)
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images, gen_states, gen_masks, gen_distrib = construct_model(
                    images,
                    actions,
                    states,
                    iter_num=self.iter_num,
                    k=conf['schedsamp_k'],
                    num_masks=conf['num_masks'],
                    context_frames=conf['context_frames'],
                    conf= conf)

        # L2 loss, PSNR for eval.
        true_fft_list, pred_fft_list = [], []
        loss, psnr_all = 0.0, 0.0

        self.fft_weights = tf.placeholder(tf.float32, [64, 64])

        for i, x, gx in zip(
                range(len(gen_images)), images[conf['context_frames']:],
                gen_images[conf['context_frames'] - 1:]):
            recon_cost_mse = mean_squared_error(x, gx)

            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(
                tf.scalar_summary(prefix + '_recon_cost' + str(i), recon_cost_mse))
            summaries.append(tf.scalar_summary(prefix + '_psnr' + str(i), psnr_i))

            recon_cost = recon_cost_mse

            loss += recon_cost

        if ('ignore_state_action' not in conf) and ('ignore_state' not in conf):
            for i, state, gen_state in zip(
                    range(len(gen_states)), states[conf['context_frames']:],
                    gen_states[conf['context_frames'] - 1:]):
                state_cost = mean_squared_error(state, gen_state) * 1e-4 * conf['use_state']
                summaries.append(
                    tf.scalar_summary(prefix + '_state_cost' + str(i), state_cost))
                loss += state_cost

        summaries.append(tf.scalar_summary(prefix + '_psnr_all', psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])

        summaries.append(tf.scalar_summary(prefix + '_loss', loss))

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)

        self.true_fft = true_fft_list
        self.pred_fft = pred_fft_list
        self.gen_images= gen_images
        self.gen_masks = gen_masks
        self.gen_distrib = gen_distrib
        self.gen_states = gen_states

    def random_shift(self, images, states, actions):
        print 'shifting the video sequence randomly in time'
        tshift = 2
        uselen = self.conf['use_len']
        fulllength = self.conf['sequence_length']
        nshifts = (fulllength - uselen) / 2 + 1
        rand_ind = np.random.randint(0, nshifts + 1, size=[1])  # sample from [0,8]
        images = tf.slice(images, [0, rand_ind * tshift, 0, 0, 0], [-1, uselen, -1, -1, -1])
        actions = tf.slice(actions, [0, rand_ind * tshift, 0], [-1, uselen, -1])
        states = tf.slice(states, [0, rand_ind * tshift, 0], [-1, uselen, -1])

        return images, states, actions



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
        conf['visual_file'] = conf['data_dir'] + '/traj_0_to_255.tfrecords'


    print 'Constructing models and inputs.'
    with tf.variable_scope('model', reuse=None) as training_scope:
        if 'sawyer' in conf:

            if 'single_view' in conf:
                images_aux1, actions, states = build_tfrecord_input(conf, training=True)
                images = images_aux1
            else:
                images_main, images_aux1, actions, states = build_tfrecord_input(conf, training=True)
                images_aux1 = tf.squeeze(images_aux1)
                images = tf.concat(4, [images_main, images_aux1])
            model = Model(conf, images, actions, states)

    with tf.variable_scope('val_model', reuse=None):
        if 'sawyer' in conf:
            if 'single_view' in conf:
                val_images_aux1, val_actions, val_states = build_tfrecord_input(conf, training=False)
                val_images = val_images_aux1
            else:
                val_images_main, val_images_aux1, val_actions, val_states = build_tfrecord_input(conf, training=False)
                val_images_aux1 = tf.squeeze(val_images_aux1)
                val_images = tf.concat(4, [val_images_main, val_images_aux1])
            val_model = Model(conf, val_images, val_actions, val_states, training_scope)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    if conf['visualize']:
        print '-------------------------------------------------------------------'
        print 'verify current settings!! '
        for key in conf.keys():
            print key, ': ', conf[key]
        print '-------------------------------------------------------------------'


        saver.restore(sess, conf['visualize'])

        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'vis',
                     val_model.iter_num: 0 }
        file_path = conf['output_dir']

        if 'fftcost' in conf:
            true_fft, pred_fft, gen_images, ground_truth, mask_list = sess.run([val_model.true_fft, val_model.pred_fft ,val_model.gen_images,
                                                            val_images, val_model.gen_masks],
                                                           feed_dict)
            cPickle.dump(true_fft, open(file_path + '/true_fft.pkl', 'wb'))
            cPickle.dump(pred_fft, open(file_path + '/pred_fft.pkl', 'wb'))
        else:
            gen_images, ground_truth, mask_list = sess.run([val_model.gen_images,
                                                            val_images, val_model.gen_masks,
                                                            ],
                                                           feed_dict)

        cPickle.dump(gen_images, open(file_path + '/gen_image.pkl','wb'))
        cPickle.dump(ground_truth, open(file_path + '/ground_truth.pkl', 'wb'))
        cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
        print 'written files to:' + file_path

        create_gif(file_path, conf)
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

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter = []
    # Run training.
    fft_weights = calc_fft_weight()

    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()
        # Generate new batch of data_files.
        feed_dict = {model.prefix: 'train',
                     model.iter_num: np.float32(itr),
                     model.lr: conf['learning_rate'],
                     model.fft_weights: fft_weights}
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
                         val_model.fft_weights: fft_weights}
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


def calc_fft_weight():

    weight = np.zeros((64,64))
    for row in range(64):
        for col in range(64):
            p = np.array([row,col])
            c = np.array([31,31])
            weight[row, col] = np.linalg.norm(p -c)**2

    weight /= np.max(weight)
    return weight

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
