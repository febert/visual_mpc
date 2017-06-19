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

from video_prediction.read_tf_record import build_tfrecord_input

from video_prediction.utils_vpred.skip_example import skip_example
from occlusionmodel import Occlusion_Model
import makegifs

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
flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable, -1 uses cpu')
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
                 ):

        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        self.conf = conf
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

        if reuse_scope is None:
            self.om = Occlusion_Model(
                images,
                actions,
                states,
                iter_num=self.iter_num,
                conf=conf)
            self.om.build()
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                self.om = Occlusion_Model(
                    images,
                    actions,
                    states,
                    iter_num=self.iter_num,
                    conf= conf)
                self.om.build()

        # L2 loss, PSNR for eval.
        loss, psnr_all = 0.0, 0.0

        for i, x, gx in zip(
                range(len(self.om.gen_images)), images[conf['context_frames']:],
                self.om.gen_images[conf['context_frames'] - 1:]):
            recon_cost_mse = mean_squared_error(x, gx)
            summaries.append(
                tf.scalar_summary(prefix + '_recon_cost' + str(i), recon_cost_mse))
            recon_cost = recon_cost_mse
            loss += recon_cost

        for i, state, gen_state in zip(
                range(len(self.om.gen_states)), states[conf['context_frames']:],
                self.om.gen_states[conf['context_frames'] - 1:]):
            state_cost = mean_squared_error(state, gen_state) * 1e-4 * conf['use_state']
            summaries.append(
                tf.scalar_summary(prefix + '_state_cost' + str(i), state_cost))
            loss += state_cost

        if 'mask_distinction_loss' in conf:
            dcost = self.distinction_loss(self.om.objectmasks) * conf['mask_distinction_loss']
            summaries.append(
                tf.scalar_summary(prefix + '_mask_distinction_cost', dcost))
            loss += dcost

        self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])

        summaries.append(tf.scalar_summary(prefix + '_loss', loss))

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)

    def distinction_loss(self, masks):
        delta = 0.
        for i in range(self.conf['num_masks']):
            for j in range(self.conf['num_masks']):
                if i == j:
                    continue
                delta -= tf.reduce_sum(tf.abs(masks[i]-masks[j]))
        return delta

def main(unused_argv, conf_script= None):

    if FLAGS.device ==-1:   # using cpu!
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tfconfig = None
    else:
        print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        tfconfig = tf.ConfigProto(gpu_options=gpu_options)

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

    print 'Constructing models and inputs.'
    with tf.variable_scope('model', reuse=None) as training_scope:
        images, actions, states = build_tfrecord_input(conf, training=True)
        model = Model(conf, images, actions, states)

    with tf.variable_scope('val_model', reuse=None):
        val_images, val_actions, val_states = build_tfrecord_input(conf, training=False)
        val_model = Model(conf, val_images, val_actions, val_states, training_scope)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)


    # Make training session.
    sess = tf.InteractiveSession(config= tfconfig)
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    if conf['visualize']:
        saver.restore(sess, conf['visualize'])

        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'vis',
                     val_model.iter_num: 0 }
        file_path = conf['output_dir']

        ground_truth, gen_images, object_masks, moved_parts, trafos, comp_factors = sess.run([
                                                        val_images,
                                                        val_model.om.gen_images,
                                                        val_model.om.objectmasks,
                                                        val_model.om.moved_imagesl,
                                                        val_model.om.list_of_trafos,
                                                        val_model.om.list_of_comp_factors
                                                        ],
                                                        feed_dict)
        dict_ = {}
        dict_['ground_truth'] = ground_truth
        dict_['gen_images'] = gen_images
        dict_['object_masks'] = object_masks
        dict_['moved_parts'] = moved_parts
        dict_['trafos'] = trafos
        dict_['comp_factors'] = comp_factors

        cPickle.dump(dict_, open(file_path + '/dict_.pkl', 'wb'))
        print 'written files to:' + file_path

        trajectories = makegifs.comp_gif(conf, conf['output_dir'])
        # utils_vpred.create_gif.comp_masks(conf['output_dir'], conf, trajectories)
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
            # oldfile = conf['output_dir'] + '/model' + str(itr - SAVE_INTERVAL)
            # if os.path.isfile(oldfile):
            #     os.system("rm {}".format(oldfile))
            #     os.system("rm {}".format(oldfile + '.meta'))
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
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
