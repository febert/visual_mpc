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

from video_prediction.utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import video_prediction.utils_vpred.create_gif

from video_prediction.read_tf_record import build_tfrecord_input

from video_prediction.utils_vpred.skip_example import skip_example

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
                 reuse_scope=None,
                 pix_distrib=None):

        construct_model = conf['downsize']

        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        r_ind = np.empty(conf['batchsize'], 2)
        r_ind[:,0] = np.random.randint(0, conf['sequence_length']-1, size=conf['batchsize'])
        for i in range(conf['batchsize']):
            r_ind[i,1] = r_ind[i,0] + 1

        images = tf.gather_nd(images, r_ind)


        import pdb; pdb.set_trace()
        # Split into timesteps.
        images = tf.split(1, images.get_shape()[1], images)
        images = [tf.squeeze(img) for img in images]

        if pix_distrib != None:
            pix_distrib = tf.split(1, pix_distrib.get_shape()[1], pix_distrib)
            pix_distrib = [tf.squeeze(pix) for pix in pix_distrib]

        if reuse_scope is None:
            gen_images, gen_states, gen_masks, gen_distrib = construct_model(
                images,
                iter_num=self.iter_num,
                num_masks=conf['num_masks'],
                cdna=conf['model'] == 'CDNA',
                dna=conf['model'] == 'DNA',
                stp=conf['model'] == 'STP',
                pix_distributions= pix_distrib)
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images, gen_states, gen_masks, gen_distrib = construct_model(
                    images,
                    iter_num=self.iter_num,
                    num_masks=conf['num_masks'],
                    cdna=conf['model'] == 'CDNA',
                    dna=conf['model'] == 'DNA',
                    stp=conf['model'] == 'STP',
                    pix_distributions=pix_distrib)

        # L2 loss, PSNR for eval.
        loss, psnr_all = 0.0, 0.0

        loss = mean_squared_error(images, gen_images)
        psnr_i = peak_signal_to_noise_ratio(images, gen_images)
        psnr_all += psnr_i
        summaries.append(
            tf.scalar_summary(prefix + '_recon_cost', loss))
        summaries.append(tf.scalar_summary(prefix + '_psnr', psnr_i))

        self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])

        summaries.append(tf.scalar_summary(prefix + '_loss', loss))

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)

        self.gen_images= gen_images
        self.gen_masks = gen_masks
        self.gen_distrib = gen_distrib


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

    print 'Constructing models and inputs.'
    with tf.variable_scope('model', reuse=None) as training_scope:
        images, actions, states = build_tfrecord_input(conf, training=True)
        model = Model(conf, images, actions, states, conf['sequence_length'], pix_distrib= )

    with tf.variable_scope('val_model', reuse=None):
        val_images, val_actions, val_states = build_tfrecord_input(conf, training=False)
        val_model = Model(conf, val_images, val_actions, val_states,
                          conf['sequence_length'], training_scope)

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
        saver.restore(sess, conf['visualize'])

        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'vis',
                     val_model.iter_num: 0 }
        gen_images, ground_truth, mask_list = sess.run([val_model.gen_images,
                                                        val_images, val_model.gen_masks],
                                                       feed_dict)
        file_path = conf['output_dir']
        cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl','wb'))
        cPickle.dump(ground_truth, open(file_path + '/ground_truth.pkl', 'wb'))
        cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
        print 'written files to:' + file_path

        trajectories = video_prediction.utils_vpred.create_gif.comp_video(conf['output_dir'], conf)
        video_prediction.utils_vpred.create_gif.comp_masks(conf['output_dir'], conf, trajectories)
        return

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
        feed_dict = {model.prefix: 'train',
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
                         val_model.prefix: 'val',
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
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
