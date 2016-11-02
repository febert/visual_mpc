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

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from read_tf_record import build_tfrecord_input

from utils_vpred.skip_example import skip_example

from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

# tf record data location:
DATA_DIR = '/home/frederik/Documents/pushing_data/train'

# local output directory
OUT_DIR = '/home/frederik/Documents/lsdc/tensorflow_data'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 15,
                     'sequence length, including context frames.')

flags.DEFINE_integer('skip_frame', 1,
                     'use ever ith frame to increase prediction horizon')

flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'CDNA',
                    'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 900.0,
                   'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')

flags.DEFINE_string('visualize', '',
                  'load model from which to generate visualizations, dont forget to set schedsamp_k to -1')

flags.DEFINE_bool('downsize', 'False',
                  'load downsized model')


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
                 images=None,
                 actions=None,
                 states=None,
                 sequence_length=None,
                 reuse_scope=None):

        if FLAGS.downsize:
            from prediction_model_downsized import construct_model
        else:
            from prediction_model import construct_model

        if sequence_length is None:
            sequence_length = FLAGS.sequence_length

        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        # Split into timesteps.
        actions = tf.split(1, actions.get_shape()[1], actions)
        actions = [tf.squeeze(act) for act in actions]
        states = tf.split(1, states.get_shape()[1], states)
        states = [tf.squeeze(st) for st in states]
        images = tf.split(1, images.get_shape()[1], images)
        images = [tf.squeeze(img) for img in images]

        print 'scheduled sampling: k=', FLAGS.schedsamp_k

        if reuse_scope is None:
            gen_images, gen_states, gen_masks = construct_model(
                images,
                actions,
                states,
                iter_num=self.iter_num,
                k=FLAGS.schedsamp_k,
                use_state=FLAGS.use_state,
                num_masks=FLAGS.num_masks,
                cdna=FLAGS.model == 'CDNA',
                dna=FLAGS.model == 'DNA',
                stp=FLAGS.model == 'STP',
                context_frames=FLAGS.context_frames)
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images, gen_states, gen_masks = construct_model(
                    images,
                    actions,
                    states,
                    iter_num=self.iter_num,
                    k=FLAGS.schedsamp_k,
                    use_state=FLAGS.use_state,
                    num_masks=FLAGS.num_masks,
                    cdna=FLAGS.model == 'CDNA',
                    dna=FLAGS.model == 'DNA',
                    stp=FLAGS.model == 'STP',
                    context_frames=FLAGS.context_frames)

        # L2 loss, PSNR for eval.
        loss, psnr_all = 0.0, 0.0
        for i, x, gx in zip(
                range(len(gen_images)), images[FLAGS.context_frames:],
                gen_images[FLAGS.context_frames - 1:]):
            recon_cost = mean_squared_error(x, gx)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(
                tf.scalar_summary(prefix + '_recon_cost' + str(i), recon_cost))
            summaries.append(tf.scalar_summary(prefix + '_psnr' + str(i), psnr_i))
            loss += recon_cost

        for i, state, gen_state in zip(
                range(len(gen_states)), states[FLAGS.context_frames:],
                gen_states[FLAGS.context_frames - 1:]):
            state_cost = mean_squared_error(state, gen_state) * 1e-4
            summaries.append(
                tf.scalar_summary(prefix + '_state_cost' + str(i), state_cost))
            loss += state_cost
        summaries.append(tf.scalar_summary(prefix + '_psnr_all', psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - FLAGS.context_frames)

        summaries.append(tf.scalar_summary(prefix + '_loss', loss))

        self.lr = tf.placeholder_with_default(FLAGS.learning_rate, ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)

        self.gen_images= gen_images
        self.gen_masks = gen_masks


def main(unused_argv):
    if FLAGS.visualize:
        FLAGS.schedsamp_k = -1

    print 'Constructing models and inputs.'
    with tf.variable_scope('model', reuse=None) as training_scope:
        images, actions, states = build_tfrecord_input(training=True)
        # if FLAGS.skip_frame:
        #     images, actions, states = skip_example(images, actions, states)
        model = Model(images, actions, states, FLAGS.sequence_length)

    with tf.variable_scope('val_model', reuse=None):
        val_images, val_actions, val_states = build_tfrecord_input(training=False)
        # if FLAGS.skip_frame:
        #     val_images, val_actions, val_states = skip_example(val_images, val_actions, val_states)
        val_model = Model(val_images, val_actions, val_states,
                          FLAGS.sequence_length, training_scope)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.train.SummaryWriter(
        FLAGS.event_log_dir, graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    if FLAGS.visualize:
        saver.restore(sess, FLAGS.visualize)

        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'vis',
                     val_model.iter_num: 0 }
        gen_images, ground_truth, mask_list = sess.run([val_model.gen_images, val_images, val_model.gen_masks], feed_dict)
        splitted = str.split(os.path.dirname(__file__), '/')
        file_path = '/'.join(splitted[:-2] + ['tensorflow_data/gifs'])

        import cPickle
        cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl','wb'))
        cPickle.dump(ground_truth, open(file_path + '/ground_truth.pkl', 'wb'))
        cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
        print 'written files to:' + file_path

        return

    itr_0 =0
    if FLAGS.pretrained_model:        # is the order of initialize_all_variables() and restore() important?!?
        saver.restore(sess, FLAGS.pretrained_model)
        # resume training at iteration step of the loaded model:
        import re
        itr_0 = re.match('.*?([0-9]+)$', FLAGS.pretrained_model).group(1)
        itr_0 = int(itr_0)
        print 'resuming training at iteration:  ', itr_0

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()

    # Run training.
    for itr in range(itr_0,FLAGS.num_iterations,1):
        t_startiter = datetime.now()
        # Generate new batch of data.
        feed_dict = {model.prefix: 'train',
                     model.iter_num: np.float32(itr),
                     model.lr: FLAGS.learning_rate}
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
            tf.logging.info('Saving model to' + FLAGS.output_dir)
            saver.save(sess, FLAGS.output_dir + '/model' + str(itr))

        if itr % 100 == 2:
            hours = (datetime.now() -starttime).seconds/3600
            tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                (datetime.now() - starttime).days,
                hours,
                (datetime.now() - starttime).seconds/60 - hours*60))
            t_iter = (datetime.now()- t_startiter).seconds*1e6 + (datetime.now()- t_startiter).microseconds
            tf.logging.info('time per iteration: {0}'.format(t_iter/1e6))
            tf.logging.info('expected time until completion: {0}h '.format(t_iter* FLAGS.num_iterations/1e6/3600))

        if (itr) % SUMMARY_INTERVAL:
            summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, FLAGS.output_dir + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
