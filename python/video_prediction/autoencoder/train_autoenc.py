import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import lsdc

from video_prediction.utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import video_prediction.utils_vpred.create_gif
from autoencoder_latentmodel import construct_model
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
                 actions=None,
                 states=None,
                 test = False
                 ):

        if test:
            self.images_01 = tf.placeholder(tf.float32, name='images',
                                            shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
            self.actions = tf.placeholder(tf.float32, name='actions',
                                     shape=(conf['batch_size'], conf['sequence_length'], 2))
            self.states = tf.placeholder(tf.float32, name='states',
                                    shape=(conf['batch_size'], conf['context_frames'], 4))

            self.images_23_rec = construct_model(conf,
                                                 self.images_01,
                                                 self.actions,
                                                 test=False)

        if not test:

            #images of size : batch_size, timesteps, 64,64,3
            ind_0 = tf.random_uniform(shape=np.array([1]), minval=0, maxval=conf['sequence_length']-4,
                                      dtype=tf.int64, seed=None, name=None)
            ind_1 = ind_0 + 1
            ind_2 = ind_0 + 2

            tzero = tf.constant(0, shape=np.array([1]), dtype=tf.int64)
            tzero3 = tf.zeros(shape=[3], dtype=tf.int64)

            self.images_01 = images_01 = tf.slice(images,
                                                  begin=tf.concat(0,[tzero,ind_0,tzero3]),
                                                  size=[-1,2,-1,-1,-1])
            self.images_23 = images_23 = tf.slice(images,
                                                  begin=tf.concat(0,[tzero,ind_2,tzero3]),
                                                  size=[-1,2,-1,-1,-1])
            self.states_01 = states_01 = tf.slice(images,
                                                  begin=tf.concat(0,[tzero,ind_0,tzero3]),
                                                  size=[-1,2,-1])
            self.states_23 = states_23 = tf.slice(images,
                                                  begin=tf.concat(0,[tzero,ind_2,tzero3]),
                                                  size=[-1,2,-1])
            self.action_1 = action_1 = tf.slice(actions,
                                                begin=tf.concat(0,[tzero,ind_1,tzero3]),
                                                size=[-1,2,-1])

        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []


        images_01_rec, pred_lt_state23, inf_lt_state23   = construct_model(conf,
                                                                            images_01,
                                                                            action_1,
                                                                            states_01,
                                                                            images_23,
                                                                            states_23,
                                                                            test = False)

        # L2 loss, PSNR for eval.
        loss = 0.0, 0.0

        recon_cost = mean_squared_error(images_01_rec, images_01)

        summaries.append(tf.scalar_summary(prefix + '_recon_cost', recon_cost))

        loss += recon_cost

        latent_state_cost = mean_squared_error(pred_lt_state23, inf_lt_state23) * 1e-4 * conf['use_state']
        summaries.append(tf.scalar_summary(prefix + 'latent_state_cost', latent_state_cost))
        loss += latent_state_cost

        self.loss = loss
        summaries.append(tf.scalar_summary(prefix + '_loss', loss))

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)


def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()


    conf_file = '/'.join(str.split(lsdc.__file__, '/')[:-3]) +\
                '/tensorflow_data/autoenc/' + FLAGS.hyper + '/conf.py'
    if not os.path.exists(conf_file):
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

    if FLAGS.visualize:
        return visualize(conf)

    print 'Constructing models and inputs.'
    with tf.variable_scope('model', reuse=None) as training_scope:
        images, actions, states = build_tfrecord_input(conf, training=True)
        images_val, actions_val, states_val = build_tfrecord_input(conf, training=False)

        train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")

        image, actions, states = tf.cond(train_cond > 0,  # if 1 use trainigbatch else validation batch
                                                   lambda: [images, actions, states],
                                                   lambda: [images_val, actions_val, states_val])
        model = Model(conf, image, actions, states)




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

    ### Begin Debug

    for s in range(5):

        images, actions, states = sess.run([images, actions, states])

        for b in range(4):
            image = Image.fromarray(np.uint8(image * 255)).show()

            print 'action shape:', actions.shape
            print 'actions:', actions
            print 'states shape:', states.shape
            print 'actions:', states

    pdb.set_trace()

    ### End Debug

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
                     model.lr: conf['learning_rate'],
                     train_cond: 1}
        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        # Print info: iteration #, cost.
        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {model.lr: 0.0,
                         model.prefix: 'val',
                         model.iter_num: np.float32(itr),
                         train_cond: 0}
            _, val_summary_str = sess.run([model.train_op, model.summ_op],
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


def visualize(conf):
    images, actions, states = build_tfrecord_input(conf, training=True)

    model = Model(conf, test=True)

    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())


    saver.restore(sess, conf['visualize'])

    images, actions, states = sess.run([images, actions, states])
    states = np.split(states, 1)
    images = np.split(images, 1)

    gen_images = [np.zeros([conf['batch_size'], 64, 64, 3]) for _ in range(conf['sequence_length'])]
    gen_images[0] = images[0]
    gen_images[1] = images[1]




    for t in range(1,conf['sequence_length']-1):
        images01 = np.zeros([conf['batch_size'], 2, 64, 64, 3])
        images01[:,0] = gen_images[t-1]
        images01[:,1] = gen_images[t]

        feed_dict ={
                    model.images_01: images01,
                    model.action_1: actions[t],
                     }

        gen_images[t+1] = sess.run([model.images_23_rec],
                                      feed_dict)



    file_path = conf['output_dir']
    cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl', 'wb'))
    cPickle.dump(images, open(file_path + '/ground_truth.pkl', 'wb'))
    print 'written files to:' + file_path
    trajectories = video_prediction.utils_vpred.create_gif.comp_video(conf['output_dir'], conf)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
