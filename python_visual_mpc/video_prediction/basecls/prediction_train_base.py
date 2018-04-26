import os
import numpy as np
import tensorflow as tf
import imp
import sys
import pickle
import pdb

import imp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from datetime import datetime
import collections
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 400

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

# How often to save a model checkpoint
SAVE_INTERVAL = 4000
from .utils.get_designated_pix import Getdesig

from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter

from python_visual_mpc.video_prediction.tracking_model.tracking_model import Tracking_Model
from python_visual_mpc.flow.descriptor_based_flow.descriptor_flow_model import Descriptor_Flow
from python_visual_mpc.flow.trafo_based_flow.correction import Trafo_Flow

from PIL import Image


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_bool('visualize', '', 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_string('visualize_check', "", 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')
    flags.DEFINE_bool('diffmotions', False, 'visualize several different motions for a single scene')

    flags.DEFINE_bool('metric', False, 'compute metric of expected distance to human-labled positions ob objects')
    flags.DEFINE_bool('create_images', False, 'whether to create images')


def main(unused_argv, conf_script= None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print('using CUDA_VISIBLE_DEVICES=', FLAGS.device)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    if conf_script == None: conf_file = FLAGS.hyper
    else: conf_file = conf_script

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    if FLAGS.visualize:
        print('creating visualizations ...')
        conf['schedsamp_k'] = -1  # don't feed ground truth
        if 'test_data_dir' in conf:
            conf['data_dir'] = conf['test_data_dir']
        else:
            conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])

        if FLAGS.visualize_check:
            conf['visualize_check'] = conf['output_dir'] + '/' + FLAGS.visualize_check
        conf['visualize'] = True

        conf['event_log_dir'] = '/tmp'
        conf.pop('use_len', None)
        conf['batch_size'] = 128

        conf['sequence_length'] = 14
        if FLAGS.diffmotions:
            conf['sequence_length'] = 30

    from .prediction_model_basecls import Base_Prediction_Model
    if 'pred_model' in conf:
        Model = conf['pred_model']
    else:
        Model = Base_Prediction_Model

    # with tf.variable_scope('generator'):  ################# just for making compatible with old training code
    if FLAGS.diffmotions or "visualize_tracking" in conf or FLAGS.metric:
        model = Model(conf, load_data=False, trafo_pix=True)
    else:
        model = Model(conf, load_data=True, trafo_pix=False)

    print('Constructing saver.')
    # Make saver.

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # remove all states from group of variables which shall be saved and restored:
    vars_no_state = filter_vars(vars)

    print('vars to be restored:')
    print('_____________')
    for var in vars:
        print(var.name)
    pdb.set_trace()

    saver = tf.train.Saver(vars_no_state, max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    if conf['visualize']:
        load_checkpoint(conf, sess, saver)
        print('-------------------------------------------------------------------')
        print('verify current settings!! ')
        for key in list(conf.keys()):
            print(key, ': ', conf[key])
        print('-------------------------------------------------------------------')

        if FLAGS.diffmotions:
            model.visualize_diffmotions(sess)
        elif FLAGS.metric:
            model.compute_metric(sess, FLAGS.create_images)
        else:
            model.visualize(sess)

        return

    itr_0 = 0
    if FLAGS.pretrained != None:
        itr_0 = load_checkpoint(conf, sess, saver, model_file=FLAGS.pretrained)
        print('resuming training at iteration: ', itr_0)

    print('-------------------------------------------------------------------')
    print('verify current settings!! ')
    for key in list(conf.keys()):
        print(key, ': ', conf[key])
    print('-------------------------------------------------------------------')

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter = []
    # Run training.

    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()
        # Generate new batch of data_files.
        feed_dict = {model.iter_num: np.float32(itr),
                     model.train_cond: 1}

        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {model.iter_num: np.float32(itr),
                         model.train_cond: 0}
            [val_summary_str] = sess.run([model.summ_op], feed_dict)
            summary_writer.add_summary(val_summary_str, itr)

        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model to' + conf['output_dir'])
            saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        t_iter.append((datetime.now() - t_startiter).seconds * 1e6 +  (datetime.now() - t_startiter).microseconds )

        if itr % 100 == 1:
            hours = (datetime.now() -starttime).seconds/3600
            tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                (datetime.now() - starttime).days,
                hours,+
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



def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print('removed state variable from saving-list: ', v.name)

    return newlist

def load_checkpoint(conf, sess, saver, model_file=None):
    """
    :param sess:
    :param saver:
    :param model_file: filename with model***** but no .data, .index etc.
    :return:
    """
    import re
    if model_file is not None:
        saver.restore(sess, model_file)
        num_iter = int(re.match('.*?([0-9]+)$', model_file).group(1))
    else:
        ckpt = tf.train.get_checkpoint_state(conf['output_dir'])
        print(("loading " + ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        num_iter = int(re.match('.*?([0-9]+)$', ckpt.model_checkpoint_path).group(1))
    conf['num_iter'] = num_iter
    return num_iter



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
