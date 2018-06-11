import os
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import imp
import sys
import pickle
import pdb

import time
import imp
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from python_visual_mpc.video_prediction.dynamic_rnn_model.alex_model_interface import Alex_Interface_Model
from python_visual_mpc.video_prediction.read_tf_records2 import build_tfrecord_input
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
from python_visual_mpc.video_prediction.utils_vpred.video_summary import convert_tensor_to_gif_summary

from datetime import datetime
import collections
import time
import copy
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

VIDEO_INTERVAL = 10000

from collections import namedtuple
Traj = namedtuple('Traj', 'images X_Xdot_full actions')
from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher


def trainvid_online(train_replay_buffer, val_replay_buffer, conf, logging_dir, gpu_id, printout=False):
    logger = Logger(logging_dir, 'trainvid_online_log.txt', printout=printout)
    logger.log('starting trainvid online')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.log('training video prediction using cuda_visible_devices=', os.environ["CUDA_VISIBLE_DEVICES"])
    from tensorflow.python.client import device_lib
    logger.log(device_lib.list_local_devices())

    if 'RESULT_DIR' in os.environ:
        conf['output_dir'] = os.environ['RESULT_DIR'] + '/modeldata'
    conf['event_log_dir'] = conf['output_dir']

    if 'TEN_DATA' in os.environ:
        tenpath = conf['pretrained_model'].partition('tensorflow_data')[2]
        conf['pretrained_model'] = os.environ['TEN_DATA'] + tenpath

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    g_vidpred = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True), graph=g_vidpred)
    with sess.as_default():
        with g_vidpred.as_default():
            tf.train.start_queue_runners(sess)
            sess.run(tf.global_variables_initializer())

            train_replay_buffer.preload(conf)

            Model = conf['pred_model']
            model = Model(conf, load_data=False, build_loss=True)
            logger.log('Constructing saver.')
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars = filter_vars(vars)
            saving_saver = tf.train.Saver(vars, max_to_keep=0)
            summary_writer = tf.summary.FileWriter(conf['event_log_dir'], graph=sess.graph, flush_secs=10)

            if conf['pred_model'] == Alex_Interface_Model:
                if gfile.Glob(conf['pretrained_model'] + '*') is None:
                    raise ValueError("Model file {} not found!".format(conf['pretrained_model']))
                model.m.restore(sess, conf['pretrained_model'])
            else:
                vars = variable_checkpoint_matcher(conf, vars, conf['pretrained_model'])
                loading_saver = tf.train.Saver(vars, max_to_keep=0)
                load_checkpoint(conf, sess, loading_saver, conf['pretrained_model'])

            logger.log('-------------------------------------------------------------------')
            logger.log('verify current settings!! ')
            for key in list(conf.keys()):
                logger.log(key, ': ', conf[key])
            logger.log('-------------------------------------------------------------------')

            tf.logging.set_verbosity(tf.logging.INFO)

            starttime = time.time()
            t_iter = []
            for itr in range(0, conf['num_iterations'], 1):

                if itr % 10 == 0:
                    tstart_rb_update = time.time()
                    train_replay_buffer.update(sess)
                    if itr % 100 == 0:
                        logger.log("took {} to update the replay buffer".format(time.time() - tstart_rb_update))

                t_startiter = time.time()
                images, states, actions = train_replay_buffer.get_batch()
                feed_dict = {model.iter_num: np.float32(itr),
                             model.images_pl: images,
                             model.actions_pl: actions,
                             model.states_pl: states
                             }
                if conf['pred_model'] == Alex_Interface_Model:
                    cost, _, summary_str = sess.run([model.m.g_loss, model.m.train_op, model.m.train_summ_op], feed_dict)
                else:
                    cost, _, summary_str = sess.run([model.loss, model.train_op, model.train_summ_op], feed_dict)
                t_iter.append(time.time() - t_startiter)

                if (itr) % 10 == 0:
                    logger.log('cost ' + str(itr) + ' ' + str(cost))

                if (itr) % VAL_INTERVAL == 0:
                    val_replay_buffer.update(sess)
                    images, states, actions = val_replay_buffer.get_batch()
                    feed_dict = {model.iter_num: np.float32(itr),
                                 model.images_pl: images,
                                 model.actions_pl: actions,
                                 model.states_pl: states
                                 }
                    if conf['pred_model'] == Alex_Interface_Model:
                        [summary_str] = sess.run([model.m.val_summ_op], feed_dict)
                    else:
                        [summary_str] = sess.run([model.val_summ_op], feed_dict)
                    summary_writer.add_summary(summary_str, itr)

                if (itr) % VIDEO_INTERVAL == 0:
                    feed_dict = {model.iter_num: np.float32(itr),
                                 model.images_pl: images,
                                 model.actions_pl: actions,
                                 model.states_pl: states
                                 }
                    video_proto = sess.run(model.train_video_summaries, feed_dict=feed_dict)
                    summary_writer.add_summary(convert_tensor_to_gif_summary(video_proto), itr)

                save_interval = conf['onpolconf']['save_interval']
                if (itr) % save_interval == 0 and itr != 0:
                    oldmodelname = conf['output_dir'] + '/model' + str(itr-save_interval )
                    if gfile.Glob(oldmodelname + '*') != [] and (itr - save_interval ) > 0:
                        print('deleting {}*'.format(oldmodelname))
                        os.system("rm {}*".format(oldmodelname))
                    logger.log('Saving model to' + conf['output_dir'])
                    newmodelname = conf['output_dir'] + '/model' + str(itr)
                    saving_saver.save(sess, newmodelname)

                if itr % 50 == 1:
                    hours = (time.time()- starttime) / 3600
                    logger.log('running for  {}h'.format(hours))
                    avg_t_iter = np.mean(t_iter[-100:])
                    logger.log('average time per iteration: {0}s'.format(avg_t_iter))
                    logger.log('expected for complete training: {0}h '.format(avg_t_iter / 3600 * conf['num_iterations']))

                if (itr) % SUMMARY_INTERVAL == 2:
                    summary_writer.add_summary(summary_str, itr)
            return t_iter




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
        print("loading " + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        num_iter = int(re.match('.*?([0-9]+)$', ckpt.model_checkpoint_path).group(1))
    conf['num_iter'] = num_iter
    return num_iter

def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print('removed state variable from saving-list: ', v.name)
    return newlist

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
