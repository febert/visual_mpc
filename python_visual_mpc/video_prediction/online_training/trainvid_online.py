import os
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import imp
import sys
import pickle
import pdb

import imp
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from python_visual_mpc.video_prediction.read_tf_records2 import build_tfrecord_input
from python_visual_mpc.visual_mpc_core.infrastructure.utility.logger import Logger
from python_visual_mpc.video_prediction.utils_vpred.video_summary import convert_tensor_to_gif_summary

from datetime import datetime
import collections
import time
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

VIDEO_INTERVAL = 10000

from collections import namedtuple
Traj = namedtuple('Traj', 'images X_Xdot_full actions')
from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher


def trainvid_online(replay_buffer, conf, agentparams, onpolparam, gpu_id):
    logger = Logger(agentparams['logging_dir'], 'trainvid_online_log.txt')
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

            preload_replay(conf, logger, onpolparam, replay_buffer, sess)

            Model = conf['pred_model']
            ###############################
            # model = Model(conf, load_data=False, trafo_pix=False, build_loss=True)
            model = Model(conf, load_data=True, trafo_pix=False, build_loss=True)
            ###########################
            logger.log('Constructing saver.')
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars = filter_vars(vars)
            saving_saver = tf.train.Saver(vars, max_to_keep=0)
            summary_writer = tf.summary.FileWriter(conf['event_log_dir'], graph=sess.graph, flush_secs=10)
            vars = variable_checkpoint_matcher(conf, vars, conf['pretrained_model'])
            loading_saver = tf.train.Saver(vars, max_to_keep=0)
            load_checkpoint(conf, sess, loading_saver, conf['pretrained_model'])

            logger.log('-------------------------------------------------------------------')
            logger.log('verify current settings!! ')
            for key in list(conf.keys()):
                logger.log(key, ': ', conf[key])
            logger.log('-------------------------------------------------------------------')

            tf.logging.set_verbosity(tf.logging.INFO)

            starttime = datetime.now()
            t_iter = []
            for itr in range(0, conf['num_iterations'], 1):
                tstart_rb_update = time.time()
                replay_buffer.update()
                logger.log("took {} to update the replay buffer".format(time.time() - tstart_rb_update))

                t_startiter = datetime.now()
                images, states, actions = replay_buffer.get_batch()
                feed_dict = {model.iter_num: np.float32(itr),
                             model.train_cond: 1,
                             model.images_pl: images,
                             model.actions_pl: actions,
                             model.states_pl: states
                             }

                cost, _, summary_str = sess.run([model.loss, model.train_op, model.train_summ_op],
                                                feed_dict)
                t_iter.append((datetime.now() - t_startiter).seconds * 1e6 + (datetime.now() - t_startiter).microseconds)

                if (itr) % 10 == 0:
                    logger.log(str(itr) + ' ' + str(cost))

                if (itr) % VIDEO_INTERVAL == 2 and hasattr(model, 'val_video_summaries'):
                    # feed_dict = {model.iter_num: np.float32(itr),
                    #              model.train_cond: 1,
                    #              model.images_pl: images,
                    #              model.actions_pl: actions,
                    #              model.states_pl: states
                    #              }
                    feed_dict = {model.iter_num: np.float32(itr),
                                 model.train_cond: 1,
                                 }
                    video_proto = sess.run(model.val_video_summaries, feed_dict=feed_dict)
                    summary_writer.add_summary(convert_tensor_to_gif_summary(video_proto), itr)

                if (itr) % conf['save_interval'] == 0:
                    logger.log('Saving model to' + conf['output_dir'])
                    newmodelname = conf['output_dir'] + '/model' + str(itr)
                    oldmodelname = conf['output_dir'] + '/model' + str(itr-conf['save_interval'])
                    saving_saver.save(sess, newmodelname)
                    if len(gfile.Glob(os.path.join(conf['data_dir'], '*'))) != 0:
                        print('deleting {}'.format(oldmodelname))
                        os.system("rm {}".format(oldmodelname))

                if itr % 50 == 1:
                    hours = (datetime.now() - starttime).seconds / 3600
                    logger.log('running for {0}d, {1}h, {2}min'.format(
                        (datetime.now() - starttime).days,
                        hours, +
                               (datetime.now() - starttime).seconds / 60 - hours * 60))
                    avg_t_iter = np.sum(np.asarray(t_iter)) / len(t_iter)
                    logger.log('time per iteration: {0}'.format(avg_t_iter / 1e6))
                    logger.log('expected for complete training: {0}h '.format(avg_t_iter / 1e6 / 3600 * conf['num_iterations']))

                if (itr) % SUMMARY_INTERVAL == 2:
                    summary_writer.add_summary(summary_str, itr)
            return t_iter


def preload_replay(conf, logger, onpolparam, replay_buffer, sess):
    logger.log('start filling replay')
    dict = build_tfrecord_input(conf, training=True)
    for i_run in range(onpolparam['fill_replay_fromsaved'] // conf['batch_size']):
        images, actions, endeff = sess.run([dict['images'], dict['actions'], dict['endeffector_pos']])
        for b in range(conf['batch_size']):
            t = Traj(images[b], endeff[b], actions[b])
            replay_buffer.push_back(t)
    logger.log('done filling replay')


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
