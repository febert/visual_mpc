from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb
import os
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import imp
import sys
import pickle
import pdb


import argparse
import errno
import itertools
import json
import math
import os
import random
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import datasets, models
from video_prediction.utils import ffmpeg_gif, tf_utils
import time
import imp
import itertools

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

VIDEO_INTERVAL = 5000

EVAL_INTERVAL = 0

PROGRESS_INTERVAL = 50



from collections import namedtuple
Traj = namedtuple('Traj', 'images X_Xdot_full actions')
from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher


def trainvid_online_alexmodel(train_replay_buffer, val_replay_buffer, conf, logging_dir, gpu_id, printout=False):
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

    Model = conf['pred_model']
    train_model = Model(conf, mode='train')
    with tf.variable_scope("", reuse=True):
        val_model = Model(conf, mode='val')

    saver = tf.train.Saver(max_to_keep=3)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    image_summaries = set(tf.get_collection(tf_utils.IMAGE_SUMMARIES))
    eval_summaries = set(tf.get_collection(tf_utils.EVAL_SUMMARIES))
    eval_image_summaries = image_summaries & eval_summaries
    image_summaries -= eval_image_summaries
    eval_summaries -= eval_image_summaries
    if SUMMARY_INTERVAL:
        summary_op = tf.summary.merge(summaries)
    if VIDEO_INTERVAL:
        image_summary_op = tf.summary.merge(list(image_summaries))
    if EVAL_INTERVAL:
        eval_summary_op = tf.summary.merge(list(eval_summaries))
        eval_image_summary_op = tf.summary.merge(list(eval_image_summaries))

    if SUMMARY_INTERVAL or VIDEO_INTERVAL or EVAL_INTERVAL:
        summary_writer = tf.summary.FileWriter(conf['output_dir'])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    global_step = tf.train.get_or_create_global_step()
    max_steps = train_model.m.hparams.max_steps
    batch_size = train_model.m.hparams.batch_size
    elapsed_times = []
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        if gfile.Glob(conf['pretrained_model'] + '*') is None:
            raise ValueError("Model file {} not found!".format(conf['pretrained_model']))
        train_model.m.restore(sess, conf['pretrained_model'])

        train_replay_buffer.preload(sess)
        val_replay_buffer.preload(sess)

        start_step = sess.run(global_step)
        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        for step in range(-1, max_steps - start_step):
            if step == 0:
                start = time.time()

            def should(freq):
                return freq and ((step + 1) % freq == 0 or (step + 1) in (0, max_steps - start_step))

            if step % 10 == 0:
                tstart_rb_update = time.time()
                train_replay_buffer.update(sess)
                val_replay_buffer.update(sess)
                if step % 100 == 0:
                    logger.log("took {} to update the replay buffer".format(time.time() - tstart_rb_update))

            fetches, feeds = {}, {}
            if step >= 0:
                fetches["train_op"] = train_model.m.train_op

            if should(PROGRESS_INTERVAL):
                fetches['d_losses'] = train_model.m.d_losses
                fetches['g_losses'] = train_model.m.g_losses
                if isinstance(train_model.m.learning_rate, tf.Tensor):
                    fetches["learning_rate"] = train_model.m.learning_rate
            if should(SUMMARY_INTERVAL):
                fetches["summary"] = summary_op
            if should(VIDEO_INTERVAL):
                fetches["image_summary"] = image_summary_op
            if should(VIDEO_INTERVAL) or should(SUMMARY_INTERVAL):
                val_images, val_states, val_actions = val_replay_buffer.get_batch()
                feeds = {val_model.images_pl: val_images,
                        val_model.actions_pl: val_actions,
                        val_model.states_pl: val_states}
            if should(EVAL_INTERVAL):
                fetches["eval_summary"] = eval_summary_op
                fetches["eval_image_summary"] = eval_image_summary_op

            images, states, actions = train_replay_buffer.get_batch()
            feeds.update({train_model.images_pl: images,
                          train_model.actions_pl: actions,
                          train_model.states_pl: states})

            run_start_time = time.time()
            results = sess.run(fetches, feed_dict=feeds)
            run_elapsed_time = time.time() - run_start_time
            elapsed_times.append(run_elapsed_time)
            if run_elapsed_time > 1.5:
                print('session.run took %0.1fs' % run_elapsed_time)

            if should(PROGRESS_INTERVAL) or should(SUMMARY_INTERVAL):
                if step >= 0:
                    elapsed_time = time.time() - start
                    average_time = elapsed_time / (step + 1)
                    images_per_sec = batch_size / average_time
                    remaining_time = (max_steps - (start_step + step)) * average_time

            if should(PROGRESS_INTERVAL):
                print("global step %d" % (global_step.eval()))
                if step >= 0:
                    print("          image/sec %0.1f  remaining %dm (%0.1fh) (%0.1fd)" %
                          (
                          images_per_sec, remaining_time / 60, remaining_time / 60 / 60, remaining_time / 60 / 60 / 24))

                for name, loss in itertools.chain(results['d_losses'].items(), results['g_losses'].items()):
                    print(name, loss)
                if isinstance(train_model.m.learning_rate, tf.Tensor):
                    print("learning_rate", results["learning_rate"])

            if should(SUMMARY_INTERVAL):
                print("recording summary")
                summary_writer.add_summary(results["summary"], global_step.eval())
                if step >= 0:
                    try:
                        from tensorboard.summary import scalar_pb
                        for name, scalar in zip(['images_per_sec', 'remaining_hours'],
                                                [images_per_sec, remaining_time / 60 / 60]):
                            summary_writer.add_summary(scalar_pb(name, scalar), global_step.eval())
                    except ImportError:
                        pass

                print("done")
            if should(VIDEO_INTERVAL):
                print("recording image summary")
                summary_writer.add_summary(
                    tf_utils.convert_tensor_to_gif_summary(results["image_summary"]), global_step.eval())
                print("done")
            if should(EVAL_INTERVAL):
                print("recording eval summary")
                summary_writer.add_summary(results["eval_summary"], global_step.eval())
                summary_writer.add_summary(
                    tf_utils.convert_tensor_to_gif_summary(results["eval_image_summary"]), global_step.eval())
                print("done")

            if should(SUMMARY_INTERVAL) or should(VIDEO_INTERVAL) or should(EVAL_INTERVAL):
                summary_writer.flush()

            if should(conf['onpolconf']['save_interval']):
                print("saving model to", conf['output_dir'])
                saver.save(sess, os.path.join(conf['output_dir'], "model"), global_step=global_step)
                print("done")

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
