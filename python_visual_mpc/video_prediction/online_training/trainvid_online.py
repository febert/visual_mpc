import os
import numpy as np
import tensorflow as tf
import imp
import sys
import pickle
import pdb

import imp
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from python_visual_mpc.video_prediction.utils_vpred.video_summary import convert_tensor_to_gif_summary

from datetime import datetime
import collections
import time
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

VIDEO_INTERVAL = 10000


from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher


def trainvid_online(replay_buffer, conf, gpu_id):
    os.environ["cuda_visible_devices"] = str(gpu_id)
    print('training video prediction using cuda_visible_devices=', gpu_id)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    pdb.set_trace()

    if 'RESULT_DIR' in os.environ:
        conf['output_dir'] = os.environ['RESULT_DIR'] + '/modeldata'
    conf['event_log_dir'] = conf['output_dir']

    Model = conf['pred_model']
    model = Model(conf, load_data=False, trafo_pix=False, build_loss=True)

    print('Constructing saver.')
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars = filter_vars(vars)
    saving_saver = tf.train.Saver(vars, max_to_keep=0)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    g_vidpred= tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True), graph=g_vidpred)
    with sess.as_default():
        with g_vidpred.as_default():
            summary_writer = tf.summary.FileWriter(conf['event_log_dir'], graph=sess.graph, flush_secs=10)

            vars = variable_checkpoint_matcher(conf, vars, conf['pretrained'])
            loading_saver = tf.train.Saver(vars, max_to_keep=0)
            load_checkpoint(conf, sess, loading_saver, conf['pretrained'])

            print('-------------------------------------------------------------------')
            print('verify current settings!! ')
            for key in list(conf.keys()):
                print(key, ': ', conf[key])
            print('-------------------------------------------------------------------')

            tf.logging.set_verbosity(tf.logging.INFO)

            starttime = datetime.now()
            t_iter = []
            for itr in range(0, conf['num_iterations'], 1):
                tstart_rb_update = time.time()
                replay_buffer.update()
                # print("took {} to update the replay buffer".format(time.time() - tstart_rb_update))

                t_startiter = datetime.now()
                images, states, actions = replay_buffer.get_batch()
                feed_dict = {model.iter_num: np.float32(itr),
                             model.train_cond: 1,
                             model.images_pl:images,
                             model.actions_pl:actions,
                             model.states_pl:states
                             }
                cost, _, summary_str = sess.run([model.loss, model.train_op, model.train_summ_op],
                                                feed_dict)

                t_iter.append((datetime.now() - t_startiter).seconds * 1e6 +  (datetime.now() - t_startiter).microseconds )

                if (itr) % 10 ==0:
                    tf.logging.info(str(itr) + ' ' + str(cost))

                if (itr) % VIDEO_INTERVAL == 2 and hasattr(model, 'val_video_summaries'):
                    feed_dict = {model.iter_num: np.float32(itr),
                                 model.train_cond: 0,
                                 model.images_pl:images,
                                 model.actions_pl:actions,
                                 model.states_pl:states
                                 }
                    video_proto = sess.run(model.val_video_summaries, feed_dict = feed_dict)
                    summary_writer.add_summary(convert_tensor_to_gif_summary(video_proto), itr)

                if (itr) % conf['save_interval'] == 2:
                    tf.logging.info('Saving model to' + conf['output_dir'])
                    saving_saver.save(sess, conf['output_dir'] + '/model' + str(itr))


                if itr % 50 == 1:
                    hours = (datetime.now() -starttime).seconds/3600
                    tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                        (datetime.now() - starttime).days,
                        hours,+
                        (datetime.now() - starttime).seconds/60 - hours*60))
                    avg_t_iter = np.sum(np.asarray(t_iter))/len(t_iter)
                    tf.logging.info('time per iteration: {0}'.format(avg_t_iter/1e6))
                    tf.logging.info('expected for complete training: {0}h '.format(avg_t_iter /1e6/3600 * conf['num_iterations']))

                if (itr) % SUMMARY_INTERVAL == 2:
                    summary_writer.add_summary(summary_str, itr)

            sess.close()
            tf.reset_default_graph()
            return np.array(t_iter)/1e6

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
