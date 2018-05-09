#########
# reads a regular dataset, performs video prediction and then saves the predicted images to a new dataset


import copy
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


from datetime import datetime
import collections
import time
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 400

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

# How often to save a model checkpoint
SAVE_INTERVAL = 4000

from python_visual_mpc.video_prediction.tracking_model.single_point_tracking_model import Single_Point_Tracking_Model
from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_bool('model', False, 'visualize latest checkpoint')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')

class Trajectory(object):
    def __init__(self):
        self.images, self.X_Xdot_full, self.actions, self.gen_images, self.gen_states = None, None, None, None, None


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

    print('creating visualizations ...')
    conf['schedsamp_k'] = -1  # don't feed ground truth


    conf['visualize'] = True # deactivates train/val splitting in the data reader.
    conf['event_log_dir'] = '/tmp'
    conf.pop('use_len', None)   # don't perform random shifting

    Model = conf['pred_model']
    model = Model(conf, load_data=True, trafo_pix=False, build_loss=False)

    print('Constructing saver.')
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars = filter_vars(vars)

    model_file = conf['pretrained_model']
    vars = variable_checkpoint_matcher(conf, vars, model_file)
    loading_saver = tf.train.Saver(vars, max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    tf.train.start_queue_runners(sess)

    sess.run(tf.global_variables_initializer())

    load_checkpoint(conf, sess, loading_saver, model_file)

    print('-------------------------------------------------------------------')
    print('verify current settings!! ')
    for key in list(conf.keys()):
        print(key, ': ', conf[key])
    print('-------------------------------------------------------------------')

    itr_0 =0

    print('-------------------------------------------------------------------')
    print('verify current settings!! ')
    for key in list(conf.keys()):
        print(key, ': ', conf[key])
    print('-------------------------------------------------------------------')

    traj_list = []
    traj_counter = 0

    while True:
        t0 = time.time()
        # Generate new batch of data_files.
        feed_dict = {model.iter_num: np.float32(0),
                     model.train_cond: 1}

        try:
            [images, states, actions, gen_images, gen_states] = sess.run([model.images,
                                                                         model.states,
                                                                         model.actions,
                                                                         model.gen_images,
                                                                         model.gen_states],
                                                                        feed_dict)
        except tf.errors.OutOfRangeError:  # leave for loop after the 1st epoch
            break
        t1 = time.time()

        images = np.stack(images, axis=0)[1:]
        images = (images * 255).astype(np.uint8)
        states = np.stack(states, axis=0)[1:]
        actions = np.stack(actions, axis=0)[1:]
        gen_images= np.stack(gen_images, axis=0)
        gen_images = (gen_images*255).astype(np.uint8)
        gen_states = np.stack(gen_states, axis=0)

        for b in range(conf['batch_size']):

            traj = Trajectory()
            traj.images= images[:,b]
            traj.X_Xdot_full= states[:,b]
            traj.actions= actions[:,b]
            traj.gen_images= gen_images[:,b]
            traj.gen_states= gen_states[:,b]

            # save tfrecords
            traj = copy.deepcopy(traj)
            traj_list.append(traj)
            if 'traj_per_file' in conf:
                traj_per_file = conf['traj_per_file']
            else:
                traj_per_file = 256

            if len(traj_list) == traj_per_file:
                print('traj_per_file', traj_per_file)
                filename = 'traj_{0}_to_{1}' \
                    .format(traj_counter - traj_per_file + 1, traj_counter)

                from python_visual_mpc.visual_mpc_core.infrastructure.utility.save_tf_record import save_tf_record
                save_tf_record(filename, traj_list, {'data_save_dir': conf['data_dest_dir']})
                traj_list = []

            traj_counter += 1

        print('complete time', time.time() - t0)
        print('time to save:', time.time() - t1)


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
