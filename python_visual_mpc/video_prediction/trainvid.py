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
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

VIDEO_INTERVAL = 10000

# How often to save a model checkpoint
SAVE_INTERVAL = 4000

from python_visual_mpc.video_prediction.tracking_model.single_point_tracking_model import Single_Point_Tracking_Model
from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize_check', "", 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('resume', None, 'path to model file from which to resume training')
    flags.DEFINE_bool('diffmotions', False, 'visualize several different motions for a single scene')
    flags.DEFINE_bool('metric', False, 'compute metric of expected distance to human-labled positions ob objects')
    flags.DEFINE_bool('float16', False, 'whether to do inference with float16')
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

    conf['event_log_dir'] = conf['output_dir']
    if FLAGS.visualize_check:
        print('creating visualizations ...')
        conf['schedsamp_k'] = -1  # don't feed ground truth

        if 'test_data_dir' in conf:
            conf['data_dir'] = conf['test_data_dir']
        elif 'test_data_ind' in conf:
            conf['data_dir'] = '/'.join(str.split(conf['data_dir'][conf['test_data_ind']], '/')[:-1] + ['test'])
        else: conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])

        if FLAGS.visualize_check:
            conf['visualize_check'] = conf['output_dir'] + '/' + FLAGS.visualize_check
        conf['visualize'] = True

        conf['event_log_dir'] = '/tmp'
        conf.pop('use_len', None)

        conf.pop('color_augmentation', None)

        if FLAGS.metric:
            conf['batch_size'] = 128
            conf['sequence_length'] = 15
        else:
            conf['batch_size'] = 40

        conf['sequence_length'] = 14
        if FLAGS.diffmotions:
            conf['sequence_length'] = 14

        # when using alex interface:
        if 'modelconfiguration' in conf:
            conf['modelconfiguration']['schedule_sampling_k'] = conf['schedsamp_k']

        if FLAGS.float16:
            print('using float16')
            conf['float16'] = ''

        build_loss = False
    else:
        build_loss = True

    Model = conf['pred_model']

    if FLAGS.diffmotions or "visualize_tracking" in conf or FLAGS.metric:
        model = Model(conf, load_data=False, trafo_pix=True, build_loss=build_loss)
    else:
        model = Model(conf, load_data=True, trafo_pix=False, build_loss=build_loss)

    print('Constructing saver.')
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars = filter_vars(vars)
    saving_saver = tf.train.Saver(vars, max_to_keep=0)


    if FLAGS.resume:
        vars = variable_checkpoint_matcher(conf, vars, FLAGS.resume)
        loading_saver = tf.train.Saver(vars, max_to_keep=0)

    if FLAGS.visualize_check:
        vars = variable_checkpoint_matcher(conf, vars, conf['visualize_check'])
        loading_saver = tf.train.Saver(vars, max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(conf['event_log_dir'], graph=sess.graph, flush_secs=10)

    if not FLAGS.diffmotions:
        tf.train.start_queue_runners(sess)

    sess.run(tf.global_variables_initializer())

    if conf['visualize']:
        if FLAGS.visualize_check:
            load_checkpoint(conf, sess, loading_saver, conf['visualize_check'])
        else: load_checkpoint(conf, sess, loading_saver)

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

    itr_0 =0
    if FLAGS.resume != None:
        itr_0 = load_checkpoint(conf, sess, loading_saver, model_file=FLAGS.resume)
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

        if len(conf['data_dir']) == 2:
            if 'scheduled_finetuning' in conf:
                dest_itr = conf['scheduled_finetuning_dest_itr']
                ratio_dest_val = conf['scheduled_finetuning_dest_value']
                ratio01 = np.array([(itr/dest_itr)*ratio_dest_val + (1.-itr/dest_itr)])
                ratio01 = np.clip(ratio01, ratio_dest_val, 1.)
                ratio01 = np.squeeze(ratio01)
                feed_dict[model.dataset_01ratio] = ratio01
            else:
                ratio01 = 0.2
                feed_dict[model.dataset_01ratio] = ratio01

            if (itr) % 10 == 0:
                print('ratio old data/batchsize:', ratio01)


        cost, _, summary_str = sess.run([model.loss, model.train_op, model.train_summ_op],
                                        feed_dict)

        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {model.iter_num: np.float32(itr),
                         model.train_cond: 0}
            if len(conf['data_dir']) == 2:
                feed_dict[model.dataset_01ratio] = ratio01
                [val_summary_str, val_0_summary_str, val_1_summary_str] = sess.run([model.val_summ_op, model.val_0_summ_op, model.val_1_summ_op], feed_dict)
                summary_writer.add_summary(val_summary_str, itr)
                summary_writer.add_summary(val_0_summary_str, itr)
                summary_writer.add_summary(val_1_summary_str, itr)
            else:
                [val_summary_str] = sess.run([model.val_summ_op], feed_dict)
                summary_writer.add_summary(val_summary_str, itr)

        if (itr) % VIDEO_INTERVAL == 2 and hasattr(model, 'val_video_summaries'):
            feed_dict = {model.iter_num: np.float32(itr),
                         model.train_cond: 0}
            video_proto = sess.run(model.val_video_summaries, feed_dict = feed_dict)
            summary_writer.add_summary(convert_tensor_to_gif_summary(video_proto))

        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model to' + conf['output_dir'])
            saving_saver.save(sess, conf['output_dir'] + '/model' + str(itr))

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

        if (itr) % SUMMARY_INTERVAL == 2:
            summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saving_saver.save(sess, conf['output_dir'] + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush()

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
