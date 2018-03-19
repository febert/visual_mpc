import tensorflow as tf
from tensorflow.python.platform import flags
import os
import imp
from python_visual_mpc.imitation_model.imitation_model import ImitationBaseModel
import numpy as np
from python_visual_mpc.video_prediction.read_tf_records2 import \
                    build_tfrecord_input as build_tfrecord
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('pretrained', '', 'pretrained model to evaluate')

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    if not os.path.exists(FLAGS.hyper):
        raise RuntimeError("Experiment configuration not found")

    hyperparams = imp.load_source('hyperparams', FLAGS.hyper)
    conf = hyperparams.configuration
    conf['visualize'] = False
    conf['pretrained'] = conf['model_dir'] + FLAGS.pretrained

    with tf.variable_scope('model', reuse = None) as training_scope:
        data_dict = build_tfrecord(conf, training=True)
        # training input images
        train_images = data_dict['images']
        # training ground truth actions/endef
        train_actions = data_dict['actions']
        train_endeffector_pos = data_dict['endeffector_pos']

        model = conf['model'](conf, train_images, train_actions, train_endeffector_pos)
        model.build()

    with tf.variable_scope('val_model', reuse = None):
        data_dict = build_tfrecord(conf, training=False)
        # validation input images
        val_images = data_dict['images']
        # validation ground truth actions/endef
        val_actions = data_dict['actions']
        val_endeffector_pos = data_dict['endeffector_pos']

        with tf.variable_scope(training_scope, reuse=True):
            val_model = conf['model'](conf, val_images, val_actions, val_endeffector_pos)
            val_model.build()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.Session(config= tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, conf['pretrained'])

    if 'MDN_loss' in conf:
        gtruth_actions, pred_mean, pred_std, pred_mix = sess.run([val_actions, val_model.means, val_model.std_dev, val_model.mixing_parameters])

        print 'gtruth_actions', gtruth_actions.shape
        print 'pred_mean', pred_mean.shape
        print 'prev_var', pred_std.shape
        print 'pred_mix', pred_mix.shape

        test_sequence = gtruth_actions[0, 0, :]
        seq_means = pred_mean[0]
        seq_std = pred_std[0]
        seq_mix = pred_mix[0]

        print 'test seq', test_sequence
        print 'mean 1', seq_means[:, 0, :]
        print 'mean 2', seq_means[:, 1, :]
        print 'mean 3', seq_means[:, 2, :]
        print 'mix', seq_mix
        print 'std dev', seq_std
    else:
        val_images, gtruth_actions, gtruth_eep, pred_actions = \
            sess.run([val_images, val_actions, val_endeffector_pos, val_model.predicted_actions])
        print 'val_images', val_images.shape
        import cv2
        for i in range(15):
            cv2.imshow('test', val_images[0, i])
            cv2.waitKey(-1 )
        print 'loss', np.sqrt(np.power(gtruth_actions - pred_actions.reshape((conf['batch_size'], -1, conf['adim'])), 2))
        print 'gtruth actions', gtruth_actions[0]
        print 'pred', pred_actions[0]
        print 'gtruth_eep', gtruth_eep[0, :, :6]
if __name__ == '__main__':
    main()