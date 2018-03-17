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

        model = ImitationBaseModel(conf, train_images, train_actions, train_endeffector_pos)
        model.build()

    with tf.variable_scope('val_model', reuse = None):
        data_dict = build_tfrecord(conf, training=False)
        # validation input images
        val_images = data_dict['images']
        # validation ground truth actions/endef
        val_actions = data_dict['actions']
        val_endeffector_pos = data_dict['endeffector_pos']

        with tf.variable_scope(training_scope, reuse=True):
            val_model = ImitationBaseModel(conf, val_images, val_actions, val_endeffector_pos)
            val_model.build()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.Session(config= tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, conf['pretrained'])

    gtruth_actions, pred_mean, pred_var, pred_mix = sess.run([val_actions, val_model.means, val_model.variance, val_model.mixing_parameters])

    print 'gtruth_actions', gtruth_actions.shape
    print 'pred_mean', pred_mean.shape
    print 'prev_var', pred_var.shape
    print 'pred_mix', pred_mix.shape

    test_sequence = gtruth_actions[0, 0, :]
    seq_means = pred_mean[0]
    seq_var = pred_var[0]
    seq_mix = pred_mix[0]

    print 'test seq', test_sequence
    print 'mean 1', seq_means[0]
    print 'mean 2', seq_means[1]
    print 'mix', seq_mix
    print 'var', seq_var

    mix_1 = np.random.multivariate_normal(seq_means[0], np.diag(np.ones(conf['adim'])) * seq_var[0], size=100)
    mix_2 = np.random.multivariate_normal(seq_means[1], np.diag(np.ones(conf['adim'])) * seq_var[1], size=100)
    final_mixs = seq_mix[0] * mix_1 + seq_mix[1] + mix_2

    diffs = final_mixs - test_sequence
    for i in range(100):
     print 'final_mixs', i, 'is', final_mixs[i]
    print np.sum(np.power(diffs, 2), axis = 1)
if __name__ == '__main__':
    main()