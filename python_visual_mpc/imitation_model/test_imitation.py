import tensorflow as tf
from tensorflow.python.platform import flags
import os
import imp
from python_visual_mpc.imitation_model.imitation_model import SimpleModel
import numpy as np
from python_visual_mpc.video_prediction.read_tf_records2 import \
                    build_tfrecord_input as build_tfrecord
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('pretrained', '', 'pretrained model to evaluate')

def gen_mix_samples(N, means, std_dev, mix_params):
    dist_choice = np.random.choice(mix_params.shape[0], size=N, p=mix_params)
    samps = []
    for i in range(N):
        dist_mean = means[dist_choice[i]]
        out_dim = dist_mean.shape[0]
        dist_std = std_dev[dist_choice[i]]
        samp = np.random.multivariate_normal(dist_mean, dist_std * dist_std * np.eye(out_dim))

        samp_l = np.exp(-0.5 * np.sum(np.square(samp - means), axis=1) / np.square(dist_std))
        samp_l /=  np.power(2 * np.pi, out_dim / 2.) * dist_std
        samp_l *= mix_params

        samps.append((samp, np.sum(samp_l)))
    return sorted(samps, key = lambda x: -x[1])
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

    if conf['model'] is SimpleModel:
        gtruth_eep, pred_mean, pred_std, pred_mix = sess.run(
            [val_endeffector_pos, val_model.means, val_model.std_dev, val_model.mixing_parameters])

        print ''
        for i in range(64):
            samps = gen_mix_samples(200, pred_mean[i], pred_std[i], pred_mix[i])
            mean_samp = np.sum(pred_mean[i] * pred_mix[i].reshape((-1, 1)), axis=0)
            print 'batch', i
            print 'top samp', samps[0][0], 'with likelihood', samps[0][1]
            print 'mean_samp', mean_samp
            print 'true final', gtruth_eep[i, -1, :4]
            print 'error', np.sum(np.square(gtruth_eep[i, -1, :4] - samps[0][0]))
            print 'error_mean', np.sum(np.square(gtruth_eep[i, -1, :4] - mean_samp))
            print ''
    elif 'MDN_loss' in conf:
        v_images, gtruth_actions, gtruth_eep, pred_mean, pred_std, pred_mix = sess.run([val_images, val_actions,val_endeffector_pos, val_model.means, val_model.std_dev,
                                                                         val_model.mixing_parameters])

        print 'gtruth_actions', gtruth_actions.shape
        print 'pred_mean', pred_mean.shape
        print 'prev_var', pred_std.shape
        print 'pred_mix', pred_mix.shape

        import cv2
        for i in range(14):
            cv2.imshow('test', v_images[0, i])
            cv2.waitKey(-1)

        #print 'true final', gtruth_eep[0, -1, :]
        #print 'pred final', final[0]
        #print ''
        print 'start eep', gtruth_eep[0, 0,:6]
        for i in range(14):
            samps = gen_mix_samples(200, pred_mean[0, i], pred_std[0, i], pred_mix[0, i])
            print 'top samp', samps[0][0], 'with likelihood', samps[0][1]
            print 'gtruth', gtruth_eep[0, i + 1, :6]
            print ''
        # for j in range(2):
        #     print 'timestep', j
        #     print 'gtruth_step', gtruth_actions[0, j, :]
        #     for i in range(conf['MDN_loss']):
        #         print 'mean', i, 'has mix', pred_mix[0, j, i], 'and std', pred_std[0, j, i]
        #         print 'with vec', pred_mean[0, j, i, :]
        #     print ''
        # print 'final_pred', final[0, :]
        # print 'true final', gtruth_eep[0, -1, :6]


    else:
        val_images, gtruth_actions, gtruth_eep, pred_actions, pred_final = \
            sess.run([val_images, val_actions, val_endeffector_pos, val_model.predicted_actions,val_model.final_frame_state_pred])
        print 'val_images', val_images.shape
        import cv2
        for i in range(15):
            cv2.imshow('test', val_images[0, i])
            cv2.waitKey(-1 )
        print 'gtruth actions', gtruth_actions[0]
        print 'pred final', pred_final[0]
        print 'pred', pred_actions[0]
        print 'gtruth_eep', gtruth_eep[0, :, :6]
if __name__ == '__main__':
    main()