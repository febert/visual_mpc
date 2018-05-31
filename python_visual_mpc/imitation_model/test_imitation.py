import tensorflow as tf
from tensorflow.python.platform import flags
import os
import imp
from python_visual_mpc.imitation_model.imitation_model import gen_mix_samples
from python_visual_mpc.imitation_model.imitation_model import SimpleModel
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
    print('using CUDA_VISIBLE_DEVICES=', FLAGS.device)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

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
        goal_image = data_dict.get('goal_image', None)
        model = conf['model'](conf, train_images, train_actions, train_endeffector_pos, goal_image)
        model.build()

    with tf.variable_scope('val_model', reuse = None):
        data_dict = build_tfrecord(conf, training=False)
        # validation input images
        val_images = data_dict['images']
        # validation ground truth actions/endef
        val_actions = data_dict['actions']
        val_endeffector_pos = data_dict['endeffector_pos']
        val_goal_image = data_dict.get('goal_image', None)
        with tf.variable_scope(training_scope, reuse=True):
            val_model = conf['model'](conf, val_images, val_actions, val_endeffector_pos, val_goal_image)
            val_model.build(is_Train = False)

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

        print('')
        for i in range(64):
            samps = gen_mix_samples(1000, pred_mean[i], pred_std[i], pred_mix[i])
            mean_samp = np.sum(pred_mean[i] * pred_mix[i].reshape((-1, 1)), axis=0)
            print('batch', i)
            print('top samp', samps[0][0], 'with likelihood', samps[0][1])
            print('mean_samp', mean_samp)
            print('true final', gtruth_eep[i, -1, :4])
            print('error', np.sum(np.square(gtruth_eep[i, -1, :4] - samps[0][0])))
            print('error_mean', np.sum(np.square(gtruth_eep[i, -1, :4] - mean_samp)))
            print('')
    elif 'MDN_loss' in conf:
        v_images, v_gimage, gtruth_actions, gtruth_eep, pred_mean, pred_std, pred_mix = sess.run([val_images, val_goal_image, val_actions,val_endeffector_pos, val_model.means, val_model.std_dev,
                                                                         val_model.mixing_parameters])
        print('gtruth_actions', gtruth_actions.shape)
        print('gtruth_eep', gtruth_eep.shape)
        print('pred_mean', pred_mean.shape)
        print('prev_var', pred_std.shape)
        print('pred_mix', pred_mix.shape)
        

        import cv2
        for i in range(14):
            cv2.imwrite('test/im_{}.png'.format(i), (v_images[0, i] * 256).astype(np.uint8)[:,:,::-1])
        
        cv2.imwrite('test/goal_image.png', (v_gimage[0] * 255).astype(np.uint8)[:,:,::-1])
        
        for i in range(14):
            samps, samps_log_l = gen_mix_samples(1000, pred_mean[0, i], pred_std[0, i], pred_mix[0, i])
            print('top samp', samps[0], 'with log likelihood', samps_log_l[0])
            print('gtruth', gtruth_eep[0, i + 1, :6])
            print('')
        # for j in range(2):
        #     print 'timestep', j
        #     print 'gtruth_step', gtruth_actions[0, j, :]
        #     for i in range(conf['MDN_loss']):
        #         print 'mean', i, 'has mix', pred_mix[0, j, i], 'and std', pred_std[0, j, i]
        #         print 'with vec', pred_mean[0, j, i, :]
        #     print ''
        # print 'final_pred', final[0, :]
        # print 'true final', gtruth_eep[0, -1, :6]

    elif 'latent_dim' in conf:
        val_images, gtruth_actions, gtruth_eep, pred_actions, rel_states = \
            sess.run([val_images, val_actions, val_endeffector_pos, val_model.predicted_actions, val_model.predicted_rel_states])
        print('val_images', val_images.shape)
       # print val_images
       # import cv2
       # for i in range(15):
       #     cv2.imshow('test', val_images[0, i, :, :, ::-1])
       #     cv2.waitKey(-1)
        print('start eep', gtruth_eep[0, 0, :6])

        for i in range(14):
            print('pred action', pred_actions[0, i])
            print('gtruth_action', gtruth_actions[0, i, :6])
           # print 'gtruth_target', gtruth_eep[0, i + 1, :6]
           # print 'pred_target', rel_states[0, i]
            print('')
    else:
        val_images, gtruth_actions, gtruth_eep, pred_actions, pred_final = \
            sess.run([val_images, val_actions, val_endeffector_pos, val_model.predicted_actions,val_model.final_frame_state_pred])
        print('val_images', val_images.shape)
        import cv2
        for i in range(15):
            cv2.imshow('test', val_images[0, i])
            cv2.waitKey(-1 )
        print('gtruth actions', gtruth_actions[0])
        print('pred final', pred_final[0])
        print('pred', pred_actions[0])
        print('gtruth_eep', gtruth_eep[0, :, :6])
if __name__ == '__main__':
    main()
