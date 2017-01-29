import tensorflow as tf
import imp
import numpy as np
from prediction_train import Model
from PIL import Image
import os

def setup_predictor(conf, gpu_id = 0):
    """
    Setup up the network for control
    :param conf_file:
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """
    if gpu_id == None:
        gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    g_predictor = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_predictor)
    with sess.as_default():
        with g_predictor.as_default():

            # print 'predictor default session:', tf.get_default_session()
            # print 'predictor default graph:', tf.get_default_graph()

            print '-------------------------------------------------------------------'
            print 'verify current settings!! '
            for key in conf.keys():
                print key, ': ', conf[key]
            print '-------------------------------------------------------------------'

            images = tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
            actions = tf.placeholder(tf.float32, name= 'actions',
                                     shape=(conf['batch_size'], conf['sequence_length'], 2))
            states = tf.placeholder(tf.float32, name='states',
                                         shape=(conf['batch_size'],conf['context_frames'] , 4))

            pix_distrib = tf.placeholder(tf.float32, shape=(conf['batch_size'], conf['context_frames'], 64, 64, 1))

            print 'Constructing model for control'
            with tf.variable_scope('model', reuse=None) as training_scope:
                model = Model(conf, images, actions, states,
                              conf['sequence_length'], reuse_scope= None, pix_distrib= pix_distrib)


            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])

            def predictor_func(input_images, one_hot_images, input_state, input_actions):
                """
                :param one_hot_images: the first two frames
                :param pixcoord: the coords of the disgnated pixel in images coord system
                :return: the predicted pixcoord at the end of sequence
                """

                itr = 0
                feed_dict = {model.prefix: 'ctrl',
                             model.iter_num: np.float32(itr),
                             model.lr: conf['learning_rate'],
                             images: input_images,
                             actions: input_actions,
                             states: input_state,
                             pix_distrib: one_hot_images
                             }
                gen_distrib, gen_images, gen_masks, gen_states = sess.run([model.gen_distrib,
                                                               model.gen_images,
                                                               model.gen_masks,
                                                               model.gen_states
                                                               ],
                                                                feed_dict)

                # summary_writer = tf.train.SummaryWriter(conf['current_dir'], flush_secs=1)
                # summary_writer.add_summary(summary_str)

                return gen_distrib, gen_images, gen_masks, gen_states

            return predictor_func