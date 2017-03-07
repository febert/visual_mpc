import tensorflow as tf
import imp
import numpy as np
from prediction_train_hidden_state import Model
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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
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



            print 'Constructing model for control'
            with tf.variable_scope('model', reuse=None) as training_scope:
                model = Model(conf, images, actions, states,
                              conf['sequence_length'])

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])

            def predictor_func(input_images, input_state, input_actions):
                """
                :param one_hot_images: the first two frames
                :param pixcoord: the coords of the disgnated pixel in images coord system
                :return: the predicted pixcoord at the end of sequence
                """

                itr = 0
                feed_dict = {model.prefix: 'ctrl',
                             model.iter_num: np.float32(itr),
                             model.lr: 0.0,
                             images: input_images,
                             actions: input_actions,
                             states: input_state,
                             }
                inf_low_state, gen_images, gen_states = sess.run([model.inf_low_state,
                                                               model.gen_images,
                                                               model.gen_states
                                                               ],
                                                                feed_dict)

                return inf_low_state, gen_images, gen_states

            return predictor_func