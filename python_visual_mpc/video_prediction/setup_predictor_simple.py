import tensorflow as tf
import imp
import numpy as np
from python_visual_mpc.video_prediction.prediction_train_sawyer import Model
from PIL import Image
import os

def setup_predictor(conf, gpu_id = 0, ngpu=None):
    """
    Setup up the network for control
    :param conf_file:
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """
    if gpu_id == None:
        gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    g_predictor = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_predictor)
    with sess.as_default():
        with g_predictor.as_default():

            # print 'predictor default session:', tf.get_default_session()
            # print 'predictor default graph:', tf.get_default_graph()

            print('-------------------------------------------------------------------')
            print('verify current settings!! ')
            for key in list(conf.keys()):
                print(key, ': ', conf[key])
            print('-------------------------------------------------------------------')

            images_pl = tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
            if 'sdim' in conf:
                sdim = conf['sdim']
            else:
                if 'sawyer' in conf:
                    sdim = 3
                else: sdim = 4
            if 'adim' in conf:
                adim = conf['adim']
            else:
                if 'sawyer' in conf:
                    adim = 4
                else: adim = 2
            print('adim', adim)
            print('sdim', sdim)

            actions_pl = tf.placeholder(tf.float32, name= 'actions',
                                     shape=(conf['batch_size'], conf['sequence_length'], adim))
            states_pl = tf.placeholder(tf.float32, name='states',
                                         shape=(conf['batch_size'],conf['context_frames'] , sdim))

            if 'no_pix_distrib' in conf:
                pix_distrib = None
            else:
                pix_distrib = tf.placeholder(tf.float32, shape=(conf['batch_size'], conf['context_frames'], 64, 64, 1))

            print('Constructing model for control')
            with tf.variable_scope('model', reuse=None) as training_scope:
                model = Model(conf, images_pl, actions_pl, states_pl,reuse_scope= None, pix_distrib= pix_distrib)

            sess.run(tf.global_variables_initializer())

            vars_without_state = filter_vars(tf.get_collection(tf.GraphKeys.VARIABLES))
            saver = tf.train.Saver(vars_without_state, max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])

            def predictor_func(input_images=None, input_one_hot_images1=None,
                               input_state=None, input_actions=None):
                """
                :param one_hot_images: the first two frames
                :return: the predicted pixcoord at the end of sequence
                """
                itr = 0
                feed_dict = {
                             model.iter_num: np.float32(itr),
                             model.lr: conf['learning_rate'],
                             images_pl: input_images,
                             actions_pl: input_actions,
                             states_pl: input_state,
                             pix_distrib: input_one_hot_images1
                             }


                gen_distrib, gen_images, gen_masks, gen_states = sess.run([model.m.gen_distrib1,
                                                                           model.m.gen_images,
                                                                           model.m.gen_masks,
                                                                           model.m.gen_states
                                                                           ],
                                                                          feed_dict)
                return gen_images, gen_distrib, None, gen_states, gen_masks


            return predictor_func

def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print('removed state variable from saving-list: ', v.name)
    return newlist