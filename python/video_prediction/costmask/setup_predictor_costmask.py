import tensorflow as tf
import imp
import numpy as np
from video_prediction.costmask.prediction_train_costmask import Model
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

            images_pl = tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
            actions_pl = tf.placeholder(tf.float32, name= 'actions',
                                     shape=(conf['batch_size'], conf['sequence_length'], 2))
            states_pl = tf.placeholder(tf.float32, name='states',
                                         shape=(conf['batch_size'],conf['context_frames'] , 4))

            init_retpos_pl = tf.placeholder(tf.float32, name='init_retpos', shape=(3))
            init_retpos = tf.expand_dims(init_retpos_pl, 0)
            init_retpos = tf.tile(init_retpos, [conf['batch_size'],1])

            if 'no_pix_distrib' in conf:
                pix_distrib = None
            else:
                pix_distrib = tf.placeholder(tf.float32, shape=(conf['batch_size'], conf['context_frames'], 64, 64, 1))

            print 'Constructing model for control'
            with tf.variable_scope('model', reuse=None) as training_scope:
                model = Model(conf, images_pl, actions_pl, states_pl, init_obj_pose=init_retpos, reuse_scope=None, pix_distrib=pix_distrib)


            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])

            def predictor_func(input_images=None, one_hot_images=None,
                               input_state=None, input_actions=None, init_retpos= None):
                """
                :param one_hot_images: the first two frames
                :param pixcoord: the coords of the disgnated pixel in images coord system
                :return: the predicted pixcoord at the end of sequence
                """


                feed_dict = {model.prefix: 'ctrl',
                             model.iter_num: 50000, # this enables movement of the costmask!
                             model.lr: 0,
                             images_pl: input_images,
                             actions_pl: input_actions,
                             states_pl: input_state,
                             pix_distrib: one_hot_images,
                             init_retpos_pl : init_retpos}

                gen_distrib, gen_images, gen_masks, gen_states, gen_retina, retpos = sess.run([model.gen_distrib,
                                                                           model.gen_images,
                                                                           model.gen_masks,
                                                                           model.gen_states,
                                                                           model.pred_retinas,
                                                                           model.retpos_list
                                                                           ],
                                                                          feed_dict)
                return gen_distrib, gen_images, gen_masks, gen_states, gen_retina, retpos

            return predictor_func