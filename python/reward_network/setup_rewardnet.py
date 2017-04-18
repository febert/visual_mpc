import tensorflow as tf
import imp
import numpy as np
from train_rewardnet import Model
from PIL import Image
import os

def setup_rewardnet(conf, gpu_id = 0):
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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
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

            current_images_pl = tf.placeholder(tf.float32, name='images',
                                    shape=(conf['batch_size'], 64, 64, 3))

            goal_image_pl = tf.placeholder(tf.float32, name='images',
                                               shape=(64, 64, 3))



            print 'Constructing model for control'
            with tf.variable_scope('trainmodel', reuse=None) as training_scope:
                model = Model(conf, currentimages=current_images_pl, goalimage=goal_image_pl)


            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])

            def predictor_func(current_images, goal_image):
                """
                :param one_hot_images: the first two frames
                :param pixcoord: the coords of the disgnated pixel in images coord system
                :return: the predicted pixcoord at the end of sequence
                """


                feed_dict = {
                             current_images_pl: current_images,
                             goal_image_pl: goal_image
                             }

                [softmax_out] = sess.run([model.softmax_output], feed_dict)
                return softmax_out

            return predictor_func