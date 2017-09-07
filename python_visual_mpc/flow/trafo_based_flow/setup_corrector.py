import tensorflow as tf
import imp
import numpy as np
from correction_train_flow import CorrectorModel
from PIL import Image
import os

def setup_corrector(conf_file):
    """
    Setup up the network for control
    :param conf_file:
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]

    hyperparams = imp.load_source('hyperparams', conf_file)
    conf = hyperparams.configuration

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    g_corrector = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_corrector)
    with sess.as_default():
        with g_corrector.as_default():

            # print 'corrector default session:', tf.get_default_session()
            # print 'corrector default graph:', tf.get_default_graph()

            print '-------------------------------------------------------------------'
            print 'verify current settings!! '
            for key in conf.keys():
                print key, ': ', conf[key]
            print '-------------------------------------------------------------------'

            tf.train.start_queue_runners(sess)

            input_distrib = tf.placeholder(tf.float32, shape=(conf['batch_size'], 64, 64, 1))

            images = tf.placeholder(tf.float32, name='images', shape=(conf['batch_size'], 2, 64, 64, 3))

            print 'Constructing Corrector...'
            with tf.variable_scope('model', reuse=None):
                model = CorrectorModel(conf, images, pix_distrib=input_distrib, train=False)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])


            def predictor_func(input_images, prev_distrib):
                """
                :param one_hot_images: the first two frames
                :param pixcoord: the coords of the disgnated pixel in images coord system
                :return: the predicted pixcoord at the end of sequence
                """


                feed_dict = {model.prefix: 'ctrl',
                             model.lr: 0,
                             images: input_images,  # could alternatively feed in gen_image
                             input_distrib: prev_distrib
                             }

                gen_image, gen_masks, output_distrib = sess.run([model.gen_images,
                                                                 model.gen_masks,
                                                                 model.gen_distrib
                                                                 ],
                                                                feed_dict)

                return gen_image, gen_masks, output_distrib

            return predictor_func