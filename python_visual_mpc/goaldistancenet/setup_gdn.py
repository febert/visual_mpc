import tensorflow as tf
import imp
import numpy as np
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet
from PIL import Image
from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher
import os
import pdb

def setup_gdn(conf, gpu_id = 0):
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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    g_predictor = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_predictor)
    with sess.as_default():
        with g_predictor.as_default():
            # print 'predictor default session:', tf.get_default_session()
            # print 'predictor default graph:', tf.get_default_graph()

            print('Constructing model Warping Network')
            model = GoalDistanceNet(conf = conf,
                                     build_loss=False,
                                     load_data = False)
            model.build_net()

            sess.run(tf.global_variables_initializer())

            if 'TEN_DATA' in os.environ:
                tenpath = conf['pretrained_model'].partition('tensorflow_data')[2]
                conf['pretrained_model'] = os.environ['TEN_DATA'] + tenpath

            print('-------------------------------------------------------------------')
            print('Goal Distance Network Settings')
            for key in list(conf.keys()):
                print(key, ': ', conf[key])
            print('-------------------------------------------------------------------')

            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars = variable_checkpoint_matcher(conf, vars, conf['pretrained_model'])
            saver = tf.train.Saver(vars, max_to_keep=0)

            # saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])
            print('gdn restore done.')

            def predictor_func(pred_images, goal_image):

                feed_dict = {
                            model.I0_pl:pred_images,
                            model.I1_pl:goal_image}

                warped_images, flow_field, warp_pts = sess.run([model.warped_I0_to_I1,
                                                                model.flow_bwd,
                                                                model.warp_pts_bwd],
                                                               feed_dict)
                return warped_images, flow_field, warp_pts

            return predictor_func