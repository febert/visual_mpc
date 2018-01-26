import tensorflow as tf
import imp
import numpy as np
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet
from PIL import Image
import os

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
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    g_predictor = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph= g_predictor)
    with sess.as_default():
        with g_predictor.as_default():
            # print 'predictor default session:', tf.get_default_session()
            # print 'predictor default graph:', tf.get_default_graph()
            print '-------------------------------------------------------------------'
            print 'Goal Distance Network Settings'
            for key in conf.keys():
                print key, ': ', conf[key]
            print '-------------------------------------------------------------------'

            print 'Constructing model for control'
            model = GoalDistanceNet(conf = conf,
                                     build_loss=False,
                                     load_data = False)

            sess.run(tf.global_variables_initializer())

            vars_without_state = filter_vars(tf.get_collection(tf.GraphKeys.VARIABLES))
            saver = tf.train.Saver(vars_without_state, max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])
            print 'gdn restore done.'

            def predictor_func(pred_images, goal_image):
                feed_dict = {
                            model.I0_pl:pred_images,
                            model.I1_pl:goal_image}

                warped_images, flow_field, warp_pts = sess.run([model.gen_image,
                                                              model.flow_field,
                                                              model.warp_pts],
                                                              feed_dict)
                return warped_images, flow_field, warp_pts

            return predictor_func

def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print 'removed state variable from saving-list: ', v.name

    return newlist