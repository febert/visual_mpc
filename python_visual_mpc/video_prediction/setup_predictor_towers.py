import tensorflow as tf
import imp
import numpy as np
import pdb
import copy

from PIL import Image
import os

from datetime import datetime

class Tower(object):
    def __init__(self, conf, gpu_id, start_images, actions, start_states, pix_distrib1,pix_distrib2):
        nsmp_per_gpu = conf['batch_size']/ conf['ngpu']
        # setting the per gpu batch_size

        # picking different subset of the actions for each gpu
        startidx = gpu_id * nsmp_per_gpu
        actions = tf.slice(actions, [startidx, 0, 0], [nsmp_per_gpu, -1, -1])
        start_images = tf.slice(start_images, [startidx, 0, 0, 0, 0], [nsmp_per_gpu, -1, -1, -1, -1])
        start_states = tf.slice(start_states, [startidx, 0, 0], [nsmp_per_gpu, -1, -1])

        pix_distrib1 = tf.slice(pix_distrib1, [startidx, 0, 0, 0, 0], [nsmp_per_gpu, -1, -1, -1, -1])
        pix_distrib2 = tf.slice(pix_distrib2, [startidx, 0, 0, 0, 0], [nsmp_per_gpu, -1, -1, -1, -1])

        print 'startindex for gpu {0}: {1}'.format(gpu_id, startidx)

        if 'pred_model' in conf:
            Model = conf['pred_model']
            print 'using pred_model', Model
        else:
            from prediction_train_sawyer import Model

        # this is to keep compatiblity with old model implementations (without basecls structure)
        if hasattr(Model,'m'):
            for name, value in Model.m.__dict__.iteritems():
                setattr(Model, name, value)

        modconf = copy.deepcopy(conf)
        modconf['batch_size'] = nsmp_per_gpu
        if 'ndesig' in conf:
            self.model = Model(modconf, start_images, actions, start_states, pix_distrib=pix_distrib1,pix_distrib2=pix_distrib2, inference=True)
        else:
            self.model = Model(modconf, start_images, actions, start_states, pix_distrib=pix_distrib1, inference=True)

def setup_predictor(conf, gpu_id=0, ngpu=1):
    """
    Setup up the network for control
    :param conf_file:
    :param ngpu number of gpus to use
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """

    from prediction_train_sawyer import Model

    conf['ngpu'] = ngpu

    start_id = gpu_id
    indexlist = [str(i_gpu) for i_gpu in range(start_id, start_id + ngpu)]
    var = ','.join(indexlist)
    print 'using CUDA_VISIBLE_DEVICES=', var
    os.environ["CUDA_VISIBLE_DEVICES"] = var
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,
                                                       allow_soft_placement=True,
                                                       log_device_placement=False))

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'Constructing multi gpu model for control...'

    images_pl = tf.placeholder(tf.float32, name='images',
                               shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3))
    if 'sdim' in conf:
        sdim = conf['sdim']
    else:
        if 'sawyer' in conf:
            sdim = 3
        else:
            sdim = 4
    if 'adim' in conf:
        adim = conf['adim']
    else:
        if 'sawyer' in conf:
            adim = 4
        else:
            adim = 2
    print 'adim', adim
    print 'sdim', sdim

    actions_pl = tf.placeholder(tf.float32, name='actions',
                                shape=(conf['batch_size'], conf['sequence_length'], adim))
    states_pl = tf.placeholder(tf.float32, name='states',
                               shape=(conf['batch_size'], conf['context_frames'], sdim))

    pix_distrib_1 = tf.placeholder(tf.float32, shape=(conf['batch_size'], conf['context_frames'], 64, 64, 1))
    pix_distrib_2 = tf.placeholder(tf.float32, shape=(conf['batch_size'], conf['context_frames'], 64, 64, 1))

    # making the towers
    towers = []
    with tf.variable_scope('model', reuse=None):
        for i_gpu in xrange(ngpu):
            with tf.device('/gpu:%d' % i_gpu):
                with tf.name_scope('tower_%d' % (i_gpu)):
                    print('creating tower %d: in scope %s' % (i_gpu, tf.get_variable_scope()))
                    # print 'reuse: ', tf.get_variable_scope().reuse
                    # towers.append(Tower(conf, i_gpu, training_scope, start_images, actions, start_states, pix_distrib_1, pix_distrib_2))
                    towers.append(Tower(conf, i_gpu, images_pl, actions_pl, states_pl, pix_distrib_1,
                                                pix_distrib_2))
                    tf.get_variable_scope().reuse_variables()

    sess.run(tf.global_variables_initializer())

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #cutting off the /model tag
    vars_no_state = dict([('/'.join(var.name.split(':')[0].split('/')[1:]), var) for var in vars])
    saver = tf.train.Saver(vars_no_state, max_to_keep=0)
    saver.restore(sess, conf['pretrained_model'])

    print 'restore done. '

    comb_gen_img = []
    comb_pix_distrib1 = []
    comb_pix_distrib2 = []
    comb_gen_states = []

    for t in range(conf['sequence_length']-1):
        t_comb_gen_img = [to.model.gen_images[t] for to in towers]
        comb_gen_img.append(tf.concat(axis=0, values=t_comb_gen_img))

        if not 'no_pix_distrib' in conf:
            t_comb_pix_distrib1 = [to.model.gen_distrib1[t] for to in towers]
            comb_pix_distrib1.append(tf.concat(axis=0, values=t_comb_pix_distrib1))
            if 'ndesig' in conf:
                t_comb_pix_distrib2 = [to.model.gen_distrib2[t] for to in towers]
                comb_pix_distrib2.append(tf.concat(axis=0, values=t_comb_pix_distrib2))

        t_comb_gen_states = [to.model.gen_states[t] for to in towers]
        comb_gen_states.append(tf.concat(axis=0, values=t_comb_gen_states))


    def predictor_func(input_images=None, input_one_hot_images1=None, input_one_hot_images2=None, input_state=None, input_actions=None):
        """
        :param one_hot_images: the first two frames
        :param pixcoord: the coords of the disgnated pixel in images coord system
        :return: the predicted pixcoord at the end of sequence
        """

        t_startiter = datetime.now()

        feed_dict = {}
        for t in towers:
            feed_dict[t.model.iter_num] = 0
            feed_dict[t.model.lr] = 0.0

        feed_dict[images_pl] = input_images
        feed_dict[states_pl] = input_state
        feed_dict[actions_pl] = input_actions

        if 'no_pix_distrib' in conf:
            gen_images, gen_states = sess.run([comb_gen_img,
                                              comb_gen_states],
                                              feed_dict)

            gen_distrib1 = None
            gen_distrib2 = None
        else:
            feed_dict[pix_distrib_1] = input_one_hot_images1
            if 'ndesig' in conf:
                print 'evaluating 2 pixdistrib..'
                feed_dict[pix_distrib_2] = input_one_hot_images2

                gen_images, gen_distrib1, gen_distrib2, gen_states = sess.run([comb_gen_img,
                                                                comb_pix_distrib1,
                                                                comb_pix_distrib2,
                                                                comb_gen_states],
                                                               feed_dict)
            else:
                gen_distrib2 = None
                gen_images, gen_distrib1, gen_states = sess.run([comb_gen_img,
                                                                comb_pix_distrib1,
                                                                comb_gen_states],
                                                               feed_dict)

        print 'time for evaluating {0} actions on {1} gpus : {2}'.format(
            conf['batch_size'],
            conf['ngpu'],
            (datetime.now() - t_startiter).seconds + (datetime.now() - t_startiter).microseconds/1e6)

        return gen_images, gen_distrib1, gen_distrib2, gen_states, None

    return predictor_func
