import tensorflow as tf
import imp
import numpy as np
import pdb

from PIL import Image
import os

from datetime import datetime

class Tower(object):
    def __init__(self, conf, gpu_id, start_images, actions, start_states, pix_distrib):
        nsmp_per_gpu = conf['batch_size']/ conf['ngpu']


        # picking different subset of the actions for each gpu
        startidx = gpu_id * nsmp_per_gpu
        per_gpu_actions = tf.slice(actions, [startidx, 0, 0], [nsmp_per_gpu, -1, -1])
        start_images = tf.slice(start_images, [startidx, 0, 0, 0, 0], [nsmp_per_gpu, -1, -1, -1, -1])
        start_states = tf.slice(start_states, [startidx, 0, 0], [nsmp_per_gpu, -1, -1])
        if 'no_pix_distrib' in conf:
         pix_distrib = None
        else:
            pix_distrib = tf.slice(pix_distrib, [startidx, 0, 0, 0, 0], [nsmp_per_gpu, -1, -1, -1, -1])

        print 'startindex for gpu {0}: {1}'.format(gpu_id, startidx)

        if 'prediction_model' in conf:
            Model = conf['prediction_model']
        else:
            from accum_tf_factorized_prediction_train_sawyer import Model

        with tf.variable_scope('', reuse=None):
            self.model = Model(conf,start_images, per_gpu_actions, start_states, pix_distrib=pix_distrib, inference=True)

def setup_predictor(conf, gpu_id=0, ngpu=1):
    """
    Setup up the network for control
    :param conf_file:
    :param ngpu number of gpus to use
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """

    if 'prediction_model' in conf:
        Model = conf['prediction_model']
    else:
        from accum_tf_factorized_prediction_train_sawyer import Model

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
    print 'verify network settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'Constructing multi gpu model for control...'

    if 'single_view' in conf:
        numcam = 1
    else:
        numcam = 2
    start_images = tf.placeholder(tf.float32, name='images',  # with zeros extension
                                    shape=(conf['batch_size'], conf['sequence_length'], 64, 64, 3*numcam))
    actions = tf.placeholder(tf.float32, name='actions',
                                    shape=(conf['batch_size'],conf['sequence_length'], 4))
    start_states = tf.placeholder(tf.float32, name='states',
                                    shape=(conf['batch_size'],conf['context_frames'], 3))
    pix_distrib = tf.placeholder(tf.float32, shape=(conf['batch_size'], conf['context_frames'], 64, 64, numcam))

    # creating dummy network to avoid issues with naming of variables
    # with tf.variable_scope('', reuse=None):
    #     model = Model(conf, start_images, actions, start_states, pix_distrib=pix_distrib, inference=True)

    # sess.run(tf.initialize_all_variables())
    # sess.run(tf.global_variables_initializer())

    # saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    # saver.restore(sess, conf['pretrained_model'])

    # checkpoint = tf.train.latest_checkpoint(conf['pretrained_model'])
    # saver.restore(sess, checkpoint)
    # print 'restore done. '

    #making the towers
    towers = []

    for i_gpu in range(ngpu):
        with tf.device('/gpu:%d' % i_gpu):
            with tf.name_scope('tower_%d' % (i_gpu)):
                print('creating tower %d: in scope %s' % (i_gpu, tf.get_variable_scope()))
                # print 'reuse: ', tf.get_variable_scope().reuse

                with tf.variable_scope('', reuse=i_gpu > 0):
                    towers.append(Tower(conf, i_gpu, start_images, actions, start_states, pix_distrib))
                # tf.get_variable_scope().reuse_variables()

    comb_gen_img = []
    comb_pix_distrib = []
    comb_gen_states = []

    for t in range(conf['sequence_length']-1):
        t_comb_gen_img = [to.model.gen_images[t] for to in towers]
        comb_gen_img.append(tf.concat(axis=0, values=t_comb_gen_img))

        if not 'no_pix_distrib' in conf:
            t_comb_pix_distrib = [to.model.gen_distrib[t] for to in towers]
            comb_pix_distrib.append(tf.concat(axis=0, values=t_comb_pix_distrib))

        t_comb_gen_states = [to.model.gen_states[t] for to in towers]
        comb_gen_states.append(tf.concat(axis=0, values=t_comb_gen_states))

    sess.run(tf.global_variables_initializer())
    restore_vars = tf.get_default_graph().get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver = tf.train.Saver(restore_vars, max_to_keep=0)
    checkpoint = tf.train.latest_checkpoint(conf['pretrained_model'])
    saver.restore(sess, checkpoint)
    print 'restore done. '

    def predictor_func(input_images=None, input_one_hot_images=None, input_state=None, input_actions=None):
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

        feed_dict[start_images] = input_images
        feed_dict[start_states] = input_state
        feed_dict[actions] = input_actions

        gen_distrib = None
        if 'no_pix_distrib' in conf:
            gen_images, gen_states = sess.run([comb_gen_img,
                                              comb_gen_states],
                                              feed_dict)

            gen_distrib = None
        else:
            feed_dict[pix_distrib] = input_one_hot_images
            gen_images, gen_distrib, gen_states = sess.run([comb_gen_img,
                                                            comb_pix_distrib,
                                                            comb_gen_states],
                                                           feed_dict)

        print 'time for evaluating {0} actions on {1} gpus : {2}'.format(
            conf['batch_size'],
            conf['ngpu'],
            (datetime.now() - t_startiter).seconds + (datetime.now() - t_startiter).microseconds/1e6)
        return gen_images, gen_distrib, gen_states

    return predictor_func