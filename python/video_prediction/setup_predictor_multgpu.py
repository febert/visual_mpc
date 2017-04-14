import tensorflow as tf
import imp
import numpy as np


from PIL import Image
import os

from datetime import datetime


class Tower(object):
    def __init__(self, conf, gpu_id, start_images, actions, start_states):
        nsmp_per_gpu = conf['batch_size']/ conf['ngpu']

        # picking different subset of the actions for each gpu
        act_startidx = gpu_id * nsmp_per_gpu
        per_gpu_actions = tf.slice(actions, [act_startidx, 0, 0], [nsmp_per_gpu, -1, -1])
        print 'startindex for gpu {0}: {1}'.format(gpu_id, act_startidx)

        Model = conf['model']
        self.model = Model(conf, input_data=[  start_images,
                                               start_states,
                                               per_gpu_actions,
                                               ])

def setup_predictor(conf, gpu_id=0, ngpu=1):
    """
    Setup up the network for control
    :param conf_file:
    :param ngpu number of gpus to use
    :return: function which predicts a batch of whole trajectories
    conditioned on the actions
    """


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

    start_images = tf.placeholder(tf.float32, name='images',  # with zeros extension
                                       shape=(1, conf['sequence_length'], 64, 64, 3))
    actions = tf.placeholder(tf.float32, name='actions',
                                  shape=(conf['batch_size'], conf['sequence_length'], 2))
    start_states = tf.placeholder(tf.float32, name='states',
                                       shape=(1, conf['context_frames'], 4))

    pix_distrib = tf.placeholder(tf.float32, shape=(1, conf['context_frames'], 64, 64, 1))


    #making the towers
    towers = []

    for i_gpu in xrange(ngpu):
        with tf.device('/gpu:%d' % i_gpu):
            with tf.name_scope('tower_%d' % (i_gpu)):
                print('creating tower %d: in scope %s' % (i_gpu, tf.get_variable_scope()))
                # print 'reuse: ', tf.get_variable_scope().reuse

                towers.append(Tower(conf, i_gpu, start_images, actions, start_states))
                tf.get_variable_scope().reuse_variables()

    comb_gen_img = []
    comb_pix_distrib = []
    comb_gen_states = []
    # import pdb;
    # pdb.set_trace()

    for t in range(conf['sequence_length']-1):
        t_comb_gen_img = [to.model.gen_images[t] for to in towers]
        comb_gen_img.append(tf.concat(0, t_comb_gen_img))

        t_comb_pix_distrib = [to.model.gen_distrib[t] for to in towers]
        comb_pix_distrib.append(tf.concat(0, t_comb_pix_distrib))

        t_comb_gen_states = [to.model.gen_states[t] for to in towers]
        comb_gen_states.append(tf.concat(0, t_comb_gen_states))

        # import pdb; pdb.set_trace()

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
    saver.restore(sess, conf['pretrained_model'])

    def predictor_func(input_images, input_one_hot_images, input_state, input_actions):
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
        feed_dict[pix_distrib] = input_one_hot_images

        gen_images, gen_distrib, gen_states = sess.run([comb_gen_img,
                                                      comb_gen_states],
                                                      feed_dict)

        print 'time for evaluating {0} actions on {1} gpus : {2}'.format(
            conf['batch_size'],
            conf['ngpu'],
            (datetime.now() - t_startiter).seconds + (datetime.now() - t_startiter).microseconds/1e6
            )
        return gen_distrib, gen_images, None, gen_states

    return predictor_func