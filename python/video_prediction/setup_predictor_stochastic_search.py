import tensorflow as tf
import imp
import numpy as np
from train_stochastic_search_multgpu import Model

from PIL import Image
import os



class Tower(object):
    def __init__(self, conf, gpu_id, reuse_scope = None):

        self.start_images = tf.placeholder(tf.float32, name='images',
                                shape=(1, conf['sequence_length'], 64, 64, 3))
        self.actions = tf.placeholder(tf.float32, name='actions',
                                 shape=(conf['batch_size'], conf['sequence_length'], 2))
        self.start_states = tf.placeholder(tf.float32, name='states',
                                shape=(1, conf['context_frames'], 4))

        self.pix_distrib = tf.placeholder(tf.float32, shape=(1, conf['context_frames'], 64, 64, 1))

        nsmp_per_gpu = conf['batch_size']/ conf['ngpu']

        start_images = tf.tile(self.start_images, [nsmp_per_gpu, 1 , 1, 1, 1])
        pix_distrib = tf.tile(self.pix_distrib, [nsmp_per_gpu, 1, 1, 1, 1])
        start_states = tf.tile(self.start_states, [nsmp_per_gpu, 1 , 1])

        # picking different subset of the actions for each gpu
        act_startidx = gpu_id * nsmp_per_gpu
        per_gpu_actions = tf.slice(self.actions, [act_startidx, 0, 0], [nsmp_per_gpu, -1, -1])

        per_gpu_noise = tf.truncated_normal([nsmp_per_gpu, conf['sequence_length'], conf['noise_dim']],
                                         mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

        self.model = Model(conf, reuse_scope =reuse_scope, input_data=[start_images,
                                                                       start_states,
                                                                       per_gpu_actions,
                                                                       per_gpu_noise,
                                                                       pix_distrib])

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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    g_predictor = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True), graph= g_predictor)
    with sess.as_default():
        with g_predictor.as_default():

            print '-------------------------------------------------------------------'
            print 'verify current settings!! '
            for key in conf.keys():
                print key, ': ', conf[key]
            print '-------------------------------------------------------------------'

            print 'Constructing multi gpu model for control...'


            # create dummy model to get the names of the variable right...
            with tf.variable_scope('train_model', reuse=None) as training_scope:
                model = Model(conf)


            #making the towers
            towers = []

            for i in xrange(conf['ngpu']):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % (i)):
                        print('creating tower %d: in scope %s' % (i, tf.get_variable_scope()))

                        towers.append(Tower(conf, i, training_scope))
                        tf.get_variable_scope().reuse_variables()

            comb_gen_img = [t.model.gen_images for t in towers]
            comb_gen_img = tf.concat(0, comb_gen_img)
            comb_pix_distrib = [t.model.gen_distrib for t in towers]
            comb_pix_distrib = tf.concat(0, comb_pix_distrib)
            comb_gen_states = [t.model.gen_states for t in towers]
            comb_gen_states = tf.concat(0, comb_gen_states)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)
            saver.restore(sess, conf['pretrained_model'])

            def predictor_func(input_images, one_hot_images, input_state, input_actions):
                """
                :param one_hot_images: the first two frames
                :param pixcoord: the coords of the disgnated pixel in images coord system
                :return: the predicted pixcoord at the end of sequence
                """

                feed_dict = {}
                for t in towers:
                    feed_dict[t.model.iter_num] = 0
                    feed_dict[t.model.lr] = 0.0
                    feed_dict[t.start_images] = input_images
                    feed_dict[t.start_states] = input_state
                    feed_dict[t.actions] = input_actions
                    feed_dict[t.pix_distrib] = one_hot_images

                # pack all towers operations which need to be evaluated in one list

                gen_images, gen_distrib, gen_states = sess.run([comb_gen_img,
                                                              comb_pix_distrib,
                                                              comb_gen_states],
                                                              feed_dict)

                return gen_distrib, gen_images, None, gen_states

            return predictor_func