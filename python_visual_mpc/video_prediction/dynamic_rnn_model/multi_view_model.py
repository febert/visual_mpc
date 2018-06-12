import tensorflow as tf
from python_visual_mpc.video_prediction.read_tf_records2 import build_tfrecord_input as build_tfrecord_fn
from python_visual_mpc.video_prediction.utils_vpred.video_summary import make_video_summaries
from python_visual_mpc.video_prediction.dynamic_rnn_model.dynamic_base_model import Dynamic_Base_Model

import pdb
from python_visual_mpc.video_prediction.basecls.utils.visualize import visualize_diffmotions, visualize, compute_metric

class Multi_View_Model(object):
    def __init__(self,
                 conf = None,
                 images=None,
                 actions=None,
                 states=None,
                 pix_distrib=None,
                 load_data=True,
                 build_loss=True
                 ):

        self.conf = conf
        self.train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self.iter_num = tf.placeholder(tf.float32, [])
        self.build_loss = build_loss

        if images is None:
            dict = build_tfrecord_fn(conf, mode='train')
            train_images, train_actions, train_states = dict['images'], dict['actions'], dict['endeffector_pos']
            dict = build_tfrecord_fn(conf, mode='val')
            val_images, val_actions, val_states = dict['images'], dict['actions'], dict['endeffector_pos']

            self.images, self.actions, self.states = tf.cond(self.train_cond > 0,  # if 1 use trainigbatch else validation batch
                                              lambda: [train_images, train_actions, train_states],
                                              lambda: [val_images, val_actions, val_states])
            if 'use_len' in conf:
                self.images, self.states, self.actions = self.random_shift(self.images, self.states, self.actions)
            pix_distrib = None
        else:
            self.images = images
            self.actions = actions
            self.states = states

        self.models = []
        for icam in range(conf['ncam']):
            if 'multi_gpu' in conf:
                with tf.device('/gpu:%d' % icam):
                    with tf.variable_scope('icam{}'.format(icam)):
                        self.models.append(self.buildnet(icam, conf, self.images, pix_distrib, self.states, self.actions))
            else:
                with tf.variable_scope('icam{}'.format(icam)):
                    self.models.append(self.buildnet(icam, conf, self.images, pix_distrib, self.states, self.actions))

        if build_loss:
            self.loss = 0.
            for icam in range(conf['ncam']):
                self.loss += self.models[icam].loss
            self.train_op = tf.group([m.train_op for m in self.models])
            self.train_summ_op = tf.summary.merge([m.train_summ_op for m in self.models])
            self.val_summ_op = tf.summary.merge([m.val_summ_op for m in self.models])

        self.gen_images = tf.concat([m.gen_images for m in self.models], axis=2)
        self.gen_states = tf.concat([m.gen_states for m in self.models], axis=2)
        if pix_distrib is not None:
            self.gen_distrib = tf.concat([m.gen_distrib for m in self.models], axis=2)

        if build_loss:
            self.train_video_summaries = make_video_summaries(conf['context_frames'], [self.images[:,:,0], self.gen_images[:,:,0],
                                                                                       self.images[:,:,1], self.gen_images[:,:,1]], 'train_images')
            self.val_video_summaries = make_video_summaries(conf['context_frames'], [self.images[:,:,0], self.gen_images[:,:,0],
                                                                                     self.images[:,:,1], self.gen_images[:,:,1]], 'val_images')

    def buildnet(self, icam, conf, images, pix_distrib, states, actions):
        print('building network for cam{}'.format(icam))
        images = images[:, :, icam]
        if pix_distrib is not None:
            pix_distrib = pix_distrib[:, :, icam]
        model = Dynamic_Base_Model(conf, images, actions, states, pix_distrib=pix_distrib,
                                   build_loss=self.build_loss, load_data=False, iternum=self.iter_num)
        return model

    def random_shift(self, images, states, actions):
        print('shifting the video sequence randomly in time')
        tshift = 3
        uselen = self.conf['use_len']
        fulllength = self.conf['sequence_length']
        nshifts = (fulllength - uselen) // tshift + 1
        rand_ind = tf.random_uniform([1], 0, nshifts, dtype=tf.int64)
        self.rand_ind = rand_ind
        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(4, dtype=tf.int64)])
        images_sel = tf.slice(images, start, [-1, uselen, -1, -1, -1, -1])
        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(1, dtype=tf.int64)])
        actions_sel = tf.slice(actions, start, [-1, uselen, -1])
        start = tf.concat(axis=0, values=[tf.zeros(1, dtype=tf.int64), rand_ind * tshift, tf.zeros(1, dtype=tf.int64)])
        states_sel = tf.slice(states, start, [-1, uselen, -1])
        return images_sel, states_sel, actions_sel

    def visualize(self, sess):
        visualize(sess, self.conf, self)
