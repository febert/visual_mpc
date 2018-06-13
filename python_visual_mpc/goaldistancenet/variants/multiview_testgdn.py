import tensorflow as tf
import numpy as np
import pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher
from python_visual_mpc.goaldistancenet.gdnet import apply_warp
import collections
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet
from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn

def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

class MulltiviewTestGDN():
    def __init__(self,
                 conf = None,
                 build_loss=False,
                 load_data=False
                 ):
        self.conf = conf
        self.gdn = []

        self.img_height = conf['orig_size'][0]
        self.img_width = conf['orig_size'][1]
        self.ncam = conf['ncam']

        self.I0_pl = []
        self.I1_pl = []
        for n in range(conf['ncam']):
            self.gdn.append(GoalDistanceNet(conf=conf, build_loss=False, load_data = False))
            self.I0_pl.append(self.gdn[-1].I0_pl)
            self.I1_pl.append(self.gdn[-1].I1_pl)
        self.I0_pl = tf.stack(self.I0_pl, axis=0)
        self.I1_pl = tf.stack(self.I1_pl, axis=0)

    def build_net(self):
        self.warped_I0_to_I1 = []
        self.flow_bwd = []
        self.warp_pts_bwd = []

        self.scopenames = []
        for n in range(self.conf['ncam']):
            name = "gdncam{}".format(n)
            with tf.variable_scope(name):
                self.gdn[n].build_net()

            self.warped_I0_to_I1.append(self.gdn[-1].warped_I0_to_I1)
            self.flow_bwd.append(self.gdn[-1].flow_bwd)
            self.warp_pts_bwd.append(self.gdn[-1].warp_pts_bwd)
            self.scopenames.append(name)
        self.warped_I0_to_I1 = tf.stack(self.warped_I0_to_I1, axis=0)
        self.flow_bwd = tf.stack(self.flow_bwd, axis=0)
        self.warp_pts_bwd = tf.stack(self.warp_pts_bwd, axis=0)


    def restore(self, sess):
        for n in range(self.conf['ncam']):
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scopenames[n])
            modelfile = self.conf['pretrained_model'][n]
            vars = variable_checkpoint_matcher(self.conf, vars, modelfile)
            saver = tf.train.Saver(vars, max_to_keep=0)
            saver.restore(sess, modelfile)
            print('gdn{} restore done.'.format(n))
