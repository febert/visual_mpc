import tensorflow as tf
import numpy as np
import fnmatch
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cPickle
import sys
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
import matplotlib.pyplot as plt
from python_visual_mpc.video_prediction.read_tf_records2 import \
                build_tfrecord_input as build_tfrecord_fn
import matplotlib.gridspec as gridspec

from python_visual_mpc.video_prediction.utils_vpred.online_reader import OnlineReader
import tensorflow.contrib.slim as slim

from python_visual_mpc.utils.colorize_tf import colorize
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.utils_vpred.online_reader import read_trajectory

from python_visual_mpc.data_preparation.gather_data import make_traj_name_list

import collections
from python_visual_mpc.goaldistancenet.gdnet import GoalDistanceNet

def mean_square(x):
    return tf.reduce_mean(tf.square(x))

def length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 3))

class Temp_DnC_GDnet(GoalDistanceNet):
    def __init__(self,
                 conf = None,
                 build_loss=True,
                 load_data = True,
                 images = None,
                 iter_num = None,
                 ):
        GoalDistanceNet.__init__(self, conf = conf,
                                 build_loss=False,
                                 load_data = load_data,
                                 images = images,
                                 iter_num = iter_num,
                                 )
        self.build_loss = build_loss
        self.load_data = load_data

    def build_net(self):
        if self.load_data:
            self.build_cons_model()
        else:
            self.gen_I1, self.warp_pts_bwd, self.flow_bwd, _ = self.warp(self.I0, self.I1)
            self.gen_I0, self.flow_fwd = None, None

        if self.build_loss:
            self.combine_losses()
            if 'sched_layer_train' in self.conf:
                self.sched_layer_train()

    def merge_t_losses(self):
        "add together losses with same name"
        print 'merging same ts'
        merged_losses = {}

        loss_list = []
        for n in self.losses.keys():
            if '/t' in n:
                n = str.split(n,'/')[:-1]
                n ='/'.join(n)
            loss_list.append(n)
        unique_names_l = set(loss_list)

        for uname in unique_names_l:
            comb_loss_val = []
            for l_ in self.losses.keys():
                if uname in l_:
                    comb_loss_val.append(self.losses[l_])
                    print "merging", l_
            print '-----'
            comb_loss_val = tf.reduce_mean(tf.stack(comb_loss_val))
            merged_losses[uname] = comb_loss_val

        self.losses = merged_losses

    def sched_layer_train(self):
        thresholds = self.conf['sched_layer_train']
        for l in range(self.n_layer):
            layer_mult = tf.cast(self.iter_num > thresholds[l], tf.float32)
            for k in self.losses.keys():
                if 'l{}'.format(l) in k:
                    self.losses[k] = layer_mult
                    print 'multiplying {} with layer_mult{}'.format(k, l)

    def combine_losses(self):
        self.merge_t_losses()
        super(Temp_DnC_GDnet, self).combine_losses()

    def build_cons_model(self):
        self.n_layer = int(np.log2(self.seq_len)) + 1
        if 'fwd_bwd' not in self.conf:
            occ_bwd = self.occ_bwd
        used = False
        flow_bwd_lm1 = None
        for l in range(self.n_layer):
            tstep = int(np.power(2, l))
            flow_bwd_l = []

            cons_loss_per_layer = 0
            for i, t in enumerate(range(0, self.seq_len - 1, tstep)):

                print 'l{}, t{}, warping im{} to im{}'.format(l, t, t, t + tstep)
                I0 = self.images[:, t]
                I1 = self.images[:, t + tstep]

                with tf.variable_scope('warpnet', reuse=used):
                    gen_I1, warp_pts_bwd, flow_bwd, _ = self.warp(I0, I1)

                self.add_pair_loss(I1, gen_I1, occ_bwd, flow_bwd, suf='/l{}/t{}'.format(l,t))
                used = True

                if flow_bwd_lm1 is not None:
                    cons_loss_per_layer += self.consistency_loss(i, flow_bwd_lm1, flow_bwd, occ_bwd)
                flow_bwd_l.append(flow_bwd)

                if i == 0:
                    self.image_summaries = self.build_image_summary(
                        [I0, I1, gen_I1, length(flow_bwd)],
                        name='warp_im{}_to_im{}'.format(t, t + tstep))

            self.losses['cons_loss/l{}'.format(l)] = cons_loss_per_layer*self.conf['cons_loss']
            flow_bwd_lm1 = flow_bwd_l

    def consistency_loss(self, i, flow_bwd_lm1, flow_bwd, occ_bwd):
        lower_level_flow = flow_bwd_lm1[i*2] + flow_bwd_lm1[i*2+1]
        return tf.reduce_mean(tf.square(lower_level_flow - flow_bwd)*occ_bwd[:,:,:, None])

def calc_warpscores(flow_field):
    return np.sum(np.linalg.norm(flow_field, axis=3), axis=[2, 3])


if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/hardthres/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')