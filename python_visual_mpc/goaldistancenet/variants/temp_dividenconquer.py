import tensorflow as tf
import numpy as np
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

        self.build_net()
        if build_loss:
            self.combine_losses()
            self.image_summaries = self.build_image_summary([self.I0, self.I1, self.gen_I1, length(self.flow_bwd)])

    def build_net(self):
        n_layer = int(np.log2(self.seq_len))

        if 'fwd_bwd' not in self.conf:
            occ_bwd = self.occ_bwd

        used = False
        flow_bwd_lm1 = None
        cons_loss_per_layer_l = []

        for l in range(n_layer):
            tstep = int(np.power(2,l+1))
            flow_bwd_l = []

            cons_loss_per_layer = 0
            for t in range(0, self.seq_len, tstep):

                I0 = self.images[:, t]
                I1 = self.images[:, t + tstep-1]
                with tf.variable_scope('warpnet', reuse=used):
                    gen_I1, warp_pts_bwd, flow_bwd, _ = self.warp(I0, I1)

                self.add_pair_loss(I1, gen_I1, occ_bwd, flow_bwd)
                used = True
                if flow_bwd_lm1 is not None:
                    cons_loss_per_layer += self.consistency_loss(t, tstep, flow_bwd_lm1, flow_bwd, occ_bwd)
                flow_bwd_l.append(flow_bwd)

            cons_loss_per_layer_l.append(cons_loss_per_layer)
            flow_bwd_lm1 = flow_bwd_l

        self.I0, self.I1, self.gen_I1 =  I0, I1, gen_I1

    def consistency_loss(self, tind, tstep, flow_bwd_lm1, flow_bwd, occ_bwd):
        lower_level_flow = 0
        for t in range(tind, tind+tstep, tstep/2):
            lower_level_flow += flow_bwd_lm1[tind]
        return tf.reduce_mean(tf.square(lower_level_flow - flow_bwd)*occ_bwd)

def calc_warpscores(flow_field):
    return np.sum(np.linalg.norm(flow_field, axis=3), axis=[2, 3])


if __name__ == '__main__':
    filedir = '/home/frederik/Documents/catkin_ws/src/visual_mpc/tensorflow_data/gdn/hardthres/modeldata'
    conf = {}
    conf['output_dir'] = filedir
    make_plots(conf, filename= filedir + '/data.pkl')