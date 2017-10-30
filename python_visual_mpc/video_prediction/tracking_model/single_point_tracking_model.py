import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.lstm_ops12 import basic_conv_lstm_cell
from python_visual_mpc.misc.zip_equal import zip_equal

import collections
import cPickle
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
import pdb
from tensorflow.contrib.layers import layer_norm
from python_visual_mpc.video_prediction.dynamic_rnn_model.ops import dense, pad2d, conv1d, conv2d, conv3d, upsample_conv2d, conv_pool2d, lrelu, instancenorm, flatten
# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
from python_visual_mpc.video_prediction.basecls.utils.transformations import dna_transformation, cdna_transformation
from python_visual_mpc.video_prediction.basecls.prediction_model_basecls import Base_Prediction_Model
from python_visual_mpc.flow.trafo_based_flow.correction import compute_motion_vector_cdna, compute_motion_vector_dna

from python_visual_mpc.flow.trafo_based_flow.correction import Trafo_Flow
from python_visual_mpc.flow.descriptor_based_flow.descriptor_flow_model import Descriptor_Flow

from python_visual_mpc.video_prediction.dynamic_rnn_model.dynamic_base_model import Dynamic_Base_Model


from python_visual_mpc.video_prediction.dynamic_rnn_model.layers import instance_norm
class Single_Point_Tracking_Model(Dynamic_Base_Model):
    def __init__(self,
                conf = None,
                trafo_pix = True,
                load_data = True,
                mode=True):
        self.build_tracker = conf['tracker']

        Dynamic_Base_Model.__init__(self,
                                        conf = conf,
                                        trafo_pix = trafo_pix,
                                        load_data = load_data,
                                        build_loss = False
                                        )

        self.layer_normalization = conf['normalization']
        if conf['normalization'] == 'ln':
            self.normalizer_fn = layer_norm
        elif conf['normalization'] == 'in':
            self.normalizer_fn = instance_norm
        elif conf['normalization'] == 'none':
            self.normalizer_fn = lambda x: x
        else:
            raise ValueError('Invalid layer normalization %s' % self.layer_normalization)

        with tf.variable_scope('tracker'):
            self.build()

    def build(self):
        """
            Build the tracking-prediction network
            :return:
        """

        self.gen_
        for i in range()
        self.build_descriptors()



    def build_descriptors():
        with tf.variable_scope('h0'):
            h0 = conv_pool2d(image, vgf_dim, kernel_size=(5, 5), strides=(2, 2))
            h0 = self.normalizer_fn(h0)
            h0 = tf.nn.relu(h0)
        with tf.variable_scope('h1'):
            h1 = conv_pool2d(, vgf_dim * 2, kernel_size = (3, 3), strides = (2, 2))
            h1 = self.normalizer_fn(h1)
            h1 = tf.nn.relu(h1)
        with tf.variable_scope('h2'):
            h2 = conv_pool2d(lstm_h1, vgf_dim * 4, kernel_size=(3, 3), strides=(2, 2))
            h2 = self.normalizer_fn(h2)
            h2 = tf.nn.relu(h2)
        with tf.variable_scope('h3'):
            h3 = upsample_conv2d(lstm_h2, vgf_dim * 2, kernel_size=(3, 3), strides=(2, 2))
            h3 = self.normalizer_fn(h3)
            h3 = tf.nn.relu(h3)
        with tf.variable_scope('h4'):
            h4 = upsample_conv2d(tf.concat([lstm_h3, h1], axis=-1), vgf_dim, kernel_size=(3, 3), strides=(2, 2))
            h4 = self.normalizer_fn(h4)
            h4 = tf.nn.relu(h4)
        with tf.variable_scope('h5'):
            h5 = upsample_conv2d(tf.concat([lstm_h4, h0], axis=-1), vgf_dim, kernel_size=(3, 3), strides=(2, 2))
            h5 = self.normalizer_fn(h5)
            h5 = tf.nn.relu(h5)
        with tf.variable_scope('h6_masks'):
            h6_masks = conv2d(h5, vgf_dim, kernel_size=(3, 3), strides=(1, 1))
            h6_masks = self.normalizer_fn(h6_masks)
            h6_masks = tf.nn.relu(h6_masks)

    self.build_loss()

    def compute_descriptors(self):




    def build_loss(self):
        summaries = []

        self.global_step = tf.Variable(0, trainable=False)
        if self.conf['learning_rate'] == 'scheduled' and not self.visualize:
            print('using scheduled learning rate')
            self.lr = tf.train.piecewise_constant(self.global_step, self.conf['lr_boundaries'], self.conf['lr_values'])
        else:
            self.lr = tf.placeholder_with_default(self.conf['learning_rate'], ())

        if not self.trafo_pix:
            # L2 loss, PSNR for eval.
            true_fft_list, pred_fft_list = [], []
            loss, psnr_all = 0.0, 0.0

            total_recon_cost = 0
            for i, x, gx in zip(
                    range(len(self.gen_images)), self.images[self.conf['context_frames']:],
                    self.gen_images[self.conf['context_frames'] - 1:]):
                recon_cost_mse = self.mean_squared_error(x, gx)
                summaries.append(tf.summary.scalar('recon_cost' + str(i), recon_cost_mse))
                total_recon_cost += recon_cost_mse
            summaries.append(tf.summary.scalar('total reconst cost', total_recon_cost))
            loss += total_recon_cost

            if ('ignore_state_action' not in self.conf) and ('ignore_state' not in self.conf):
                for i, state, gen_state in zip(
                        range(len(self.gen_states)), self.states[self.conf['context_frames']:],
                        self.gen_states[self.conf['context_frames'] - 1:]):
                    state_cost = self.mean_squared_error(state, gen_state) * 1e-4 * self.conf['use_state']
                    summaries.append(tf.summary.scalar('state_cost' + str(i), state_cost))
                    loss += state_cost

            #tracking frame matching cost:
            total_frame_match_cost = 0
            for i, im, gen_im in zip_equal(range(len(self.tracking_gen_images)),
                                       self.images[1:], self.tracking_gen_images):
                cost = self.mean_squared_error(im, gen_im) * self.conf['track_agg_fact']
                total_frame_match_cost += cost
            summaries.append(tf.summary.scalar('total_frame_match_cost', total_frame_match_cost))
            loss += total_frame_match_cost

            #adding transformation aggreement cost:
            total_trans_agg_cost = 0
            for i, k1, k2 in zip_equal(range(len(self.tracking_kerns)), self.tracking_kerns, self.pred_kerns):
                cost = self.mean_squared_error(k1, k2) * self.conf['track_agg_fact']
                total_trans_agg_cost += cost
            summaries.append(tf.summary.scalar('total_trans_agg_cost', total_trans_agg_cost))
            loss += total_trans_agg_cost

            self.loss = loss = loss / np.float32(len(self.images) - self.conf['context_frames'])

            summaries.append(tf.summary.scalar('loss', loss))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, self.global_step)
            self.summ_op = tf.summary.merge(summaries)


    def visualize(self, sess, images, actions, states):
        print '-------------------------------------------------------------------'
        print 'verify current settings!! '
        for key in self.conf.keys():
            print key, ': ', self.conf[key]
        print '-------------------------------------------------------------------'

        import re
        itr_vis = re.match('.*?([0-9]+)$', self.conf['visualize']).group(1)

        self.saver.restore(self.sess, self.conf['visualize'])
        print 'restore done.'

        feed_dict = {self.lr: 0.0,
                     self.iter_num: 0,
                     self.train_cond: 1}

        file_path = self.conf['output_dir']

        ground_truth, gen_images, gen_masks, pred_flow, track_flow = self.sess.run([self.images,
                                                                               self.gen_images,
                                                                               self.gen_masks,
                                                                               self.prediction_flow,
                                                                               self.tracking_flow01
                                                                               ],
                                                                              feed_dict)

        dict = collections.OrderedDict()
        dict['iternum'] = itr_vis
        dict['ground_truth'] = ground_truth
        dict['gen_images'] = gen_images
        dict['prediction_flow'] = pred_flow
        dict['tracking_flow'] = track_flow
        # dict['gen_masks'] = gen_masks

        cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
        print 'written files to:' + file_path

        v = Visualizer_tkinter(dict, numex=10, append_masks=True, filepath=self.conf['output_dir'])
        v.build_figure()
