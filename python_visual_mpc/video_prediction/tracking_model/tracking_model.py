import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.lstm_ops12 import basic_conv_lstm_cell
from python_visual_mpc.misc.zip_equal import zip_equal

import collections
import pickle
from python_visual_mpc.video_prediction.utils_vpred.animate_tkinter import Visualizer_tkinter
import pdb

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
from python_visual_mpc.video_prediction.basecls.utils.transformations import dna_transformation, cdna_transformation
from python_visual_mpc.video_prediction.basecls.prediction_model_basecls import Base_Prediction_Model
from python_visual_mpc.flow.trafo_based_flow.correction import compute_motion_vector_cdna, compute_motion_vector_dna

from python_visual_mpc.flow.trafo_based_flow.correction import Trafo_Flow
from python_visual_mpc.flow.descriptor_based_flow.descriptor_flow_model import Descriptor_Flow

class Tracking_Model(Base_Prediction_Model):
    def __init__(self,
                conf = None,
                trafo_pix = True,
                load_data = True,
                mode=True):
        self.build_tracker = conf['tracker']
        Base_Prediction_Model.__init__(self,
                                        conf = conf,
                                        trafo_pix = trafo_pix,
                                        load_data = load_data,
                                        )

    def build(self):
        """
            Build the tracking-prediction network
            :return:
        """

        batch_size, img_height, img_width, self.color_channels = self.images[0].get_shape()[0:4]

        if self.states != None:
            current_state = self.states[0]
        else:
            current_state = None

        if self.k == -1:
            feedself = True
        else:
            # Scheduled sampling:
            # Calculate number of ground-truth frames to pass in.
            self.num_ground_truth = tf.to_int32(
                tf.round(tf.to_float(batch_size) * (self.k / (self.k + tf.exp(self.iter_num / self.k)))))
            feedself = False

        # LSTM state sizes and states.

        if 'lstm_size' in self.conf:
            self.lstm_size = self.conf['lstm_size']
            print('using lstm size', self.lstm_size)
        else:
            self.lstm_size = np.int32(np.array([16, 32, 64, 32, 16]))

        self.lstm_state1, self.lstm_state2, self.lstm_state3, self.lstm_state4 = None, None, None, None
        self.lstm_state5, self.lstm_state6, self.lstm_state7 = None, None, None

        self.tracking_kerns = []
        self.tracking_gen_images = []
        self.tracking_flow01 = []
        self.tracking_flow10 = []
        self.tracking_gendistrib = []
        self.descp0 = []
        self.descp1 = []

        for t, image, action in zip(list(range(len(self.images)-1)), self.images[:-1], self.actions[:-1]):
            print(t)


            self.prev_image, self.prev_pix_distrib1, self.prev_pix_distrib2 = self.get_input_image(
                feedself,
                image,
                t)

            self.reuse = bool(self.gen_images)
            current_state = self.build_network_core(action, current_state, self.prev_image)

            print('building tracker...')
            tracker = self.build_tracker(self.conf,
                                [self.images[t],
                                self.images[t + 1]],
                                pix_distrib_input= self.prev_pix_distrib1,
                                reuse=self.reuse)

            self.tracking_gendistrib.append(tracker.gen_distrib_output)
            self.tracking_gen_images.append(tracker.gen_images)
            self.tracking_kerns.append(tracker.kernels)


            self.tracking_flow01.append(tracker.flow_vectors01)
            if isinstance(tracker, Descriptor_Flow):
                if 'forward_backward' in self.conf:
                    self.tracking_flow10.append(tracker.flow_vectors10)
                self.descp0.append(tracker.d0)
                self.descp1.append(tracker.d1)

        self.build_loss()


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
                    list(range(len(self.gen_images))), self.images[self.conf['context_frames']:],
                    self.gen_images[self.conf['context_frames'] - 1:]):
                recon_cost_mse = self.mean_squared_error(x, gx)
                summaries.append(tf.summary.scalar('recon_cost' + str(i), recon_cost_mse))
                total_recon_cost += recon_cost_mse
            summaries.append(tf.summary.scalar('total reconst cost', total_recon_cost))
            loss += total_recon_cost

            if ('ignore_state_action' not in self.conf) and ('ignore_state' not in self.conf):
                for i, state, gen_state in zip(
                        list(range(len(self.gen_states))), self.states[self.conf['context_frames']:],
                        self.gen_states[self.conf['context_frames'] - 1:]):
                    state_cost = self.mean_squared_error(state, gen_state) * 1e-4 * self.conf['use_state']
                    summaries.append(tf.summary.scalar('state_cost' + str(i), state_cost))
                    loss += state_cost

            #tracking frame matching cost:
            total_frame_match_cost = 0
            for i, im, gen_im in zip_equal(list(range(len(self.tracking_gen_images))),
                                       self.images[1:], self.tracking_gen_images):
                cost = self.mean_squared_error(im, gen_im) * self.conf['track_agg_fact']
                total_frame_match_cost += cost
            summaries.append(tf.summary.scalar('total_frame_match_cost', total_frame_match_cost))
            loss += total_frame_match_cost

            #adding transformation aggreement cost:
            total_trans_agg_cost = 0
            for i, k1, k2 in zip_equal(list(range(len(self.tracking_kerns))), self.tracking_kerns, self.pred_kerns):
                cost = self.mean_squared_error(k1, k2) * self.conf['track_agg_fact']
                total_trans_agg_cost += cost
            summaries.append(tf.summary.scalar('total_trans_agg_cost', total_trans_agg_cost))
            loss += total_trans_agg_cost

            self.loss = loss = loss / np.float32(len(self.images) - self.conf['context_frames'])

            summaries.append(tf.summary.scalar('loss', loss))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, self.global_step)
            self.summ_op = tf.summary.merge(summaries)


    def visualize(self, sess, images, actions, states):
        print('-------------------------------------------------------------------')
        print('verify current settings!! ')
        for key in list(self.conf.keys()):
            print(key, ': ', self.conf[key])
        print('-------------------------------------------------------------------')

        import re
        itr_vis = re.match('.*?([0-9]+)$', self.conf['visualize']).group(1)

        self.saver.restore(self.sess, self.conf['visualize'])
        print('restore done.')

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

        pickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
        print('written files to:' + file_path)

        v = Visualizer_tkinter(dict, numex=10, append_masks=True, filepath=self.conf['output_dir'])
        v.build_figure()
