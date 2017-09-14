import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.lstm_ops12 import basic_conv_lstm_cell
from python_visual_mpc.misc.zip_equal import zip_equal

import pdb

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12
from python_visual_mpc.video_prediction.basecls.utils.transformations import dna_transformation, cdna_transformation
from python_visual_mpc.video_prediction.basecls.prediction_model_basecls import Base_Prediction_Model
from python_visual_mpc.flow.trafo_based_flow.correction import compute_motion_vector_cdna, compute_motion_vector_dna, construct_correction

class Tracking_Model(Base_Prediction_Model):
    def __init__(self,
                conf = None,
                trafo_pix = True,
                load_data = True,
                mode=True):
        Base_Prediction_Model.__init__(self,
                                        conf = conf,
                                        trafo_pix = trafo_pix,
                                        load_data = load_data,
                                        mode=mode)


    def build(self):
        """
            Build the tracking-prediction network
            :return:
        """

        if 'kern_size' in self.conf.keys():
            KERN_SIZE = self.conf['kern_size']
        else:
            KERN_SIZE = 5

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
            print 'using lstm size', self.lstm_size
        else:
            self.lstm_size = np.int32(np.array([16, 32, 64, 32, 16]))

        self.lstm_state1, self.lstm_state2, self.lstm_state3, self.lstm_state4 = None, None, None, None
        self.lstm_state5, self.lstm_state6, self.lstm_state7 = None, None, None

        self.tracking_kerns = []
        self.tracking_flow = []

        t = -1
        for image, action in zip(self.images[:-1], self.actions[:-1]):
            t += 1
            print t


            self.prev_image, self.prev_pix_distrib1, self.prev_pix_distrib2 = self.get_input_image(
                feedself,
                image,
                t)

            self.reuse = bool(self.gen_images)
            current_state = self.build_network_core(action, current_state, image)

            flow_vectors, kernels = self.construct_correction(self.images[t], self.images[t+1])
            self.tracking_kerns.append(kernels)
            self.tracking_flow.append(flow_vectors)

        self.build_loss()


    def construct_correction(self,
                             image0,
                             image1,
                             pix_distrib_input=None):

        """Build network for predicting optical flow

        """
        print 'build tracker'
        gen_images, gen_masks, pix_distrib_output, flow_vectors, kernels = construct_correction(self.conf, [image0, image1], reuse=self.reuse)

        return flow_vectors, kernels

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

            for i, x, gx in zip(
                    range(len(self.gen_images)), self.images[self.conf['context_frames']:],
                    self.gen_images[self.conf['context_frames'] - 1:]):
                recon_cost_mse = self.mean_squared_error(x, gx)

                summaries.append(tf.summary.scalar('recon_cost' + str(i), recon_cost_mse))

                recon_cost = recon_cost_mse

                loss += recon_cost

            if ('ignore_state_action' not in self.conf) and ('ignore_state' not in self.conf):
                for i, state, gen_state in zip(
                        range(len(self.gen_states)), self.states[self.conf['context_frames']:],
                        self.gen_states[self.conf['context_frames'] - 1:]):
                    state_cost = self.mean_squared_error(state, gen_state) * 1e-4 * self.conf['use_state']
                    summaries.append(
                        tf.summary.scalar('state_cost' + str(i), state_cost))
                    loss += state_cost

            #adding tracking-prediction aggreement cost:
            for i, k1, k2 in zip_equal(range(len(self.tracking_kerns)), self.tracking_kerns, self.pred_dna_kerns):
                cost = self.mean_squared_error(k1, k2) * self.conf['track_agg_fact']
                summaries.append(
                    tf.summary.scalar('track_agg_cost' + str(i), cost))
                loss += cost

            self.loss = loss = loss / np.float32(len(self.images) - self.conf['context_frames'])

            summaries.append(tf.summary.scalar('loss', loss))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, self.global_step)
            self.summ_op = tf.summary.merge(summaries)
