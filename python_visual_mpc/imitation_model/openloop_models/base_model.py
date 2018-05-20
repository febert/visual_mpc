from python_visual_mpc.imitation_model.attention_models.base_model import BaseAttentionModel
import tensorflow as tf
from python_visual_mpc.imitation_model.imitation_model import ImitationBaseModel, NUMERICAL_EPS, gen_mix_samples
import tensorflow.contrib.slim as slim
import numpy as np

class LSTMAttentionOpenLoop(BaseAttentionModel):
    def build(self, is_Train = True):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1

        input_images = tf.reshape(self.images[:, :-1,:,:,:], shape=(in_batch * in_time, in_rows, in_cols, 3))
        output_end_effector = self.gtruth_endeffector_pos[:, 1:]

        #builds convolutional feature points
        conv_features = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))

        prev_dec_out = conv_features

        for i in range(self.conf['num_repeats']):
            #decoder cell
            with tf.variable_scope('stack_{}'.format(i)):
                dec_masked_self_attention = self._multihead_attention(prev_dec_out, prev_dec_out, causal_mask=True, is_training=is_Train)
                dec_out = self._lstmforward_layer(dec_masked_self_attention)

            prev_dec_out = dec_out

        self._build_openloop_loss(prev_dec_out, output_end_effector)

class LSTMOpenLoopActions(BaseAttentionModel):
    def query(self, sess, traj, t, images = None, end_effector = None, actions = None):
        if t == 0:
            f_dict = {self.input_images: images, self.input_end_effector : end_effector}
            mdn_mix, mdn_std_dev, mdn_means = sess.run([self.mixing_parameters, self.std_dev, self.means],
                                                        feed_dict=f_dict)
            self.actions = np.zeros((15, self.adim), dtype=np.float64)
            for i in range(15):
                samps, samps_log_l = gen_mix_samples(self.conf.get('N_GEN', 200), mdn_means[0, i], mdn_std_dev[0, i], mdn_mix[0, i])
                self.actions[i] = samps[0, :].astype(np.float64)

        return self.actions[t] + traj.target_qpos[t, :] * traj.mask_rel
         
    def build_sim(self):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        num_mix = self.conf['MDN_loss']
        in_batch, in_rows, in_cols = self.images.get_shape()[0], self.images.get_shape()[2], self.images.get_shape()[3]
        in_time, feature_dim, T = 1, 132, 15
        conv_in = tf.reshape(self.images, shape=(in_batch, in_rows, in_cols, 3))
        conv_features = tf.reshape(self._build_conv_layers(conv_in), shape=(in_batch, in_time, -1))

        input_features = tf.concat([conv_features, self.gtruth_endeffector_pos[:,:,:-1]], -1)
        zero_pad =  tf.zeros((in_batch, T - in_time, feature_dim), dtype = conv_features.dtype)
        prev_dec_out = tf.concat([input_features , zero_pad], 1)
        
        print('LSTM generation with {} time-steps'.format(T))
        for j in range(self.conf['num_repeats']):
            with tf.variable_scope('stack_{}'.format(j)):
                prev_dec_out = self._lstmforward_layer(prev_dec_out, is_training = False)

        with tf.variable_scope('full_loss'):
            mixture_activations = slim.layers.fully_connected(prev_dec_out, (2 + self.sdim) * num_mix,
                                                              scope='predicted_mixtures', activation_fn=None)
            mixture_activations = tf.reshape(mixture_activations, shape=(-1, num_mix, 2 + self.sdim))
        
        self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
        self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
        self.means = mixture_activations[:, :, 2:]

        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=[in_batch, T, num_mix])
        self.std_dev = tf.reshape(self.std_dev, shape=[in_batch, T, num_mix])
        self.means = tf.reshape(self.means, shape=[in_batch, T, num_mix, self.sdim])

        return self.mixing_parameters, self.std_dev, self.means

    def build(self, is_Train = True):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        
        conv_in = tf.reshape(self.images, shape=(in_batch * in_time, in_rows, in_cols, 3))

        #builds convolutional feature points
        conv_features = tf.reshape(self._build_conv_layers(conv_in), shape=(in_batch, in_time, -1))

        input_features = tf.concat([conv_features, self.gtruth_endeffector_pos[:,:,:-1]], -1)
        feature_dim = input_features.get_shape()[-1]

        losses, open_loop_logl, open_loop_l2 = [], None, None
        for i in range(in_time):
            #decoder cell
            zero_pad = tf.zeros((in_batch, in_time - 1 - i, feature_dim), dtype = conv_features.dtype)
            frame_features = tf.reshape(input_features[:, i], shape=(in_batch, 1, feature_dim))
            prev_dec_out = tf.concat([frame_features , zero_pad], 1)
            print('LSTM generation with {} time-steps'.format(in_time - i))
            for j in range(self.conf['num_repeats']):
                with tf.variable_scope('stack_{}'.format(j), reuse =  i > 0):
                    prev_dec_out = self._lstmforward_layer(prev_dec_out, is_training = is_Train)

            with tf.variable_scope('full_loss', reuse = i > 0):
                self._build_loss(prev_dec_out, self.gtruth_actions[:,i:], in_time - i)
            losses.append(self.summaries.pop('loss'))

            if i == 0:
                open_loop_logl, open_loop_l2 = self.summaries.pop('log_likelihood'), self.summaries.pop('diagnostic_l2loss')

        self.loss = tf.reduce_sum(losses)

        self.summaries['loss'] = self.loss
        self.summaries['log_likelihood'] = open_loop_logl
        self.summaries['diagnostic_l2loss'] = open_loop_l2
class LSTMOpenLoopStates(BaseAttentionModel):
    def query(self, sess, traj, t, images = None, end_effector = None, actions = None):
        if t == 0:
            f_dict = {self.input_images: images, self.input_end_effector : end_effector}
            mdn_mix, mdn_std_dev, mdn_means = sess.run([self.mixing_parameters, self.std_dev, self.means],
                                                        feed_dict=f_dict)
            self.states = np.zeros((15, self.adim), dtype=np.float64)
            for i in range(15):
                samps, samps_log_l = gen_mix_samples(self.conf.get('N_GEN', 200), mdn_means[0, i], mdn_std_dev[0, i], mdn_mix[0, i])
                self.states[i] = samps[0, :].astype(np.float64)
                if self.states[i, -1] > 0.05:
                    self.states[i, -1] = 1
                else:
                    self.states[i, -1] = -1
        
        return self.states[t]

    def build_sim(self):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        num_mix = self.conf['MDN_loss']
        in_batch, in_rows, in_cols = self.images.get_shape()[0], self.images.get_shape()[2], self.images.get_shape()[3]
        in_time, feature_dim, T = 1, 132, 15
        conv_in = tf.reshape(self.images, shape=(in_batch, in_rows, in_cols, 3))
        conv_features = tf.reshape(self._build_conv_layers(conv_in), shape=(in_batch, in_time, -1))

        input_features = tf.concat([conv_features, self.gtruth_endeffector_pos[:,:,:-1]], -1)
        zero_pad =  tf.zeros((in_batch, T - in_time, feature_dim), dtype = conv_features.dtype)
        prev_dec_out = tf.concat([input_features , zero_pad], 1)

        print('LSTM generation with {} time-steps'.format(T))
        for j in range(self.conf['num_repeats']):
            with tf.variable_scope('stack_{}'.format(j)):
                prev_dec_out = self._lstmforward_layer(prev_dec_out, is_training = False)

        with tf.variable_scope('full_loss'):
            mixture_activations = slim.layers.fully_connected(prev_dec_out, (2 + self.sdim) * num_mix,
                                                              scope='predicted_mixtures', activation_fn=None)
            mixture_activations = tf.reshape(mixture_activations, shape=(-1, num_mix, 2 + self.sdim))

        self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
        self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
        self.means = mixture_activations[:, :, 2:]

        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=[in_batch, T, num_mix])
        self.std_dev = tf.reshape(self.std_dev, shape=[in_batch, T, num_mix])
        self.means = tf.reshape(self.means, shape=[in_batch, T, num_mix, self.sdim])

        return self.mixing_parameters, self.std_dev, self.means

    def build(self, is_Train = True):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        in_batch, in_time, in_rows, in_cols, _ = self.images[:, :-1].get_shape()

        conv_in = tf.reshape(self.images[:,:-1], shape=(in_batch * in_time, in_rows, in_cols, 3))

        #builds convolutional feature points
        conv_features = tf.reshape(self._build_conv_layers(conv_in), shape=(in_batch, in_time, -1))

        input_features = tf.concat([conv_features, self.gtruth_endeffector_pos[:,:-1,:-1]], -1)
        feature_dim = input_features.get_shape()[-1]

        losses, open_loop_logl, open_loop_l2 = [], None, None
        for i in range(in_time):
            #decoder cell
            zero_pad = tf.zeros((in_batch, in_time - 1 - i, feature_dim), dtype = conv_features.dtype)
            frame_features = tf.reshape(input_features[:, i], shape=(in_batch, 1, feature_dim))
            prev_dec_out = tf.concat([frame_features , zero_pad], 1)
            print('LSTM generation with {} time-steps'.format(in_time - i))
            for j in range(self.conf['num_repeats']):
                with tf.variable_scope('stack_{}'.format(j), reuse =  i > 0):
                    prev_dec_out = self._lstmforward_layer(prev_dec_out, is_training = is_Train)

            with tf.variable_scope('full_loss', reuse = i > 0):
                self._build_loss(prev_dec_out, self.gtruth_endeffector_pos[:,i + 1:], in_time - i)
            losses.append(self.summaries.pop('loss'))

            if i == 0:
                open_loop_logl, open_loop_l2 = self.summaries.pop('log_likelihood'), self.summaries.pop('diagnostic_l2loss')

        self.loss = tf.reduce_sum(losses)

        self.summaries['loss'] = self.loss
        self.summaries['log_likelihood'] = open_loop_logl
        self.summaries['diagnostic_l2loss'] = open_loop_l2

