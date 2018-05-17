import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from python_visual_mpc.imitation_model.imitation_model import ImitationBaseModel, gen_mix_samples, NUMERICAL_EPS

class BaseAttentionModel(ImitationBaseModel):
    def _create_pos_encoding(self, inputs):
        T_input, dim = tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        freq_dim_arg = tf.cast(tf.exp(tf.linspace(tf.constant(0., dtype=tf.float64), np.log(10000) * (- 2 * (dim - 1) / dim), dim)), dtype=tf.float32)
        freq_T_arg = tf.linspace(0., tf.cast(T_input - 1, tf.float32), T_input)
        freq_arg = tf.reshape(freq_T_arg, (-1, 1)) * tf.reshape(freq_dim_arg, (1, -1))
        freq_arg = tf.reshape(freq_arg, (-1, dim // 2, 2))

        sin_args, cos_args = tf.sin(freq_arg[:, :, 0]), tf.cos(freq_arg[:, :, 1])

        pos_encoding = tf.concat((tf.expand_dims(sin_args, -1), tf.expand_dims(cos_args, -1)), -1)

        return tf.cast(tf.stop_gradient(tf.expand_dims(tf.reshape(pos_encoding, (-1, dim)), 0)), dtype=inputs.dtype)
    def _multihead_attention(self, Q_in, K_in, causal_mask = False, is_training = False):
        """
        Credit to kyubyong park for a very helpful reference implementation https://github.com/Kyubyong/transformer/blob/master/modules.py
        """
        num_heads = self.conf['num_heads']
        d_head = int(Q_in.get_shape()[2]) // num_heads
        b_size = tf.shape(Q_in)[0]
        
        Q = tf.layers.dense(Q_in, d_head * num_heads) #(B, T_q, H * D) 
        K = tf.layers.dense(K_in, d_head * num_heads) #(B, T_k, H * D)
        V = tf.layers.dense(K_in, d_head * num_heads) #(B, T_k, H * D)

        Q_heads = tf.concat(tf.split(Q, num_heads, axis = 2), axis = 0) #(B * H, T_q,  D)
        K_heads = tf.concat(tf.split(K, num_heads, axis = 2), axis = 0) #(B * H, T_k,  D)
        V_heads = tf.concat(tf.split(V, num_heads, axis = 2), axis = 0) #(B * H, T_k,  D)

        scaled_dot_attention = tf.matmul(Q_heads, tf.transpose(K_heads, [0, 2, 1])) / np.sqrt(d_head) #(B * H, T_q, T_k)
        
        if causal_mask:
            c_mask = tf.ones_like(scaled_dot_attention[0, :, :])
            c_mask = tf.matrix_band_part(c_mask, -1, 0)
            c_mask = tf.tile(tf.expand_dims(c_mask, 0), [b_size * num_heads, 1, 1])

            neg_inf = tf.ones_like(c_mask) * (-2 ** 32 + 1)
            scaled_dot_attention = tf.where(tf.equal(c_mask, 0), neg_inf, scaled_dot_attention)
        
        
        attention_weights = tf.nn.softmax(scaled_dot_attention)
        
        
        if 'dropout' in self.conf:
            attention_weights = tf.layers.dropout(attention_weights, rate=self.conf['dropout'], training=tf.convert_to_tensor(is_training))
                
        output_heads = tf.matmul(attention_weights, V_heads) #(B * H, T_k, D)
        outputs = tf.concat(tf.split(output_heads, num_heads, axis = 0), axis = 2) + Q_in #(B, T_k, D * H)
        
        return tf.contrib.layers.layer_norm(outputs, begin_norm_axis = 2)

    def _feedforward_layer(self, inputs):
        num_inner, num_out = self.conf['feedforward_dim'], int(inputs.get_shape()[2])

        inner_mult = tf.layers.conv1d(inputs, num_inner, 1, activation=tf.nn.relu)
        outer_mult = tf.layers.conv1d(inner_mult, num_out, 1)

        return tf.contrib.layers.layer_norm(outer_mult + inputs, begin_norm_axis = 2)
    def _lstmforward_layer(self, inputs):
        num_out = int(inputs.get_shape()[2])
        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.conf['lstmforward_dim'])
        lstm_out, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=inputs, dtype=inputs.dtype)
        
        outer_mult = tf.layers.conv1d(lstm_out, num_out, 1)
        
        return tf.contrib.layers.layer_norm(outer_mult + inputs, begin_norm_axis = 2) 

class AttentionMDNStates(BaseAttentionModel):
    def query(self, sess, traj, t, images = None, end_effector = None, actions = None):

        f_dict = {self.input_images: images}
        mdn_mix, mdn_std_dev, mdn_means = sess.run([self.mixing_parameters, self.std_dev, self.means],
                                                        feed_dict=f_dict)
        
        samps, samps_log_l = gen_mix_samples(self.conf.get('N_GEN', 1000), mdn_means[0, -1], mdn_std_dev[0, -1], mdn_mix[0, -1])
        actions = samps[0, :].astype(np.float64)
        #actions = np.sum(mdn_means[0, -1] * mdn_mix[0, -1].reshape(-1, 1), axis = 0)
        
        if actions[-1] > 0.05:
            actions[-1] = 1
        else:
            actions[-1] = -1
        return actions
    def build_sim(self):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        num_mix = self.conf['MDN_loss']
        in_batch, in_rows, in_cols = self.images.get_shape()[0], self.images.get_shape()[2], self.images.get_shape()[3]
        in_time = tf.shape(self.images)[1]

        input_images = tf.reshape(self.images, shape=(-1, in_rows, in_cols, 3))
        
        #builds convolutional feature points

        conv_features = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, 128))
        
        pos_decoder_inputs = conv_features + self._create_pos_encoding(conv_features)
        prev_dec_out = pos_decoder_inputs

        for i in range(self.conf['num_repeats']):
            #decoder cell
            dec_masked_self_attention = self._multihead_attention(prev_dec_out, prev_dec_out, causal_mask=True, is_training=False)
            dec_out = self._feedforward_layer(dec_masked_self_attention)

            prev_dec_out = dec_out

        mixture_activations = slim.layers.fully_connected(prev_dec_out, (2 + self.sdim) * num_mix,
                                                              scope='predicted_mixtures', activation_fn=None)
        mixture_activations = tf.reshape(mixture_activations, shape=(-1, num_mix, 2 + self.sdim))
        self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
        self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
        self.means = mixture_activations[:, :, 2:]

        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=[in_batch, in_time, num_mix])
        self.std_dev = tf.reshape(self.std_dev, shape=[in_batch, in_time, num_mix])
        self.means = tf.reshape(self.means, shape=[in_batch, in_time, num_mix, self.sdim])

        return self.mixing_parameters, self.std_dev, self.means

    def build(self, is_Train = True):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1

        input_images = tf.reshape(self.images[:, :-1,:,:,:], shape=(in_batch * in_time, in_rows, in_cols, 3))
        output_end_effector = tf.reshape(self.gtruth_endeffector_pos[:, 1:], shape = (in_batch * in_time, self.sdim))
        #builds convolutional feature points
        
        conv_features = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))

        pos_decoder_inputs = conv_features + self._create_pos_encoding(conv_features)
        prev_dec_out = pos_decoder_inputs
        
        for i in range(self.conf['num_repeats']):
            #decoder cell
            dec_masked_self_attention = self._multihead_attention(prev_dec_out, prev_dec_out, causal_mask=True, is_training=is_Train)
            dec_out = self._feedforward_layer(dec_masked_self_attention)

            prev_dec_out = dec_out
        
        self._build_loss(prev_dec_out, output_end_effector, in_time)
        
        num_mix = self.conf['MDN_loss']
        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
        self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
        self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim))

class LSTMAttentionMDNStates(AttentionMDNStates):
    def build_sim(self):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        num_mix = self.conf['MDN_loss']
        in_batch, in_rows, in_cols = self.images.get_shape()[0], self.images.get_shape()[2], self.images.get_shape()[3]
        in_time = tf.shape(self.images)[1]

        input_images = tf.reshape(self.images, shape=(-1, in_rows, in_cols, 3))
        
        #builds convolutional feature points

        self.conv_features = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, 128))

        prev_dec_out = self.conv_features

        for i in range(self.conf['num_repeats']):
            #decoder cell
            with tf.variable_scope('stack_{}'.format(i)):
                dec_masked_self_attention = self._multihead_attention(prev_dec_out, prev_dec_out, causal_mask=True, is_training=False)
                dec_out = self._lstmforward_layer(dec_masked_self_attention)

            prev_dec_out = dec_out

        mixture_activations = slim.layers.fully_connected(prev_dec_out, (2 + self.sdim) * num_mix,
                                                              scope='predicted_mixtures', activation_fn=None)
        mixture_activations = tf.reshape(mixture_activations, shape=(-1, num_mix, 2 + self.sdim))
        self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
        self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
        self.means = mixture_activations[:, :, 2:]

        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
        self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
        self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim))

    def build(self, is_Train = True):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1

        input_images = tf.reshape(self.images[:, :-1,:,:,:], shape=(in_batch * in_time, in_rows, in_cols, 3))
        output_end_effector = tf.reshape(self.gtruth_endeffector_pos[:, 1:], shape = (in_batch * in_time, self.sdim))
        #builds convolutional feature points

        conv_features = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))
  
        prev_dec_out = conv_features

        for i in range(self.conf['num_repeats']):
            #decoder cell
            with tf.variable_scope('stack_{}'.format(i)):
                dec_masked_self_attention = self._multihead_attention(prev_dec_out, prev_dec_out, causal_mask=True, is_training=is_Train)
                dec_out = self._lstmforward_layer(dec_masked_self_attention)

            prev_dec_out = dec_out

        self._build_loss(prev_dec_out, output_end_effector, in_time)

        num_mix = self.conf['MDN_loss']
        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
        self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
        self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim))

