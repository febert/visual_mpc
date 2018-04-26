import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.contrib.slim as slim
import numpy as np
import os



NUMERICAL_EPS = 1e-7


def gen_mix_samples(N, means, std_dev, mix_params):

    dist_choice = np.random.choice(mix_params.shape[0], size=N, p=mix_params)
    samps = []
    for i in range(N):
        dist_mean = means[dist_choice[i]]
        out_dim = dist_mean.shape[0]
        dist_std = std_dev[dist_choice[i]]
        samp = np.random.multivariate_normal(dist_mean, dist_std * dist_std * np.eye(out_dim))

        samp_l = np.exp(-0.5 * np.sum(np.square(samp - means), axis=1) / np.square(dist_std))
        samp_l /= np.power(2 * np.pi, out_dim / 2.) * dist_std
        samp_l *= mix_params

        samps.append((samp, np.sum(samp_l)))
    return sorted(samps, key=lambda x: -x[1])

class ImitationBaseModel:
    def __init__(self, conf, images, actions, end_effector, goal_image = None):
        self.input_images, self.input_actions, self.input_end_effector, self.goal_image = images, actions, end_effector, goal_image
        self.conf = conf
        assert ('adim' in self.conf), 'must specify action dimension in conf wiht key adim'
        assert (self.conf['adim'] == actions.get_shape()[2]), 'conf adim does not match input actions'
        assert ('sdim' in self.conf), 'must specify state dimension in conf with key sdim'
        assert (self.conf['sdim'] == end_effector.get_shape()[2]), 'conf sdim does not match input end_effector'
        self.sdim, self.adim = self.conf['sdim'], self.conf['adim']

        vgg19_path = './'
        if 'vgg19_path' in conf:
            vgg19_path = conf['vgg19_path']

        # for vgg layer
        self.vgg_dict = np.load(os.path.join(vgg19_path, "vgg19.npy"), encoding='latin1').item()

        #input images
        self.images = images

        raw_input_splits = tf.split(actions, self.adim, axis=-1)
        raw_input_splits[-1] = tf.clip_by_value(raw_input_splits[-1], 0, 0.1)
        self.gtruth_actions = tf.concat(raw_input_splits, axis=-1)

        # ground truth ep
        self.gtruth_endeffector_pos = end_effector

    def _build_conv_layers(self, input_images):
        layer1 = tf_layers.layer_norm(self.vgg_layer(input_images[:, :, :, ::-1]), scope='conv1_norm')
        layer2 = tf_layers.layer_norm(
            slim.layers.conv2d(layer1, 32, [3, 3], stride=2, scope='conv2'), scope='conv2_norm')

        layer3 = tf_layers.layer_norm(
            slim.layers.conv2d(layer2, 32, [3, 3], stride=2, scope='conv3'), scope='conv3_norm')
        layer4 = tf_layers.layer_norm(
            slim.layers.conv2d(layer3, 48, [3, 3], stride=1, scope='conv4'), scope='conv4_norm')
        layer5 = tf_layers.layer_norm(
            slim.layers.conv2d(layer4, 64, [3, 3], stride=1, scope='conv5'), scope='conv5_norm')

        batch_size, num_rows, num_cols, num_fp = layer5.get_shape()
        # print 'shape', layer3.get_shape
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]

        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        x_map = tf.convert_to_tensor(x_map)
        y_map = tf.convert_to_tensor(y_map)

        x_map = tf.reshape(x_map, [num_rows * num_cols])
        y_map = tf.reshape(y_map, [num_rows * num_cols])

        features = tf.reshape(tf.transpose(layer5, [0, 3, 1, 2]), [-1, num_rows * num_cols])
        softmax = tf.nn.softmax(features)

        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
        return tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])

    def _build_loss(self, last_fc, input_action, T):
        _, input_dim = input_action.get_shape()
        input_dim = int(input_dim)

        self.debug_input = input_action

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            mixture_activations = slim.layers.fully_connected(last_fc, (input_dim + 2) * num_mix,
                                                              scope='predicted_mixtures', activation_fn=None)
            mixture_activations = tf.reshape(mixture_activations, shape=(-1, num_mix, input_dim + 2))
            self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
            self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
            self.means = mixture_activations[:, :, 2:]

            if 'round_down' in self.conf:
                thresh_mask = tf.abs(self.means) > self.conf['round_down']
                self.means = tf.cast(thresh_mask, self.means.dtype) * self.means

            gtruth_mean_sub = tf.reduce_sum(
                tf.square(self.means - tf.reshape(input_action, shape=(-1, 1, input_dim))), axis=-1)

            self.likelihoods = tf.exp(-0.5 * gtruth_mean_sub / tf.square(self.std_dev)) / self.std_dev / np.power(
                2 * np.pi, input_dim / 2.) * self.mixing_parameters
            self.lg_likelihoods = tf.log(tf.reduce_sum(self.likelihoods, axis=-1) + NUMERICAL_EPS)
            self.MDN_log_l = tf.reduce_sum(tf.log(tf.reduce_sum(self.likelihoods, axis=-1) + NUMERICAL_EPS)) \
                            / self.conf['batch_size'] / float(int(T))
            self.loss = - self.MDN_log_l

            mix_mean = tf.reduce_sum(self.means * tf.reshape(self.mixing_parameters, shape=(-1, num_mix, 1)), axis=1)

            self.diagnostic_l2loss = tf.reduce_sum(tf.square(input_action - mix_mean)) / self.conf['batch_size']

        else:
            self.predicted_actions = slim.layers.fully_connected(last_fc, input_dim, scope='predicted_actions',
                                                                 activation_fn=None)
            if 'round_down' in self.conf:
                thresh_mask = tf.abs(self.predicted_actions) > self.conf['round_down']
                self.predicted_actions = tf.cast(thresh_mask, self.predicted_actions.dtype) * self.predicted_actions

            total_loss = tf.reduce_sum(tf.square(input_action - self.predicted_actions)) \
                         + 0.5 * tf.reduce_sum(tf.abs(input_action - self.predicted_actions))
            self.action_loss = total_loss / float(self.conf['batch_size'] * int(T))
            self.loss = self.action_loss


    def build(self):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, tf_layers.layer_norm]):
            in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()

            input_images = tf.reshape(self.images, shape = (in_batch * in_time, in_rows, in_cols, 3))
            input_action = tf.reshape(self.gtruth_actions, shape = (in_batch * in_time, self.adim))
            input_end_effector = tf.reshape(self.gtruth_endeffector_pos, shape=(in_batch * in_time, self.sdim))

            fp_flat = self._build_conv_layers(input_images)

            conv_out = tf.concat([fp_flat,
                                  input_end_effector],
                                 1)

            last_fc = slim.layers.fully_connected(conv_out, 100, scope='fc1')

            self._build_loss(last_fc, input_action, in_time)

    def _dif_loss(self, pred, real, l1_scale = 0.5):
        return tf.reduce_mean(tf.square(pred - real)) + l1_scale * tf.reduce_mean(tf.abs(pred - real))
    # Source: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
    def vgg_layer(self, images):
        if images.dtype is tf.uint8:
            bgr_scaled = tf.to_float(images)
        else:
            bgr_scaled = images * 255.

        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        blue, green, red = tf.split(axis=-1, num_or_size_splits=3, value=bgr_scaled)


        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        name = "conv1_1"
        with tf.variable_scope(name):
            filt = tf.constant(self.vgg_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bgr, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(self.vgg_dict[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases)

            out = tf.nn.relu(bias)

        return out


class SimpleModel(ImitationBaseModel):
    def build(self):
        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        # get first images
        start_image = self.images[:, 0, :, :, :]
        #get final end effector
        #start_xy = self.gtruth_endeffector_pos[:, 0, :2]
        target = self.gtruth_endeffector_pos[:, 0, :4]
        #target = tf.concat([start_xy, final_conf], axis=1)

        fp_flat = self._build_conv_layers(start_image)
        layer_1 = slim.layers.fully_connected(fp_flat, 100)
        layer_2 = slim.layers.fully_connected(layer_1, 50)
        self._build_loss(layer_2, target, 1)
        self.final_frame_aux_loss = self.loss

        if 'MDN_loss' in self.conf:
            self.loss += self.diagnostic_l2loss


class ImitationLSTMModel(ImitationBaseModel):
    def build(self):

        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        #images are flattened for convolution
        input_images = tf.reshape(self.images, shape=(in_batch * in_time, in_rows, in_cols, 3))
        #actions are flattened for loss
        input_action = tf.reshape(self.gtruth_actions, shape = (in_batch * in_time, self.adim))
        #but configs are not (fed into lstm)
        input_end_effector = self.gtruth_endeffector_pos

        fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))

        #see if model can predict task from first frame
        first_frame_features = fp_flat[:, 0, :]
        real_final = tf.gather(input_end_effector[:, -1, :], [0, 1, 3], axis = 1)

        self.final_frame_state_pred = slim.layers.fully_connected(first_frame_features, 3,
                                    scope='predicted_final', activation_fn=None)

        self.final_frame_aux_loss = self._dif_loss(self.final_frame_state_pred, real_final)

        first_hidden_pred_in = tf.concat([self.final_frame_state_pred,self.gtruth_endeffector_pos[:, 0, :]], axis = -1)
        first_initial = slim.layers.fully_connected(first_hidden_pred_in, self.conf['lstm_layers'][0],
                                                    scope='first_hidden', activation_fn=tf.tanh)

        lstm_in = fp_flat
        lstm_layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(l)) for l in self.conf['lstm_layers']])

        all_initial = tuple(
            [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, self.conf['lstm_layers'][0])), first_initial)]
            + [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, l)), tf.zeros((in_batch, l)))
               for l in self.conf['lstm_layers'][1:]])

        last_fc, states = tf.nn.dynamic_rnn(cell=lstm_layers, inputs=lstm_in, initial_state=all_initial,
                                            dtype=tf.float32, parallel_iterations=int(in_batch))

        last_fc = tf.reshape(last_fc, shape=(in_batch * in_time, -1))
        self._build_loss(last_fc, input_action, in_time)

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
            self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
            self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.adim))
        else:
            self.predicted_actions = tf.reshape(self.predicted_actions, shape=(in_batch, in_time, self.adim))

        self.loss += self.final_frame_aux_loss

class ImitationLSTMModelState(ImitationBaseModel):
    def query(self, sess, traj, t, images = None, end_effector = None, actions = None):

        f_dict = {self.input_images: images, self.input_end_effector: end_effector}
        mdn_mix, mdn_std_dev, mdn_means = sess.run([self.mixing_parameters, self.std_dev, self.means],
                                                        feed_dict=f_dict)

        samps = gen_mix_samples(self.conf.get('N_GEN', 200), mdn_means[-1, t], mdn_std_dev[-1, t], mdn_mix[-1, t])
        actions = samps[0][0].astype(np.float64)
        if actions[-1] > 0.05:
            actions[-1] = 21
        else:
            actions[-1] = -100
        return actions

    def build(self, is_Test = False):
        if is_Test:
            in_batch, in_rows, in_cols = self.images.get_shape()[0], self.images.get_shape()[2], self.images.get_shape()[3]
            in_time = tf.shape(self.images)[1]

            input_images = tf.reshape(self.images, shape=[-1, in_rows, in_cols, 3])
            fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, 128))

            first_initial = slim.layers.fully_connected(self.gtruth_endeffector_pos[:, 0, :],
                                                        self.conf['lstm_layers'][0],
                                                        scope='first_hidden', activation_fn=tf.tanh)
            lstm_in = fp_flat

            lstm_layers = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(l)) for l in self.conf['lstm_layers']])

            all_initial = tuple(
                [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, self.conf['lstm_layers'][0])), first_initial)]
                + [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, l)), tf.zeros((in_batch, l)))
                   for l in self.conf['lstm_layers'][1:]])

            last_fc, states = tf.nn.dynamic_rnn(cell=lstm_layers, inputs=lstm_in, initial_state=all_initial,
                                                dtype=tf.float32, parallel_iterations=int(in_batch))

            last_fc = tf.reshape(last_fc, shape=(-1, self.conf['lstm_layers'][-1]))

            num_mix = self.conf['MDN_loss']
            mixture_activations = slim.layers.fully_connected(last_fc, (2 + self.sdim) * num_mix,
                                                              scope='predicted_mixtures', activation_fn=None)
            mixture_activations = tf.reshape(mixture_activations, shape=(-1, num_mix, 2 + self.sdim))
            self.mixing_parameters = tf.nn.softmax(mixture_activations[:, :, 0])
            self.std_dev = tf.exp(mixture_activations[:, :, 1]) + NUMERICAL_EPS
            self.means = mixture_activations[:, :, 2:]

            self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=[in_batch, in_time, num_mix])
            self.std_dev = tf.reshape(self.std_dev, shape=[in_batch, in_time, num_mix])
            self.means = tf.reshape(self.means, shape=[in_batch, in_time, num_mix, self.sdim])

            return self.mixing_parameters, self.std_dev, self.means

        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1
        #images are flattened for convolution
        input_images = tf.reshape(self.images[:, :-1,:,:,:], shape=(in_batch * in_time, in_rows, in_cols, 3))
        #actions are flattened for loss
        #but configs are not (fed into lstm)
        output_end_effector = tf.reshape(self.gtruth_endeffector_pos[:, 1:], shape = (in_batch * in_time, self.sdim))

        fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))

        #see if model can predict task from first frame
        first_initial = slim.layers.fully_connected(self.gtruth_endeffector_pos[:, 0, :], self.conf['lstm_layers'][0],
                                    scope='first_hidden', activation_fn = tf.tanh)
        lstm_in = fp_flat

        lstm_layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(l)) for l in self.conf['lstm_layers']])

        all_initial = tuple([tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, self.conf['lstm_layers'][0])), first_initial)]
                            + [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, l)), tf.zeros((in_batch, l)))
                               for l in self.conf['lstm_layers'][1:]])

        last_fc, states = tf.nn.dynamic_rnn(cell = lstm_layers, inputs = lstm_in, initial_state=all_initial,
                                            dtype = tf.float32, parallel_iterations=int(in_batch))
        last_fc = tf.reshape(last_fc, shape=(in_batch * in_time, -1))

        self._build_loss(last_fc, output_end_effector, in_time)

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
            self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
            self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim))
        else:
            self.predicted_actions = tf.reshape(self.predicted_actions, shape=(in_batch, in_time, self.sdim))


        #self.loss += 0.1 * self.diagnostic_l2loss
class ImitationLSTMModelStateLegacy(ImitationBaseModel):

    def build(self):

        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1
        #images are flattened for convolution
        input_images = tf.reshape(self.images[:, :-1,:,:,:], shape=(in_batch * in_time, in_rows, in_cols, 3))
        #actions are flattened for loss
        #but configs are not (fed into lstm)
        output_end_effector = tf.reshape(self.gtruth_endeffector_pos[:, 1:, :self.sdim/2], shape = (in_batch * in_time, self.sdim/2))

        fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))

        #see if model can predict task from first frame
        first_frame_features = fp_flat[:, 0, :]
        self.final_frame_state_pred = slim.layers.fully_connected(first_frame_features, self.sdim / 2,
                                    scope='predicted_final', activation_fn=None)
        raw_final_loss = tf.reduce_sum(tf.square(self.final_frame_state_pred - self.gtruth_endeffector_pos[:, -1, :self.sdim/2])) \
                         + 0.5 * tf.reduce_sum(tf.abs(self.final_frame_state_pred - self.gtruth_endeffector_pos[:, -1, :self.sdim/2]))
        self.final_frame_aux_loss = raw_final_loss / float(self.conf['batch_size'])

        final_pred_broadcast = tf.tile(tf.reshape(tf.stop_gradient(self.final_frame_state_pred),
                                                  shape = (in_batch, 1, self.sdim /2)),
                                       [1,in_time, 1])

        lstm_in = tf.concat([fp_flat, final_pred_broadcast],-1)

        lstm_layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(l) for l in self.conf['lstm_layers']])

        last_fc, states = tf.nn.dynamic_rnn(cell = lstm_layers, inputs = lstm_in,
                                            dtype = tf.float32, parallel_iterations=int(in_batch))
        last_fc = tf.reshape(last_fc, shape=(in_batch * in_time, -1))

        self._build_loss(last_fc, output_end_effector, in_time)

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
            self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
            self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim / 2))
        else:
            self.predicted_actions = tf.reshape(self.predicted_actions, shape=(in_batch, in_time, self.sdim / 2))

        self.loss += self.final_frame_aux_loss

class ImitationLSTMVAEAction(ImitationBaseModel):
    def query(self, sess, traj, t, images = None, end_effector = None, actions = None):
        if t == 0:
            self.latent_vec = np.random.normal(size = (1, self.conf['latent_dim']))

        f_dict = {self.input_images: images[:, :, :, :, ::-1], self.latent_sample_pl: self.latent_vec}
        pred_deltas = sess.run(self.predicted_actions, feed_dict=f_dict)
        #print(pred_deltas[0, -1])
        actions = (pred_deltas[0, -1, :5]  + traj.target_qpos[t, :]) * traj.mask_rel
        if pred_deltas[0, -1, 4] >= 0.03:
            print('close')
            actions[-1] = 21
        else:
            print('open')
            actions[-1] = -100
        return actions

    def build(self, is_Test = False):
        latent_dim = self.conf['latent_dim']

        if is_Test:
            in_batch, in_rows, in_cols = self.images.get_shape()[0], self.images.get_shape()[2], \
                                         self.images.get_shape()[3]
            in_time = tf.shape(self.images)[1]
            input_images = tf.reshape(self.images[:, :, :, :, :], shape=(-1, in_rows, in_cols, 3))

            fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, 128))
            lstm_in = slim.layers.fully_connected(fp_flat, latent_dim,
                                                  scope='lstm_in', activation_fn=None)

            latent_state = slim.layers.fully_connected(fp_flat[:, 0, :], 2 * latent_dim,
                                                       scope='latent_state', activation_fn=tf.tanh)

            self.latent_mean, latent_std_logits = tf.split(latent_state, 2, axis=-1)
            self.latent_std = tf.exp(latent_std_logits)
            self.latent_sample_pl = tf.placeholder(tf.float32, [1, latent_dim])
            latent_sample = self.latent_mean + self.latent_std * self.latent_sample_pl

            lstm_layers = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(l)) for l in self.conf['lstm_layers']])
            all_initial = tuple(
                [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, self.conf['lstm_layers'][0])), latent_sample)]
                + [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, l)), tf.zeros((in_batch, l)))
                   for l in self.conf['lstm_layers'][1:]])
            last_fc, states = tf.nn.dynamic_rnn(cell=lstm_layers, inputs=lstm_in, initial_state=all_initial,
                                                dtype=tf.float32, parallel_iterations=int(in_batch))

            self.predicted_actions = slim.layers.fully_connected(last_fc, self.sdim,
                                                                 scope='action_predictions', activation_fn=None)


            return self.predicted_actions

        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        # in_time -= 1
        #images are flattened for batch convolution
        input_images = tf.reshape(self.images[:, :,:,:,:], shape=(in_batch * in_time, in_rows, in_cols, 3))

        input_actions = self.gtruth_actions[:, :, :]
        # first_rel_state = tf.reshape(self.gtruth_endeffector_pos[:, 0, :4], (in_batch, 1, -1))
        # next_rel_states = self.gtruth_endeffector_pos[:, 1:, :4]
        # gripper_mask = tf.concat(
        #                [tf.cast(tf.expand_dims(self.gtruth_endeffector_pos[:, :-1, -1] <= 0.01, -1), tf.float32),
        #                tf.cast(tf.expand_dims(self.gtruth_endeffector_pos[:, :-1, -1] > 0.01, -1), tf.float32)],
        #                -1
        #                )

        fp_flat = tf.reshape(self._build_conv_layers(input_images), shape=(in_batch, in_time, -1))
        lstm_in = slim.layers.fully_connected(fp_flat, latent_dim,
                                                   scope='lstm_in', activation_fn=None)

        latent_state = slim.layers.fully_connected(fp_flat[:, 0, :], 2 * latent_dim,
                                    scope='latent_state', activation_fn=tf.tanh)

        self.latent_mean, latent_std_logits = tf.split(latent_state, 2, axis = -1)
        self.latent_std = tf.exp(latent_std_logits)
        latent_sample = self.latent_mean + self.latent_std * tf.random_normal(tf.shape(self.latent_mean))


        lstm_layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(l)) for l in self.conf['lstm_layers']])
        all_initial = tuple(
            [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, self.conf['lstm_layers'][0])), latent_sample)]
            + [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, l)), tf.zeros((in_batch, l)))
               for l in self.conf['lstm_layers'][1:]])
        last_fc, states = tf.nn.dynamic_rnn(cell=lstm_layers, inputs=lstm_in, initial_state=all_initial,
                                            dtype=tf.float32, parallel_iterations=int(in_batch))

        self.predicted_actions = slim.layers.fully_connected(last_fc, self.sdim,
                                    scope='action_predictions', activation_fn=None)

        # self.predicted_rel_states = tf.cumsum(self.predicted_actions[:, :, :4], axis = 1) + first_rel_state
        # self.predicted_gripper_states = self.predicted_actions[:, :, -2:]

        latent_var = tf.square(self.latent_std)
        self.latent_loss = 0.5 * tf.reduce_mean(tf.square(self.latent_mean) + latent_var - tf.log(latent_var) - 1)

        # self.action_loss = 2 * self._dif_loss(self.predicted_actions[:, :, :4], input_actions[:, :, :4]) + \
        #                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = gripper_mask,
        #                                                                    logits=self.predicted_gripper_states))
        self.action_loss = 2 * self._dif_loss(self.predicted_actions, input_actions)

        self.loss = self.latent_loss + self.action_loss

        self.final_frame_aux_loss = self.latent_loss

class ImitationLSTMModelStateGoalImage(ImitationBaseModel):

    def build(self, is_Test = False):

        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1
        #images are flattened for convolution
        goal_images = tf.reshape(self.goal_image, (in_batch, 1, in_rows, in_cols, 3))
        conv_input_images = tf.reshape(tf.concat((self.images[:, :-1,:,:,:], goal_images),1) , \
                                        shape=(in_batch * (in_time + 1), in_rows, in_cols, 3))
        #actions are flattened for loss
        #but configs are not (fed into lstm)
        output_end_effector = tf.reshape(self.gtruth_endeffector_pos[:, 1:], shape = (in_batch * in_time, self.sdim))

        fp_flat = tf.reshape(self._build_conv_layers(conv_input_images), shape=(in_batch, in_time + 1, -1))

        #see if model can predict task from first frame
        goal_features = fp_flat[:, -1, :]
        first_initial = slim.layers.fully_connected(goal_features, self.conf['lstm_layers'][0],
                                    scope='first_hidden', activation_fn = tf.tanh)
        lstm_in = fp_flat[:, :-1, :]

        lstm_layers = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(l)) for l in self.conf['lstm_layers']])

        all_initial = tuple([tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, self.conf['lstm_layers'][0])), first_initial)]
                            + [tf.contrib.rnn.LSTMStateTuple(tf.zeros((in_batch, l)), tf.zeros((in_batch, l)))
                               for l in self.conf['lstm_layers'][1:]])

        last_fc, states = tf.nn.dynamic_rnn(cell = lstm_layers, inputs = lstm_in, initial_state=all_initial,
                                            dtype = tf.float32, parallel_iterations=int(in_batch))
        last_fc = tf.reshape(last_fc, shape=(in_batch * in_time, -1))

        self._build_loss(last_fc, output_end_effector, in_time)

        if 'MDN_loss' in self.conf:
            num_mix = self.conf['MDN_loss']
            self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
            self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
            self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim))
        else:
            self.predicted_actions = tf.reshape(self.predicted_actions, shape=(in_batch, in_time, self.sdim))