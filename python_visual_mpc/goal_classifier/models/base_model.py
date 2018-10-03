import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import numpy as np


class BaseGoalClassifier:
    def __init__(self, conf, datasets=None):
        self._train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self._hp = self._default_hparams().override_from_dict(conf)

        if datasets is not None:
            self._goal_images = tf.cast(self._get_trainval_batch('goal_image', datasets), tf.float32)
            self._input_im = tf.cast(self._get_trainval_batch('final_frame', datasets), tf.float32)
            self._label = self._get_trainval_batch('label', datasets)
        else:
            raise NotImplementedError("Test functionality not implemented")
        print('Checkpoint at: {} \n'.format(self._hp.save_dir))

    def _get_trainval_batch(self, key, datasets):
        train_data = tf.concat([d[key] for d in datasets], 0)
        val_data = tf.concat([d[key, 'val'] for d in datasets], 0)
        return tf.cond(self._train_cond > 0, lambda: train_data, lambda: val_data)

    def _default_hparams(self):
        params = {
            'pretrained_path': '{}/weights'.format(os.environ['VMPC_DATA_DIR']),
            'max_steps': 10000,
            'save_dir': './base_model/',
            'conv_kernel': 3,
            'conv_layers': [48, 64, 64, 64],
            'pool_layers': [ 2, 1, 1, 1],
            'fc_layers': [2048],
            'lr': 1e-3,
            'build_loss': True
        }
        return HParams(**params)

    def build(self, global_step):
        with tf.variable_scope("goal_classifier") as classifier_scope:
            bottom_goal_feat, bottom_input_feat = self._vgg_layer(self._goal_images), self._vgg_layer(self._input_im)
            last_conv = bottom_goal_feat.get_shape().as_list()[-1]
            for i, n_out in enumerate(self._hp.conv_layers):
                top_goal_feat = tf.contrib.layers.conv2d(bottom_goal_feat, n_out, self._hp.conv_kernel,
                                                         scope="tr_conv{}".format(i))
                top_input_feat = tf.contrib.layers.conv2d(bottom_input_feat, n_out, self._hp.conv_kernel,
                                                         scope="tr_conv{}".format(i), reuse=True)

                if self._hp.pool_layers[i] > 1:
                    k_size = self._hp.pool_layers[i]
                    bottom_goal_feat = tf.nn.max_pool(top_goal_feat, ksize=[1, k_size, k_size, 1],
                                                      strides=[1, k_size, k_size, 1], padding='SAME')
                    bottom_input_feat = tf.nn.max_pool(top_input_feat, ksize=[1, k_size, k_size, 1],
                                                       strides=[1, k_size, k_size, 1], padding='SAME')
                elif last_conv == n_out:
                    bottom_input_feat = bottom_input_feat + top_input_feat
                    bottom_goal_feat = bottom_goal_feat + top_goal_feat
                else:
                    bottom_goal_feat, bottom_input_feat = top_goal_feat, top_input_feat
                last_conv = n_out

            goal_features = tf.reshape(bottom_goal_feat, [-1, np.prod(bottom_goal_feat.get_shape().as_list()[1:])])
            input_features = tf.reshape(bottom_input_feat, [-1,np.prod(bottom_input_feat.get_shape().as_list()[1:])])
            fc_features = tf.concat([goal_features, input_features], -1)

            for i, n_out in enumerate(self._hp.fc_layers):
                fc_out = tf.layers.dense(fc_features, n_out, name="fc_{}".format(i),
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                fc_features = tf.layers.batch_normalization(fc_out, name="batch_norm_{}".format(i))
                print(i, fc_features)

            self._logits = tf.layers.dense(fc_features, 2)
            if self._hp.build_loss:
                self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self._label),
                                                           logits = self._logits))
                self._train_op = tf.train.AdamOptimizer(self._hp.lr).minimize(self._loss, global_step=global_step)

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=classifier_scope.name)
        self._saver = tf.train.Saver(vars, max_to_keep=0)

    def base_feed_dict(self, is_train):
        train_val = 0
        if is_train:
            train_val = 1
        return {self._train_cond: train_val}

    def _vgg_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _vgg_conv(self, vgg_dict, bottom, name):
        with tf.variable_scope(name, reuse=True):
            filt = tf.constant(vgg_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(vgg_dict[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _vgg_layer(self, images):
        """
        :param images: float32 RGB in range 0 - 255
        :return:
        """
        vgg_dict = np.load(os.path.join(self._hp.pretrained_path, "vgg19.npy"), encoding='latin1').item()
        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        red, green, blue = tf.split(axis=-1, num_or_size_splits=3, value=images)

        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        conv1_1 = self._vgg_conv(vgg_dict, bgr, "conv1_1")
        conv1_2 = self._vgg_conv(vgg_dict, conv1_1, "conv1_2")
        conv1_out = self._vgg_pool(conv1_2, "pool1")

        return conv1_out

    def step(self, sess, global_step, eval_summaries=False):
        fetches = {}
        _, fetches['train_loss'] = sess.run([self._train_op, self._loss], feed_dict={self._train_cond: 1})

        if eval_summaries:
            fetches['val_loss'], test_label, test_logit = sess.run([self._loss, self._label, self._logits], feed_dict={self._train_cond: 0})
            labels, preds = np.argmax(test_label, axis=-1), np.argmax(test_logit, axis=-1)
            fetches['val/perc_correct_summary'] = np.sum(np.abs(labels - preds).astype(np.float32) / labels.shape[0])
            fetches['train/loss_summary'], fetches['val/loss_summary'] = fetches['train_loss'], fetches['val_loss']
        return fetches

    @property
    def max_steps(self):
        return self._hp.max_steps

    @property
    def save_dir(self):
        self._create_save_dir()
        return self._hp.save_dir

    def _create_save_dir(self):
        if os.path.isfile(self._hp.save_dir):
            raise IOError('Path: {} refers to existing file'.format(self._hp.save_dir))
        if not os.path.exists(self._hp.save_dir):
            os.makedirs(self._hp.save_dir)

    def restore(self, sess, checkpoint_path):
        if checkpoint_path is None:
            if os.path.exists(self._hp.save_dir):
                raise IOError("Save Directory {} exists! (please clear)".format(self._hp.save_dir))
            return
        self._hp.save_dir = checkpoint_path
        self._saver.restore(sess, checkpoint_path)

    def save(self, sess, global_step):
        self._create_save_dir()
        self._saver.save(sess, os.path.join(self._hp.save_dir, "model"), global_step=global_step)