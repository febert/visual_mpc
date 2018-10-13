import tensorflow as tf
from tensorflow.contrib.training import HParams
import os
import numpy as np


class BaseGoalClassifier:
    def __init__(self, conf, conf_override, datasets=None):
        self._train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self._hp = self._default_hparams().override_from_dict(conf)
        self._hp.parse(conf_override)
        if datasets is not None:
            self._goal_images = tf.cast(self._get_trainval_batch('goal_image', datasets), tf.float32)
            self._input_im = tf.cast(self._get_trainval_batch('final_frame', datasets), tf.float32)
            self._label = self._get_trainval_batch('label', datasets)
        else:
            raise NotImplementedError("Test functionality not implemented")
        print('Checkpoint at: {} \n'.format(self._hp.save_dir))
        self._created_var_scopes = set()

    def _get_trainval_batch(self, key, datasets):
        train_data = tf.concat([d[key] for d in datasets], 0)
        val_data = tf.concat([d[key, 'val'] for d in datasets], 0)
        return tf.cond(self._train_cond > 0, lambda: train_data, lambda: val_data)

    def _default_hparams(self):
        params = {
            'pretrained_path': '{}/weights'.format(os.environ['VMPC_DATA_DIR']),
            'max_steps': 80000,
            'save_dir': './base_model/',
            'conv_kernel': 3,
            'conv_layers': 3,
            'num_channels': [256, 128],
            'fc_layers': [128],
            'lr': 1e-3,
            'build_loss': True
        }
        return HParams(**params)

    def build(self, global_step):
        # _conv_layer(input, n_out, name)
        with tf.variable_scope("goal_classifier") as classifier_scope:
            vgg_feats = [self._vgg_layer(self._goal_images), self._vgg_layer(self._input_im)]
            bottom_feats = self._conv_layer(vgg_feats, self._hp.num_channels[0], "vgg_to_input")

            for i in range(self._hp.conv_layers):
                # conv to inner dim
                inner_feats = self._conv_layer(bottom_feats, self._hp.num_channels[1], "inner_conv_{}".format(i))
                outer_feats = self._conv_layer(inner_feats, self._hp.num_channels[0], "outer_conv_{}".format(i))
                skip_feats = [outer_feats[j] + bottom_feats[j] for j in range(len(outer_feats))]
                bottom_feats = self._batch_norm_layer(skip_feats, "bnorm_conv_{}".format(i))

            concat_feats = tf.concat(self._spatial_softmax(bottom_feats, "spatial_softmax_0"), 1)

            for i, n_out in enumerate(self._hp.fc_layers):
                fc_out = tf.layers.dense(concat_feats, n_out, name="fc_{}".format(i), activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                concat_feats = self._batch_norm_layer(fc_out, name="fc_batch_norm_{}".format(i))

            self._logits = tf.layers.dense(concat_feats, 2, kernel_initializer=tf.contrib.layers.xavier_initializer())

            if self._hp.build_loss:
                self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self._label),
                                                           logits = self._logits))
                self._train_op = tf.train.AdamOptimizer(self._hp.lr).minimize(self._loss, global_step=global_step)

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=classifier_scope.name)
        global_step_collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global_step')
        self._saver = tf.train.Saver(vars + global_step_collection, max_to_keep=0)

    def _get_reuse_and_add(self, name):
        reuse = name in self._created_var_scopes
        if not reuse:
            self._created_var_scopes.add(name)
        return reuse

    def base_feed_dict(self, is_train):
        train_val = 0
        if is_train:
            train_val = 1
        return {self._train_cond: train_val}

    def _spatial_softmax(self, input, name="spatial_softmax_feats"):
        reuse = self._get_reuse_and_add(name)

        if isinstance(input, list):
            assert len(input) > 0, "list must have at least one tensor"
            with tf.variable_scope(name, reuse=reuse):
                first_feat = [tf.contrib.layers.spatial_softmax(input[0], name="sft")]
            later_feat = []
            for im_in in input[1:]:
                with tf.variable_scope(name, reuse=True):
                    later_feat.append(tf.contrib.layers.spatial_softmax(im_in, name="sft"))
            return first_feat + later_feat

        with tf.variable_scope(name, reuse=reuse):
            output = tf.contrib.layers.spatial_softmax(input, name="sft")
        return output

    def _batch_norm_layer(self, input, name):
        reuse = self._get_reuse_and_add(name)

        if isinstance(input, list):
            assert len(input) > 0, "list must have at least one tensor"
            first_out = [tf.layers.batch_normalization(input[0], name=name, reuse=reuse)]
            later_out = [tf.layers.batch_normalization(im_in, name=name, reuse=True) for im_in in input[1:]]
            return first_out + later_out

        return tf.layers.batch_normalization(input, name=name, reuse=reuse)

    def _conv_layer(self, input, n_out, name, kernel=None):
        if kernel is None:
            kernel = self._hp.conv_kernel
        a_fn = tf.nn.relu

        reuse = self._get_reuse_and_add(name)

        if isinstance(input, list):
            assert len(input) > 0, "size 0 array given"
            first_conv = [tf.contrib.layers.conv2d(input[0], n_out, kernel, scope=name, reuse=reuse, activation_fn=a_fn)]
            later_conv = [tf.contrib.layers.conv2d(im_in, n_out, kernel, scope=name, reuse=True, activation_fn=a_fn)
                                                                                                for im_in in input[1:]]
            return first_conv + later_conv

        return tf.contrib.layers.conv2d(input, n_out, kernel, scope=name, reuse=reuse, activation_fn=a_fn)

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

        return conv1_2

    def step(self, sess, global_step, eval_summaries=False):
        fetches = {}
        _, fetches['train_loss'] = sess.run([self._train_op, self._loss], feed_dict={self._train_cond: 1})

        if eval_summaries:
            fetches['val_loss'], test_label, test_logit = sess.run([self._loss, self._label, self._logits], feed_dict={self._train_cond: 0})
            labels, preds = np.argmax(test_label, axis=-1), np.argmax(test_logit, axis=-1)
            fetches['val/error_summary'] = np.sum(np.abs(labels - preds).astype(np.float32) / labels.shape[0])

            pred_pos, pred_neg = np.sum(preds), np.sum(1 - preds)
            fetches['val/false_positive_summary'], fetches['val/false_negative_summary'] = 0., 0.
            if pred_pos > 0:
                fetches['val/false_positive_summary'] = np.sum(preds[np.where(labels < 1)]) / float(pred_pos)
            if pred_neg > 0:
                fetches['val/false_negative_summary'] = np.sum(1 - preds[np.where(labels > 0)]) / float(pred_neg)

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
