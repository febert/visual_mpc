from .base_model import BaseGoalClassifier
import tensorflow as tf


class UnconditionedGoalClassifier(BaseGoalClassifier):
    def __init__(self, conf, datasets=None):
        self._train_cond = tf.placeholder(tf.int32, shape=[], name="train_cond")
        self._hp = self._default_hparams().override_from_dict(conf)

        if datasets is not None:
            self._input_im = tf.cast(self._get_trainval_batch('images', datasets), tf.float32)
            self._label = self._get_trainval_batch('label', datasets)
        else:
            raise NotImplementedError("Test functionality not implemented")
        print('Checkpoint at: {} \n'.format(self._hp.save_dir))
        self._created_var_scopes = set()

    def _default_hparams(self):
        params = super(UnconditionedGoalClassifier, self)._default_hparams()
        params.add_hparam('ncam', 2)
        return params

    def build(self, global_step):
        # _conv_layer(input, n_out, name)
        with tf.variable_scope("goal_classifier") as classifier_scope:
            vgg_feats = [self._vgg_layer(self._input_im[:, i]) for i in range(self._hp.ncam)]
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