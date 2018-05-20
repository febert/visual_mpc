from python_visual_mpc.imitation_model.attention_models.base_model import BaseAttentionModel
import tensorflow as tf

class AttentionGoalImage(BaseAttentionModel):
    def build(self, is_Train = True):
        assert 'MDN_loss' in self.conf, "MODEL ONLY SUPPORTS MDN LOSS"
        assert self.goal_image is not None, "MODEL REQUIRES INPUT GOALIMAGE"
        in_batch, in_time, in_rows, in_cols, _ = self.images.get_shape()
        in_time -= 1
        
        input_goal_image = tf.reshape(self.goal_image, shape=(in_batch, 1, in_rows, in_cols, 3))
        input_images = tf.concat([input_goal_image, self.images[:, :-1]], 1)
        
        output_end_effector = tf.reshape(self.gtruth_endeffector_pos[:, 1:], shape = (in_batch * in_time, self.sdim))
        
        #builds convolutional feature points
        conv_in = tf.reshape(input_images, shape=(-1, in_rows, in_cols, 3))
        conv_features = tf.reshape(self._build_conv_layers(conv_in), shape=(in_batch, in_time + 1, -1))
  
        prev_dec_out = conv_features[:, 1:]
        prev_enc_out = tf.reshape(conv_features[:, 0], shape=(in_batch, 1, -1))

        for i in range(self.conf['num_repeats']):
            #decoder cell
            with tf.variable_scope('stack_{}'.format(i)):
                enc_out = self._feedforward_layer(prev_enc_out, is_training = is_Train)

                #dec_masked_self_attention = self._multihead_attention(prev_dec_out, prev_dec_out, causal_mask = True, is_training = is_Train)
                dec_enc_attention = self._multihead_attention(prev_dec_out, enc_out, is_training = is_Train)
                dec_out = self._lstmforward_layer(dec_enc_attention)

            prev_dec_out = dec_out
            prev_enc_out = enc_out

        self._build_loss(prev_dec_out, output_end_effector, in_time)

        num_mix = self.conf['MDN_loss']
        self.mixing_parameters = tf.reshape(self.mixing_parameters, shape=(in_batch, in_time, num_mix))
        self.std_dev = tf.reshape(self.std_dev, shape=(in_batch, in_time, num_mix))
        self.means = tf.reshape(self.means, shape=(in_batch, in_time, num_mix, self.sdim))
        
        self.loss += 0.1 * self.diagnostic_l2loss
