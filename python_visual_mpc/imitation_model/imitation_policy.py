from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
import imp
import tensorflow as tf
import os
import numpy as np
import cv2
class ImitationPolicy(Policy):
    def __init__(self, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.policyparams = policyparams

        hyperparams = imp.load_source('hyperparams', self.policyparams['net_config'])
        self.net_config = hyperparams.configuration

        self.policyparams['pretrained'] = os.path.join(self.net_config['model_dir'], self.policyparams['pretrained'])

        self.img_height, self.img_width = self.agentparams['image_height'], self.agentparams['image_width']
        self.adim, self.sdim = self.net_config['adim'], self.net_config['sdim']

        self.images_pl = tf.placeholder(tf.float32, [1, None, self.img_height, self.img_width, 3])
        actions = tf.placeholder(tf.float32, [1, None, self.adim])
        self.end_effector_pos_pl = tf.placeholder(tf.float32, [1, None, self.sdim])

        with tf.variable_scope('model', reuse=None) as training_scope:
            model = self.net_config['model'](self.net_config, self.images_pl, actions, self.end_effector_pos_pl)
            self.mdn_mix, self.mdn_std_dev, self.mdn_means = model.build(is_Test = True)

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(vars, max_to_keep=0)

        self.sess = tf.Session()
        tf.train.start_queue_runners(self.sess)
        self.sess.run(tf.global_variables_initializer())

        saver.restore(self.sess, self.policyparams['pretrained'])

    def gen_mix_samples(self, means, std_dev, mix_params):
        N = self.policyparams['N_GEN']
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

    def act(self, traj, t, init_model=None):
        # if t == 0:
        #     self.targetxy = traj.Object_pose[t, 0, :2]
        #     actions = np.zeros(5)
        #
        #     delta = self.targetxy - traj.target_qpos[t, :2]
        #     norm = np.sqrt(np.sum(np.square(delta)))
        #     new_norm = min(norm, 1.0)
        #
        #     actions[:2] = traj.target_qpos[t, :2] + delta / norm * new_norm
        #     actions[2] = 0.13
        #     actions[-1] = -100
        #     return actions - traj.target_qpos[t, :] * traj.mask_rel

        sample_images = traj._sample_images[:t + 1].reshape((1, -1, self.img_height, self.img_width, 3)).astype(np.float32)
        sample_images /= 256.
        sample_eep = traj.target_qpos[:t + 1, :].reshape((1, -1, self.sdim)).astype(np.float32)
        #sample_eep = traj.X_Xdot_full[t, :].reshape((1, -1, self.sdim)).astype(np.float32)

        f_dict = {self.images_pl: sample_images, self.end_effector_pos_pl : sample_eep}
        mdn_mix, mdn_std_dev, mdn_means = self.sess.run([self.mdn_mix, self.mdn_std_dev, self.mdn_means], feed_dict=f_dict)

        samps = self.gen_mix_samples(mdn_means[-1, t], mdn_std_dev[-1, t], mdn_mix[-1, t])
        print 'cur qpos', traj.target_qpos[t, :]
        print 'top samp', samps[0][0], 'with likelihood', samps[0][1]

        # actions = np.zeros(5)
        # actions[:4] = samps[0][0][:4]
        # actions[-1] = -100
        # return actions - traj.target_qpos[t, :] * traj.mask_rel

        actions = samps[0][0].astype(np.float64)
        if actions[-1] > 0.05:
            actions[-1] = 21
        else:
            actions[-1] = -100

        return actions - traj.target_qpos[t, :] * traj.mask_rel