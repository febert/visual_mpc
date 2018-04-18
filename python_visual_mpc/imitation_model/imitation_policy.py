from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy
import importlib.machinery
import importlib.util
import tensorflow as tf
import os
import numpy as np

class ImitationPolicy(Policy):
    MODEL_CREATION = False
    def __init__(self, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.policyparams = policyparams


        loader = importlib.machinery.SourceFileLoader('mod_hyper', self.policyparams['net_config'])
        spec = importlib.util.spec_from_loader(loader.name, loader)
        conf = importlib.util.module_from_spec(spec)
        loader.exec_module(conf)
        self.net_config = conf.configuration

        self.policyparams['pretrained'] = os.path.join(self.net_config['model_dir'], self.policyparams['pretrained'])

        self.img_height, self.img_width = self.agentparams['image_height'], self.agentparams['image_width']
        self.adim, self.sdim = self.net_config['adim'], self.net_config['sdim']

        self._init_net()

    def _init_net(self):
        images_pl = tf.placeholder(tf.uint8, [1, None, self.img_height, self.img_width, 3])
        actions = tf.placeholder(tf.float32, [1, None, self.adim])
        end_effector_pos_pl = tf.placeholder(tf.float32, [1, None, self.sdim])
        if ImitationPolicy.MODEL_CREATION:
            with tf.variable_scope('model', reuse=True) as training_scope:
                self.model = self.net_config['model'](self.net_config, images_pl, actions, end_effector_pos_pl)
                self.model.build(is_Test=True)
        else:
            with tf.variable_scope('model', reuse=None) as training_scope:
                self.model = self.net_config['model'](self.net_config, images_pl, actions, end_effector_pos_pl)
                self.model.build(is_Test=True)

            ImitationPolicy.MODEL_CREATION = True

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(vars, max_to_keep=0)

        self.sess = tf.Session()
        tf.train.start_queue_runners(self.sess)
        self.sess.run(tf.global_variables_initializer())

        saver.restore(self.sess, self.policyparams['pretrained'])



    def act(self, traj, t, init_model = None, goal_object_pose = None, hyperparams = None, goal_image = None):
        sample_images = traj._sample_images[:t + 1].reshape((1, -1, self.img_height, self.img_width, 3))
        sample_eep = traj.target_qpos[:t + 1, :].reshape((1, -1, self.sdim)).astype(np.float32)

        actions = self.model.query(self.sess, traj, t, images=sample_images, end_effector=sample_eep)
        return actions - traj.target_qpos[t, :] * traj.mask_rel