""" This file defines the sample class. """
import numpy as np


class Trajectory(object):
    def __init__(self, hyperparams, netconf = None):

        self.T = hyperparams['T']

        if netconf != None:
            if 'single_view' in netconf:
                img_channels = 3
            else:
                img_channels = 6
        else:
            img_channels = 3

        self._sample_images = np.zeros((self.T,
                                        hyperparams['image_height'],
                                        hyperparams['image_width'],
                                        img_channels), dtype='uint8')

        if 'large_images_retina' in hyperparams:
            self.large_images_retina = np.zeros((self.T,
                                            hyperparams['large_images_retina'],
                                            hyperparams['large_images_retina'],
                                            img_channels), dtype='uint8')

            self.initial_ret_pos = np.zeros(2, dtype=np.int64)

        # for storing the terminal predicted images of the K best actions at each time step:
        self.final_predicted_images = []
        self.predicted_images = None
        self.gtruth_images = None

        if 'action_dim' in hyperparams:
            self.U = np.empty([self.T, hyperparams['action_dim']])
        else:
            self.U = np.empty([self.T, 2])

        if 'state_dim' in hyperparams:
            state_dim = hyperparams['state_dim']
        else:
            state_dim = 2

        self.X_full = np.empty([self.T, state_dim])
        self.Xdot_full = np.empty([self.T, state_dim])
        self.Object_pose = np.empty((self.T, hyperparams['num_objects'], 3))

        self.max_move_pose = np.empty((self.T, 3))

        self.X_Xdot_full = np.empty([self.T, 2*state_dim])

        self.desig_pos = np.empty([self.T, 2])
        self.score = np.empty([self.T])

        self.touchdata = np.zeros([self.T, 20])


