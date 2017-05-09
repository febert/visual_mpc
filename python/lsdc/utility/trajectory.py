""" This file defines the sample class. """
import numpy as np


class Trajectory(object):
    def __init__(self, hyperparams):

        self.T = hyperparams['T']

        self._sample_images = np.zeros((self.T,
                                        hyperparams['image_height'],
                                        hyperparams['image_width'],
                                        hyperparams['image_channels']), dtype='uint8')

        # for storing the terminal predicted images of the K best actions at each time step:
        self.final_predicted_images = []
        self.predicted_images = None
        self.gtruth_images = None

        if 'actiondim' in hyperparams:
            self.U = np.empty([self.T, hyperparams['actiondim']])
        else:
            self.U = np.empty([self.T, 2])

        self.X_full = np.empty([self.T, 2])
        self.Xdot_full = np.empty([self.T, 2])
        self.Object_pos = np.empty((self.T, hyperparams['num_objects'], 3))
        self.X_Xdot_full = np.empty([self.T, 4])

        self.desig_pos = np.empty([self.T, 2])
        self.score = np.empty([self.T])

        self.touchdata = np.zeros([self.T, 20])
