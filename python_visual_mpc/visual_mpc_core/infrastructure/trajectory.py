""" This file defines the sample class. """
import numpy as np

class Trajectory(object):
    def __init__(self, agentparams, netconf = None):

        self.T = agentparams['T']

        if netconf != None:
            if 'single_view' in netconf:
                img_channels = 3
            else:
                img_channels = 6
        else:
            img_channels = 3

        self._sample_images = np.zeros((self.T,
                                        netconf['img_height'],
                                        netconf['img_width'],
                                        img_channels), dtype='uint8')

        # for storing the terminal predicted images of the K best actions at each time step:
        self.final_predicted_images = []
        self.predicted_images = None
        self.gtruth_images = None

        if 'adim' in agentparams:
            self.U = np.empty([self.T, agentparams['adim']])
        else:
            self.U = np.empty([self.T, 2])

        if 'sdim' in agentparams:
            state_dim = agentparams['sdim']
        else:
            state_dim = 2

        self.X_full = np.empty([self.T, state_dim])
        self.Xdot_full = np.empty([self.T, state_dim])
        self.X_Xdot_full = np.empty([self.T, 2*state_dim])

        self.desig_pos = np.empty([self.T, 2])
        self.score = np.empty([self.T])

