""" This file defines the linear Gaussian policy class. """
import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy

class Pos_Controller(Policy):
    """
    PD-Position control
    """
    def __init__(self, agent_params, policy_params):
        Policy.__init__(self)
        self.x_g = np.array([0,0])
        self.policy_params = policy_params
        self.target = np.empty(2)
        self.inc = np.empty(2)
        # print 'init low level ctrl'

    def act(self, X, Xdot, sample_images, t, target = None):
        """
        Steer the ball to a particualr target
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            x_data_idx: data_files indexes for x
            target: the target point for the control,
            if left empty assuming data collection run

        """

        if self.policy_params['mode']== 'relative':

            if self.policy_params['randomtargets']:
                # set target position to initial pos at startup:
                if t == 0:
                    self.target[:] = X

                assert target == None
                new_point_freq = self.policy_params['repeats']
                if t % new_point_freq ==0:
                    #set new target point
                    cov = np.diag(np.ones(2) * self.policy_params['std_dev']**2)
                    self.inc = np.random.multivariate_normal([0, 0], cov)
                    self.target += self.inc
            else:
                self.target = target

        if self.policy_params['mode'] == 'absolute':
            new_point_freq = self.policy_params['repeats']
            if t % new_point_freq ==0:
                #set new target point
                self.target = np.random.uniform(-0.45, 0.45, 2)

        # stiffness = 50
        stiffness = 100
        damping = 20

        force = (self.target - X) * stiffness - Xdot * damping

        # print 't:',t, 'current pos:', X,  'target:', self.target
        # print 'pos error norm: ', np.linalg.norm(self.target - X)

        if target == None:
            return force, self.inc
        else:
            return force