import tensorflow as tf
from  python_visual_mpc.visual_mpc_core.algorithm.cem_controller_goalimage_sawyer import CEM_controller
import numpy as np


def oldmpc2imitation_conv(mpc_state):
    open_state = np.zeros((1, 5))
    open_state[0, :4] = mpc_state[:4]
    if mpc_state[-1] > 0.05:
        open_state[0, -1] = 0.1
    else:
        open_state[0, -1] = 0
    return open_state.reshape((1, 1, -1))

class Openloop_CEM_controller(CEM_controller):
    def __init__(self, ag_params, policyparams, predictor, openloop_policy):
        super().__init__(ag_params, policyparams, predictor)
        self.openloop_policy = openloop_policy

    def sample_actions(self, last_frames, last_states):
        """
        This function is meant to allow openloop predictors to be
        plugged into CEM loop. Unfortunately, it has not yet been tested.
        """
        input_frames = (last_frames * 255).astype(np.uint8).reshape((1, 1, -1))
        input_state = conf['openloop_conv_state'](last_states)
        actions = openloop_predictor(sel_1, input_state, self.M)

        return actions
