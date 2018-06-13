from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy

class VisualMPCPolicy(Policy):
    def __init__(self, action_proposal_conf, agent_params, policyparams,predictor=None, goal_image_warper=None):
        self._agent_params = agent_params
        self._policy_params = policyparams

        if 'use_server' in policyparams:
            raise NotImplementedError("Need to implement server code")
        else:
            self._predictor = predictor
            self._goal_warper = goal_image_warper
            self._cem_controller = None

    def act(self, traj, t, desig_pix=None, goal_pix = None, goal_image = None):
        if t == 0:
            self._cem_controller = self._policy_params['cem_type'](self._agent_params, self._policy_params,
                                                                     self._predictor, self._goal_warper)
            self._desig_pix = desig_pix
            self._goal_pix = goal_pix
            self._goal_image = goal_image

        action, plan_stats = self._cem_controller.act(traj, t, self._desig_pix, self._goal_pix, self._goal_image)

        return action


