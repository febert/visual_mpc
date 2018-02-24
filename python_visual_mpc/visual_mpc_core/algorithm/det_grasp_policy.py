import numpy as np

from python_visual_mpc.visual_mpc_core.algorithm.policy import Policy

class DeterministicGraspPolicy(Policy):
    def __init__(self, agentparams, policyparams):
        Policy.__init__(self)
        self.agentparams = agentparams
        self.policyparams = policyparams
        self.adim = agentparams['adim']

        assert self.adim >= 4, 'must have at least x, y, z + gripper actions'

        self.moveto = True
        self.drop = False
        self.graspTime = 0
        self.grasp = False
        self.lift = False

    def act(self, traj, t, init_model = None):
        if t == 0:
            self.moveto = True
            self.drop = False
            self.lift = False
            self.grasp = False
            self.graspTime = 0

        if self.grasp and self.graspTime > 10:
            print 'lifting at time', t, '!', 'have z', traj.X_full[t, 2]
            self.grasp = False
            self.lift = True

        if self.drop and traj.X_full[t, 2] <= -0.079:
            print 'grasping at time', t, '!', 'have z', traj.X_full[t, 2]
            self.drop = False
            self.grasp = True

        if self.moveto and np.linalg.norm(traj.Object_pose[t, 0, :2] - traj.X_full[t, :2], 2) <= self.agentparams['drop_thresh']:
            print 'swapping at time', t, '!'
            self.moveto = False
            self.drop = True

        if self.moveto:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = self.agentparams['ztarget']
            actions[3] = -100
            return actions, None

        elif self.drop:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = -0.08
            actions[3] = -100
            return actions, None
        elif self.lift:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = self.agentparams['ztarget']
            actions[3] = 21

            return actions, None
        elif self.grasp:
            actions = np.zeros(self.adim)
            actions[:2] = traj.Object_pose[t, 0, :2]
            actions[2] = -0.08
            actions[3] = 5. / 10 * self.graspTime
            self.graspTime += 1
            return actions, None
