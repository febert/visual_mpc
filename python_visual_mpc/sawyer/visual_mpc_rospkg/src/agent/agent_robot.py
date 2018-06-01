import numpy as np
import rospy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_wsg_controller import WSGRobotController
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_dualcam_recorder import RobotDualCamRecorder, Trajectory

from python_visual_mpc.visual_mpc_core.agent.agent_mjc import  get_target_qpos

class AgentSawyer:
    def __init__(self, agent_params):
        self._hyperparams = agent_params

        # initializes node and creates interface with Sawyer
        self._controller = WSGRobotController()
        self._recorder = RobotDualCamRecorder(agent_params, self._controller)

        self.control_rate = rospy.Rate(agent_params['control_rate'])

    def sample(self, policy):
        traj_ok = False
        max_tries = self._hyperparams.get('max_tries', 100)
        cntr = 0

        while not traj_ok and cntr < max_tries:
            traj, traj_ok = self.rollout(policy)

    def rollout(self, policy):
        traj = Trajectory(self._hyperparams)
        traj_ok = True

        for t in xrange(self._hyperparams['T']):
            if not self._recorder.store_recordings(traj, t):
                traj_ok = False
                break

            mj_U = policy.act(traj, t)

            for _ in xrange(self._hyperparams['substeps']):
                self.control_rate.sleep()

        return traj, traj_ok
