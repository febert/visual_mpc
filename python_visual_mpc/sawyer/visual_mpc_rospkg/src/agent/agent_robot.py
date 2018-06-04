import numpy as np
import rospy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_wsg_controller import WSGRobotController
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_dualcam_recorder import RobotDualCamRecorder, Trajectory

from python_visual_mpc.visual_mpc_core.agent.utils.target_qpos_utils import get_target_qpos
import copy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.primitives_regintervals import zangle_to_quat
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils import inverse_kinematics
from sensor_msgs.msg import JointState
class AgentSawyer:
    def __init__(self, agent_params):
        self._hyperparams = agent_params

        # initializes node and creates interface with Sawyer
        self._controller = WSGRobotController(agent_params['control_rate'])
        self._recorder = RobotDualCamRecorder(agent_params, self._controller)

        self._controller.reset_with_impedance()


    def sample(self, policy):
        traj_ok = False
        max_tries = self._hyperparams.get('max_tries', 100)
        cntr = 0
        traj = None

        while not traj_ok and cntr < max_tries:
            traj, traj_ok = self.rollout(policy)

        return traj, traj_ok

    def rollout(self, policy):
        traj = Trajectory(self._hyperparams)
        traj_ok = True

        t_down = 0
        gripper_up, gripper_closed = False, False

        self._controller.reset_with_impedance()

        for t in xrange(self._hyperparams['T']):
            if not self._recorder.store_recordings(traj, t):
                traj_ok = False
                break
            if t == 0:
                self.prev_qpos = copy.deepcopy(traj.robot_states[0])
                self.next_qpos = copy.deepcopy(traj.robot_states[0])
                traj.target_qpos[0] = copy.deepcopy(traj.robot_states[0])
            else:
                self.prev_qpos = copy.deepcopy(self.next_qpos)
                error = np.abs(self.prev_qpos[:3] - traj.robot_states[t, :3])
                print('At time {} des vs actual xyz error {}'.format(t, error))

            mj_U = policy.act(traj, t)

            self.next_qpos, t_down, gripper_up, gripper_closed = get_target_qpos(
                self.next_qpos, self._hyperparams, mj_U, t, gripper_up, gripper_closed, t_down,
                traj.robot_states[t, 2])
            traj.target_qpos[t + 1] = copy.deepcopy(self.next_qpos)

            target_pose = self.state_to_pose(self.next_qpos)

            start_joints = self._controller.limb.joint_angles()
            try:
                target_ja = inverse_kinematics.get_joint_angles(target_pose, seed_cmd=start_joints,
                                                                use_advanced_options=True)
            except ValueError:
                rospy.logerr('no inverse kinematics solution found, '
                             'going to reset robot...')
                current_joints = self._controller.limb.joint_angles()
                self._controller.limb.set_joint_positions(current_joints)
                return None, False

            self._controller.move_with_impedance_sec(target_ja)
            if self.next_qpos[-1] > 0.05:
                self._controller.close_gripper()
            else:
                self._controller.open_gripper()


        return traj, traj_ok

    def state_to_pose(self, target_state):
        quat = zangle_to_quat(target_state[3])
        desired_pose = inverse_kinematics.get_pose_stamped(target_state[0],
                                                           target_state[1],
                                                           target_state[2],
                                                           quat)
        return desired_pose



    def get_int_state(self, substep, prev, next):
        assert substep >= 0 and substep < self._hyperparams['substeps']
        return substep/float(self._hyperparams['substeps'])*(next - prev) + prev