import numpy as np
import rospy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_wsg_controller import WSGRobotController
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_dualcam_recorder import RobotDualCamRecorder, Trajectory

from python_visual_mpc.visual_mpc_core.agent.utils.target_qpos_utils import get_target_qpos
import copy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.primitives_regintervals import zangle_to_quat
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils import inverse_kinematics
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.misc.camera_calib.calibrated_camera import CalibratedCamera
class AgentSawyer:
    def __init__(self, agent_params):
        print('CREATING AGENT FOR ROBOT: {}'.format(agent_params.get('robot_name', 'vestri')))
        self._hyperparams = agent_params

        # initializes node and creates interface with Sawyer
        self._controller = WSGRobotController(agent_params['control_rate'], agent_params.get('robot_name', 'vestri'))
        self._recorder = RobotDualCamRecorder(agent_params, self._controller)

        self._controller.reset_with_impedance()

        if 'rpn_objects' in agent_params:
            self._calibrated_camera = CalibratedCamera(agent_params.get('robot_name', 'vestri'))


    def sample(self, policy, itr):
        traj_ok = False
        max_tries = self._hyperparams.get('max_tries', 100)
        cntr = 0
        traj = None

        if itr % 100 == 0 and itr > 0:
            self._controller.redistribute_objects()

        while not traj_ok and cntr < max_tries:
            traj, traj_ok = self.rollout(policy)

        return traj, traj_ok

    def rollout(self, policy):
        traj = Trajectory(self._hyperparams)
        traj_ok = True

        self.t_down = 0
        self.gripper_up, self.gripper_closed = False, False

        self._controller.reset_with_impedance()              # go to neutral

        if 'rpn_objects' in self._hyperparams:
            if not self._recorder.store_recordings(traj, 0): #grab frame of robot in neutral
                return None, False
            neutral_img = traj.raw_images[0, 0]
            _, rbt_coords = self._calibrated_camera.object_points(neutral_img)

            traj.Object_pose = np.zeros((1, len(rbt_coords), 3))
            for i, r in enumerate(rbt_coords):
                traj.Object_pose[0, i] = r

        self._controller.reset_with_impedance(angles=self.random_start_angles(), open_gripper=False,
                                              stiffness=self._hyperparams['impedance_stiffness'])

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

                diff = traj.robot_states[t, :3] - self.prev_qpos[:3]
                euc_error, abs_error = np.linalg.norm(diff), np.abs(diff)
                print("at time {}, l2 error {} and abs_dif {}".format(t, euc_error, abs_error))

            mj_U = policy.act(traj, t)

            if 'check_preplan' in self._hyperparams and t == 0:
                test_next = copy.deepcopy(self.next_qpos)
                test_params = copy.deepcopy(self._hyperparams)
                test_t_down = 0
                test_g_up = False
                test_g_closed = False

                test_ja = self._controller.limb.joint_angles()

                for test_t in range(self._hyperparams['T']):
                    if test_t == 0:
                        test_action = copy.deepcopy(mj_U)
                    else:
                        test_action = copy.deepcopy(policy.act(None, test_t))
                    test_next, test_t_down, test_g_up, test_g_closed = get_target_qpos(test_next, test_params,
                                                    test_action, test_t, test_g_up, test_g_closed, test_t_down, test_next[2])

                    try:
                        test_ja = inverse_kinematics.get_joint_angles(self.state_to_pose(test_next), seed_cmd=test_ja,
                                                            use_advanced_options=True)
                    except ValueError:
                        return None, False

            self.next_qpos, self.t_down, self.gripper_up, self.gripper_closed = get_target_qpos(
                self.next_qpos, self._hyperparams, mj_U, t, self.gripper_up, self.gripper_closed, self.t_down,
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

            wait_change = (self.next_qpos[-1] > 0.05) != (self.prev_qpos[-1] > 0.05)        #wait for gripper to ack change in status


            if self.next_qpos[-1] > 0.05:
                self._controller.close_gripper(wait_change)
            else:
                self._controller.open_gripper(wait_change)

            self._controller.move_with_impedance_sec(target_ja, duration=self._hyperparams['step_duration'])

        return traj, traj_ok

    def random_start_angles(self, rng = np.random.uniform):
        rand_ok = False
        start_joints = self._controller.limb.joint_angles()
        while not rand_ok:
            rand_state = np.zeros(self._hyperparams['adim'])
            for i in range(self._hyperparams['adim'] - 1):
                rand_state[i] = rng(self._hyperparams['targetpos_clip'][0][i], self._hyperparams['targetpos_clip'][1][i])
            start_pose = self.state_to_pose(rand_state)

            try:
                start_joints = inverse_kinematics.get_joint_angles(start_pose, seed_cmd=start_joints,
                                                                use_advanced_options=True)
                start_joints = [start_joints[n] for n in self._controller.limb.joint_names()]
                rand_ok = True
            except ValueError:
                rand_ok = False
        return start_joints


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

