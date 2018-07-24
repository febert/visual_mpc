from python_visual_mpc.visual_mpc_core.envs.base_env import BaseEnv
import numpy as np
import random
from geometry_msgs.msg import Quaternion as Quaternion_msg
from pyquaternion import Quaternion
from python_visual_mpc.visual_mpc_core.agent.general_agent import Image_Exception
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.util.limb_recorder import LimbWSGRecorder
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.util.camera_recorder import CameraRecorder
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.util.impedance_wsg_controller import ImpedanceWSGController
from python_visual_mpc.visual_mpc_core.envs.sawyer_robot.visual_mpc_rospkg.src.utils import inverse_kinematics
import copy


def quat_to_zangle(quat):
    """
    :param quat: robot rotation quaternion (assuming rotation around z-axis)
    :return: Rotation angle in z-axis
    """
    angle = -(Quaternion(axis = [0,1,0], angle = np.pi).inverse * Quaternion(quat)).angle
    if angle < 0:
        return angle + 2 * np.pi
    return angle


def zangle_to_quat(zangle):
    """
    :param zangle in radians
    :return: quaternion
    """
    return (Quaternion(axis=[0,1,0], angle=np.pi) * Quaternion(axis=[0, 0, 1], angle=2 * np.pi - zangle)).elements


def state_to_pose(xyz, quat):
    """
    :param xyz: desired pose xyz
    :param quat: quaternion around z angle in [w, x, y, z] format
    :return: stamped pose
    """
    quat = Quaternion_msg(  # downward and turn a little
        x=quat[1],
        y=quat[2],
        z=quat[3],
        w=quat[0]
    )

    desired_pose = inverse_kinematics.get_pose_stamped(xyz[0],
                                                       xyz[1],
                                                       xyz[2],
                                                       quat)
    return desired_pose


def pose_to_ja(target_pose, start_joints, tolerate_ik_error=False, debug_z = None):
    try:
        return inverse_kinematics.get_joint_angles(target_pose, seed_cmd=start_joints,
                                                        use_advanced_options=True)
    except ValueError:
        if tolerate_ik_error:
            raise ValueError("IK failure")    # signals to agent it should reset
        else:
            print 'zangle was {}'.format(debug_z)
            raise EnvironmentError("IK Failure")   # agent doesn't handle EnvironmentError


class BaseSawyerEnv(BaseEnv):
    def __init__(self, robot_name, substeps = 1, opencv_tracking=False, save_videos=False,
                 OFFSET_TOL = 0.06, duration = 1.5, mode_rel = np.array([True, True, True, True, False])):
        print('initializing environment for {}'.format(robot_name))
        self._robot_name = robot_name
        self._setup_robot()

        if opencv_tracking:
            self._obs_tol = 0.1
        else:
            self._obs_tol = OFFSET_TOL

        self._controller = ImpedanceWSGController(800, robot_name)
        self._limb_recorder = LimbWSGRecorder(self._controller)
        self._main_cam = CameraRecorder('/camera0/image_raw', opencv_tracking, save_videos)
        self._left_cam = CameraRecorder('/camera1/image_raw', opencv_tracking, save_videos)


        img_dim_check = (self._main_cam.img_height, self._main_cam.img_width) == \
                        (self._left_cam.img_height, self._left_cam.img_width)
        assert img_dim_check, 'Camera image streams do not match)'
        self._height, self._width = self._main_cam.img_height, self._main_cam.img_width

        assert substeps > 0, "Number of substeps must be positive"
        self._base_adim, self._base_sdim, self._substeps = 5, 5, substeps
        self._adim, self._sdim, self.mode_rel = None, None, mode_rel

        self._reset_counter, self._previous_target_qpos, self._duration = 0, None, duration
        self.num_objects = None  # for agent linkup.

    def _setup_robot(self):
        low_angle = np.pi / 2
        high_angle = 3 * np.pi / 2 - 0.001
        if self._robot_name == 'vestri':
            self._low_bound = np.array([0.42, -0.24, 0.184, low_angle, -1])
            self._high_bound = np.array([0.87, 0.22, 0.32, high_angle, 1])
        elif self._robot_name == 'sudri':
            self._low_bound = np.array([0.375, -0.22, 0.184, low_angle, -1])
            self._high_bound = np.array([0.825, 0.24, 0.32, high_angle, 1])
        else:
            raise ValueError("Supported robots are vestri/sudri")

    def step(self, action):
        """
        Applies the action and steps simulation
        :param action: action at time-step
        :return: obs dict where:
                  -each key is an observation at that step
                  -keys are constant across entire datastep (e.x. every-timestep has 'state' key)
                  -keys corresponding to numpy arrays should have constant shape every timestep (for caching)
                  -images should be placed in the 'images' key in a (ncam, ...) array
        """
        target_qpos = np.clip(self._next_qpos(action), self._low_bound, self._high_bound)
        wait_change = (target_qpos[-1] > 0) != (self._previous_target_qpos[-1] > 0)
        if target_qpos[-1] > 0:
            self._controller.close_gripper(wait_change)
        else:
            self._controller.open_gripper(wait_change)
        self._move_to_state(target_qpos[:3], target_qpos[3])

        self._previous_target_qpos = target_qpos
        return self._get_obs()

    def _init_dynamics(self):
        """
        Initializes custom dynamics for action space
        :return: None
        """
        raise NotImplementedError

    def _next_qpos(self, action):
        """
        Next target state given current state/actions
        :return: next_state
        """
        raise NotImplementedError

    def _get_obs(self):
        obs = {}
        j_angles, j_vel, eep, gripper_state, force_sensor = self._limb_recorder.get_state()
        obs['qpos'] = j_angles
        obs['qvel'] = j_vel

        print 'delta: ', np.linalg.norm(eep[:2] - self._previous_target_qpos[:2])
        print 'angle dif: ', abs(quat_to_zangle(eep[3:]) - self._previous_target_qpos[3]) * 180 / np.pi

        state = np.zeros(self._base_sdim)
        state[:3] = eep[:3]
        state[3] = quat_to_zangle(eep[3:])
        state[4] = gripper_state * self._low_bound[-1] + (1 - gripper_state) * self._high_bound[-1]
        obs['state'] = state
        obs['finger_sensors'] = force_sensor

        self._previous_obs = copy.deepcopy(obs)
        obs['images'] = self.render()

        return obs

    def _move_to_state(self, xyz, zangle):
        waypoints = []
        for i in xrange(1, self._substeps + 1):
            t = i / float(self._substeps)
            interp_xyz = (1 - t) * self._previous_target_qpos[:3] + t * xyz
            interp_zangle = (1 - t) * self._previous_target_qpos[3] + t * zangle
            interp_pose = state_to_pose(interp_xyz, zangle_to_quat(interp_zangle))

            interp_ja = pose_to_ja(interp_pose, self._limb_recorder.get_joint_cmd(), debug_z=interp_zangle * 180 / np.pi)
            interp_ja = [interp_ja[j] for j in self._limb_recorder.get_joint_names()]
            waypoints.append(interp_ja)

        self._controller.move_with_impedance(waypoints, self._duration)

    def _reset_previous_qpos(self):
        eep = self._limb_recorder.get_state()[2]
        self._previous_target_qpos = np.zeros(self._base_sdim)
        self._previous_target_qpos[:3] = eep[:3]
        self._previous_target_qpos[3] = quat_to_zangle(eep[3:])
        self._previous_target_qpos[4] = -1

    def reset(self):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        self._controller.neutral_with_impedance()
        self._controller.close_gripper(True)
        self._controller.open_gripper(True)

        self._reset_previous_qpos()

        self._move_to_state(self._previous_target_qpos[:3], np.pi/2)

        rand_xyz = np.random.uniform(self._low_bound[:3], self._high_bound[:3])
        rand_zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])

        self._move_to_state(rand_xyz, rand_zangle)
        self._init_dynamics()

        self._reset_previous_qpos()

        return self._get_obs()

    def valid_rollout(self):
        """
        Checks if the environment is currently in a valid state
        Common invalid states include:
            - object falling out of bin
            - mujoco error during rollout
        :return: bool value that is False if rollout isn't valid
        """
        return True

    def goal_reached(self):
        """
        Checks if the environment hit a goal (if environment has goals)
            - e.x. if goal is to lift object should return true if object lifted by gripper
        :return: whether or not environment reached goal state
        """
        raise NotImplementedError("Environment has No Goal")

    def has_goal(self):
        """
        :return: Whether or not environment has a goal
        """
        return False

    def render(self, mode='dual'):
        """ Grabs images form cameras.
        If returning multiple images asserts timestamps are w/in OBS_TOLERANCE, and raises Image_Exception otherwise

        - dual: renders both left and main cameras
        - left: renders only left camera
        - main: renders only main (front) camera
        :param mode: Mode to render with (dual by default)
        :return: uint8 numpy array with rendering from sim
        """
        cameras = [self._main_cam, self._left_cam]
        if mode == 'left':
            cameras = [self._left_cam]
        elif mode == 'main':
            cameras = [self._main_cam]

        time_stamps = []
        cam_imgs = []
        for c, recorder in enumerate(cameras):
            stamp, image = recorder.get_image()
            time_stamps.append(stamp)
            cam_imgs.append(image)

        for index, i in enumerate(time_stamps[:-1]):
            for j in time_stamps[index + 1:]:
                if abs(i - j) > self._obs_tol:
                    raise Image_Exception

        images = np.zeros((len(cameras), self._height, self._width, 3), dtype=np.uint8)
        for c, img in enumerate(cam_imgs):
            images[c] = img[:, :, ::-1]
        return images

    @property
    def adim(self):
        """
        :return: Environment's action dimension
        """
        return self._adim

    @property
    def sdim(self):
        """
        :return: Environment's state dimension
        """
        return self._sdim

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)