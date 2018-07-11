from python_visual_mpc.visual_mpc_core.envs.mujoco_env import BaseMujocoEnv
import python_visual_mpc
import numpy as np
import mujoco_py
from pyquaternion import Quaternion
from python_visual_mpc.visual_mpc_core.envs.cartgripper_env.util.create_xml import create_object_xml, create_root_xml, clean_xml
import moviepy.editor as mpy
import time

import copy

def quat_to_zangle(quat):
    angle = -(Quaternion(axis = [0,1,0], angle = np.pi).inverse * Quaternion(quat)).angle
    if angle < 0:
        return angle + 2 * np.pi
    return angle

def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return (Quaternion(axis=[0,1,0], angle=np.pi) * Quaternion(axis=[0, 0, -1], angle= zangle)).elements

BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/mjc_models/sawyer_assets/sawyer_xyz/'

low_bound = np.array([-0.27, 0.52, 0.15, 0, -1])
high_bound = np.array([0.27, 0.95, 0.3, 2 * np.pi - 0.001, 1])

class BaseSawyerEnv(BaseMujocoEnv):
    def __init__(self, filename, mode_rel, num_objects = 1, object_mass = 1, friction=1.0, finger_sensors=True,
                 maxlen=0.12, minlen=0.01, preload_obj_dict=None, object_meshes=None, obj_classname = 'freejoint',
                 block_height=0.02, block_width = 0.02, viewer_image_height = 480, viewer_image_width = 640,
                 skip_first=40, substeps=100, randomize_initial_pos = True):
        base_filename = asset_base_path + filename
        self.obj_stat_prop = create_object_xml(base_filename, num_objects, object_mass,
                                               friction, object_meshes, finger_sensors,
                                               maxlen, minlen, preload_obj_dict, obj_classname,
                                               block_height, block_width)
        gen_xml = create_root_xml(base_filename)
        super().__init__(gen_xml, viewer_image_height, viewer_image_width)
        clean_xml(gen_xml)

        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.]
                    )

        self._base_sdim, self._base_adim, self.mode_rel = 5, 5, mode_rel
        self.num_objects, self.skip_first, self.substeps = num_objects, skip_first, substeps
        self.randomize_initial_pos = randomize_initial_pos
        self.finger_sensors, self.object_sensors = finger_sensors, object_meshes is not None

        self._previous_target_qpos, self._n_joints = None, 9

    def _clip_gripper(self):
        self.sim.data.qpos[7:9] = np.clip(self.sim.data.qpos[7:9], [-0.055, 0.0027], [-0.0027, 0.055])

    def reset(self):
        if self.randomize_initial_pos:
            self.sim.data.set_mocap_pos('mocap', np.random.uniform(low_bound[:3], high_bound[:3]))
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))
        else:
            self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.17]))
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.pi))
        #reset gripper
        self.sim.data.qpos[7:9] = [-0.05573738, 0.05573735]
        self.sim.data.ctrl[:] = [-1, 1]

        for i in range(self.num_objects):
            rand_xyz = np.random.uniform(low_bound[:3], high_bound[:3])
            rand_xyz[2] = 0.3
            self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = rand_xyz

        finger_force = np.zeros(2)
        for _ in range(self.skip_first):
            for _ in range(10):
                self._clip_gripper()
                self.sim.step()
            if self.finger_sensors:
                finger_force += self.sim.data.sensordata[:2]
        finger_force /= 10 * self.skip_first
        if np.sum(finger_force) > 0:
            print(self.sim.data.sensordata)
            print(finger_force)

        self._previous_target_qpos = np.zeros(self._base_sdim)
        self._previous_target_qpos[:3] = self.sim.data.get_body_xpos('hand')
        self._previous_target_qpos[3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        self._previous_target_qpos[-1] = low_bound[-1]

        self._init_dynamics()

        return self._get_obs(finger_force)

    def _get_obs(self, finger_sensors):
        obs, touch_offset = {}, 0
        # report finger sensors as needed
        if self.finger_sensors:
            print(finger_sensors)
            obs['finger_sensors'] = finger_sensors
            touch_offset = 2

        # joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:self._n_joints].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:self._n_joints].squeeze())

        # control state
        obs['state'] = np.zeros(self._base_sdim)
        obs['state'][:3] = self.sim.data.get_body_xpos('hand')[:3]
        obs['state'][3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        obs['state'][-1] = self._previous_target_qpos[-1]

        # report object poses
        obs['object_poses_full'] = np.zeros((self.num_objects, 7))
        obs['object_poses'] = np.zeros((self.num_objects, 3))
        for i in range(self.num_objects):
            fullpose = self.sim.data.qpos[i * 7 + self._n_joints:(i + 1) * 7 + self._n_joints].squeeze().copy()

            if self.object_sensors:
                fullpose[:3] = self.sim.data.sensordata[touch_offset + i * 3:touch_offset + (i + 1) * 3].copy()
            obs['object_poses_full'][i] = fullpose
            obs['object_poses'][i, :2] = fullpose[:2]
            obs['object_poses'][i, 2] = quat_to_zangle(fullpose[3:])

        # copy non-image data for environment's use (if needed)
        self._last_obs = copy.deepcopy(obs)

        # get images
        obs['images'] = self.render()

        return obs

    def step(self, action):
        target_qpos = np.clip(self._next_qpos(action), low_bound, high_bound)
        assert target_qpos.shape[0] == self._base_sdim
        finger_force = np.zeros(2)

        self.sim.data.set_mocap_quat('mocap', zangle_to_quat(target_qpos[3]))
        for st in range(self.substeps):
            alpha = 1.
            if not self.substeps == 1:
                alpha = st / (self.substeps - 1)

            cur_xyz = self.sim.data.get_body_xpos('hand').copy()
            desired_xyz = cur_xyz * (1 - alpha) + alpha * target_qpos[:3]
            self.sim.data.set_mocap_pos('mocap', desired_xyz)

            self.sim.data.ctrl[0] = target_qpos[-1]
            self.sim.data.ctrl[1] = -target_qpos[-1]

            for _ in range(10):
                self._clip_gripper()
                if self.finger_sensors:
                    finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())
                self.sim.step()

        finger_force /= self.substeps * 10

        self._previous_target_qpos = target_qpos
        return self._get_obs(finger_force)

    def valid_rollout(self):
        object_zs = self._last_obs['object_poses_full'][:, 2]
        return not any(object_zs < -2e-2)

    def _next_qpos(self, action):
        raise NotImplementedError

    def _init_dynamics(self):
        raise NotImplementedError


if __name__ == '__main__':
        env = BaseSawyerEnv('sawyer_grasp.xml', np.array([True, True, True, True, False]))
        avg_100 = 0.
        for _ in range(100):
            timer = time.time()
            env.render()
            avg_100 += time.time() - timer
        avg_100 /= 100
        print('avg_100', avg_100)
