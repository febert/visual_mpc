from python_visual_mpc.visual_mpc_core.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import python_visual_mpc
import numpy as np
import mujoco_py
from pyquaternion import Quaternion
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.util.create_xml import create_object_xml, create_root_xml, clean_xml
from mujoco_py.builder import MujocoException
import copy
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_gif
import os
import time
from python_visual_mpc.visual_mpc_core.agent.general_agent import Environment_Exception

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

low_bound = np.array([-0.23, 0.62, 0.15, 0, -1])
high_bound = np.array([0.23, 0.95, 0.3, 2 * np.pi - 0.001, 1])
NEUTRAL_JOINTS = np.array([1.65474475, - 0.53312487, - 0.65980174, 1.1841825, 0.62772584, 1.11682223, 1.31015104, -0.05, 0.05])


class BaseSawyerMujocoEnv(BaseMujocoEnv):
    def __init__(self, env_params_dict, reset_state=None):
        assert 'filename' in env_params_dict, "Sawyer model filename required"
        params_dict = copy.deepcopy(env_params_dict)
        #TF HParams can't handle list Hparams well, this is cleanest workaround for object_meshes
        if 'object_meshes' in params_dict:
            object_meshes = params_dict.pop('object_meshes')
        else:
            object_meshes = None

        _hp = self._default_hparams()
        for name, value in params_dict.items():
            print('setting param {} to value {}'.format(name, value))
            _hp.set_hparam(name, value)

        base_filename = asset_base_path + _hp.filename
        friction_params = (_hp.friction, 0.1, 0.02)
        reset_xml = None
        if reset_state is not None:
            reset_xml = reset_state['reset_xml']

        self._reset_xml = create_object_xml(base_filename, _hp.num_objects, _hp.object_mass,
                                               friction_params, object_meshes, _hp.finger_sensors,
                                               _hp.maxlen, _hp.minlen, reset_xml,
                                                _hp.obj_classname, _hp.block_height, _hp.block_width)
        gen_xml = create_root_xml(base_filename)
        super().__init__(gen_xml, _hp)
        if _hp.clean_xml:
            clean_xml(gen_xml)


        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.]
                    )

        self._base_sdim, self._base_adim, self.mode_rel = 5, 5, _hp.mode_rel
        self.num_objects, self.skip_first, self.substeps = _hp.num_objects, _hp.skip_first, _hp.substeps
        self.randomize_initial_pos = _hp.randomize_initial_pos
        self.finger_sensors, self._maxlen = _hp.finger_sensors, _hp.maxlen

        self._previous_target_qpos, self._n_joints = None, 9
        self._read_reset_state = reset_state

        if self._hp.verbose_dir is not None:
            self._verbose_vid = []
    
    def _default_hparams(self):
        default_dict = {'filename': None,
                        'mode_rel': [True, True, True, True, False],
                        'num_objects': 1,
                        'object_mass': 1.0,
                        'friction': 1.0,
                        'finger_sensors': True,
                        'maxlen': 0.12,
                        'minlen': 0.01,
                        'obj_classname': 'freejoint',
                        'block_height': 0.02, 
                        'block_width': 0.02,
                        'skip_first': 80,
                        'substeps': 1000,
                        'randomize_initial_pos': True,
                        'verbose_dir': None,
                        'print_delta': False,
                        'clean_xml': True}
          
        parent_params = super()._default_hparams()
        parent_params.set_hparam('ncam', 2)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _clip_gripper(self):
        self.sim.data.qpos[7:9] = np.clip(self.sim.data.qpos[7:9], [-0.055, 0.0027], [-0.0027, 0.055])

    def render(self):
        imgs = super().render()
        if self._hp.verbose_dir is not None:
            self._verbose_vid.append(imgs[0, :, :, ::-1].copy())
        return imgs

    def _render_verbose(self):
        return self._verbose_vid.append(super().render()[0])

    def qpos_reset(self, qpos, qvel):
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        self.sim.set_state(sim_state)
        self.sim.forward()

        # do some other stuff that appeared to be necessary
        self._previous_target_qpos = np.zeros(self._base_sdim)
        self._previous_target_qpos[:3] = self.sim.data.get_body_xpos('hand')
        self._previous_target_qpos[3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        self._previous_target_qpos[-1] = low_bound[-1]

        finger_force = self.sim.data.sensordata[:2]
        self._init_dynamics()
        return self._get_obs(finger_force), None

    def reset(self, reset_state=None):
        """
        It's pretty important that we specify which reset functions to call
        instead of using super().reset() and self.reset()
           - That's because Demonstration policies use multiple inheritance to function and the recursive
             self.reset() results in pretty nasty errors. The pro to this approach is demonstration envs are easy to create
        """
        if reset_state is not None:
            self._read_reset_state = reset_state

        BaseMujocoEnv.reset(self)

        last_rands, write_reset_state = [], {}
        write_reset_state['reset_xml'] = copy.deepcopy(self._reset_xml)

        margin = 1.2 * self._maxlen
        if self._hp.verbose_dir is not None:
            print('resetting')

        if self._hp.verbose_dir is not None and len(self._verbose_vid) > 0:
            gif_num = np.random.randint(200000)
            npy_to_gif(self._verbose_vid, self._hp.verbose_dir + '/worker{}_verbose_traj_{}'.format(os.getpid(), gif_num), 20)
            self._verbose_vid = []


        def samp_xyz_rot():
            rand_xyz = np.random.uniform(low_bound[:3] + self._maxlen / 2 + 0.02, high_bound[:3] - self._maxlen / 2 + 0.02)
            rand_xyz[-1] = 0.05
            return rand_xyz, np.random.uniform(-np.pi / 2, np.pi / 2)

        object_poses = np.zeros((self.num_objects, 7))
        for i in range(self.num_objects):
            if self._read_reset_state is not None:
                obji_xyz = self._read_reset_state['object_qpos'][i][:3]
                obji_quat = self._read_reset_state['object_qpos'][i][3:]
            else:
                obji_xyz, rot = samp_xyz_rot()
                samp_cntr = 0
                #rejection sampling to ensure objects don't crowd each other
                while len(last_rands) > 0 and min([np.linalg.norm(obji_xyz[:2] - obj_j[:2])
                                                   for obj_j in last_rands]) < margin:
                    if samp_cntr >=100:          # avoid infinite looping by generating new env
                        return BaseSawyerMujocoEnv.reset(self)
                    obji_xyz, rot = samp_xyz_rot()
                    samp_cntr += 1
                last_rands.append(obji_xyz)

                obji_quat = Quaternion(axis=[0, 0, -1], angle=rot).elements
            object_poses[i, :3] = obji_xyz
            object_poses[i, 3:] = obji_quat

        self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.5]))
        self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))

        write_reset_state['object_qpos'] = copy.deepcopy(object_poses)
        object_poses = object_poses.reshape(-1)

        #placing objects then resetting to neutral risks bad contacts
        try:
            for s in range(5):
                self.sim.data.qpos[self._n_joints:] = object_poses.copy()
                self.sim.step()
            self.sim.data.qpos[:9] = NEUTRAL_JOINTS
            for _ in range(5):
                self.sim.step()
                if self._hp.verbose_dir is not None:
                    self._render_verbose()
        except MujocoException:
            return BaseSawyerMujocoEnv.reset(self)

        if self._read_reset_state is not None:
            end_eff_xyz = self._read_reset_state['state'][:3]
            end_eff_quat = zangle_to_quat(self._read_reset_state['state'][3])
        elif self.randomize_initial_pos:
            end_eff_xyz = np.random.uniform(low_bound[:3], high_bound[:3])
            while len(last_rands) > 0 and min([np.linalg.norm(end_eff_xyz[:2]-obj_j[:2]) for obj_j in last_rands]) < margin:
                end_eff_xyz = np.random.uniform(low_bound[:3], high_bound[:3])
            end_eff_quat = zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3]))
        else:
            end_eff_xyz = np.array([0, 0.5, 0.17])
            end_eff_quat = zangle_to_quat(np.pi)

        write_reset_state['state'] = np.zeros(7)
        write_reset_state['state'][:3], write_reset_state['state'][3:] = end_eff_xyz.copy(), end_eff_quat.copy()

        finger_force = np.zeros(2)
        if self._hp.verbose_dir is not None:
            print('skip_first: {}'.format(self.skip_first))


        assert self.skip_first > 25, "Skip first should be at least 15"
        sim_state = self.sim.get_state()
        self.sim.data.qpos[:9] = NEUTRAL_JOINTS
        self.sim.data.qpos[self._n_joints:] = object_poses.copy()
        sim_state.qvel[:] = np.zeros_like(self.sim.data.qvel)
        self.sim.set_state(sim_state)

        for t in range(self.skip_first):
            if t < 20:
                if t < 5:
                    self.sim.data.qpos[self._n_joints:] = object_poses.copy()
                reset_xyz = (low_bound[:3] + high_bound[:3]) * 0.5
                reset_xyz[-1] = 0.4
                self.sim.data.set_mocap_pos('mocap', reset_xyz)
                self.sim.data.set_mocap_quat('mocap', zangle_to_quat(0))
                # reset gripper
                self.sim.data.qpos[7:9] = NEUTRAL_JOINTS[7:9]
                self.sim.data.ctrl[:] = [-1, 1]
            else:
                self.sim.data.set_mocap_pos('mocap', end_eff_xyz)
                self.sim.data.set_mocap_quat('mocap', end_eff_quat)
                # reset gripper
                self.sim.data.qpos[7:9] = NEUTRAL_JOINTS[7:9]
                self.sim.data.ctrl[:] = [-1, 1]

            if self._hp.verbose_dir is not None and t % 2 == 0:
                print('skip: {}'.format(t))
                self._render_verbose()

            for _ in range(20):
                self._clip_gripper()
                try:
                    self.sim.step()

                except MujocoException:
                    #if randomly generated start causes 'bad' contacts Mujoco will error. Have to reset again
                    print('except')
                    return BaseSawyerMujocoEnv.reset(self)

            if self.finger_sensors:
                finger_force += self.sim.data.sensordata[:2]
        if self._hp.verbose_dir is not None:
            print('after')
        finger_force /= 10 * self.skip_first

        self._previous_target_qpos = np.zeros(self._base_sdim)
        self._previous_target_qpos[:3] = self.sim.data.get_body_xpos('hand')
        self._previous_target_qpos[3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        self._previous_target_qpos[-1] = low_bound[-1]

        self._init_dynamics()

        if self._read_reset_state is not None:
            self._check_positions(end_eff_xyz, end_eff_quat, object_poses)

        obs, reset = self._get_obs(finger_force), write_reset_state
        obs['control_delta'] = np.zeros(4)
        return obs, write_reset_state

    def _init_dynamics(self):
        raise NotImplementedError

    def _check_positions(self, des_end_eff_xyz, des_end_eff_quat, des_object_poses):
        ob_thresh = 0.06
        if np.linalg.norm(self.sim.data.qpos[self._n_joints:] - des_object_poses) > ob_thresh:
            raise Environment_Exception

        end_eff_pose = np.zeros(7)
        end_eff_pose[:3] = self.sim.data.get_body_xpos('hand')[:3]
        end_eff_pose[3:] = self.sim.data.get_body_xquat('hand')
        arm_thresh = 1e-3
        # if np.linalg.norm(end_eff_pose - np.concatenate([end_eff_xyz, end_eff_quat])) > arm_thresh:
        delta_angle = quat_to_zangle(end_eff_pose[3:]) - quat_to_zangle(des_end_eff_quat)
        if np.linalg.norm(np.array([np.sin(delta_angle), np.cos(delta_angle)]) - np.array([0,1])) > arm_thresh:
            raise Environment_Exception
        if np.linalg.norm(end_eff_pose[:3] - des_end_eff_xyz) > arm_thresh:
            raise Environment_Exception

    def current_obs(self):
        finger_force = np.zeros(2)
        if self.finger_sensors:
            finger_force += self.sim.data.sensordata[:2]
        return self._get_obs(finger_force)

    def _get_obs(self, finger_sensors=None):
        obs, touch_offset = {}, 0
        # report finger sensors as needed
        if self.finger_sensors:
            obs['finger_sensors'] = np.array([np.max(finger_sensors)]).reshape(-1)
            touch_offset = 2

        # joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:self._n_joints].squeeze())
        obs['qpos_full'] = copy.deepcopy(self.sim.data.qpos)
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:self._n_joints].squeeze())
        obs['qvel_full'] = copy.deepcopy(self.sim.data.qvel.squeeze())

        # control state
        obs['state'] = np.zeros(self._base_sdim)
        obs['state'][:3] = self.sim.data.get_body_xpos('hand')[:3]
        obs['state'][3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        obs['state'][-1] = self._previous_target_qpos[-1]

        # report object poses according to sensors (this is not equal to object_qpos)!!
        obs['object_poses_full'] = np.zeros((self.num_objects, 7))
        obs['object_qpos'] = np.zeros((self.num_objects, 7))
        obs['object_poses'] = np.zeros((self.num_objects, 3))
        for i in range(self.num_objects):
            fullpose = self.sim.data.qpos[i * 7 + self._n_joints:(i + 1) * 7 + self._n_joints].squeeze().copy()
            fullpose[:3] = self.sim.data.sensordata[touch_offset + i * 3:touch_offset + (i + 1) * 3]

            obs['object_poses_full'][i] = fullpose
            obs['object_poses'][i, :2] = fullpose[:2]
            obs['object_poses'][i, 2] = Quaternion(fullpose[3:]).angle
            obs['object_qpos'][i] = self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + (i+1)*7]

        # copy non-image data for environment's use (if needed)
        self._last_obs = copy.deepcopy(obs)
        # get images
        obs['images'] = self.render()
        obs['obj_image_locations'] = self.get_desig_pix(self._frame_width)
        obs['goal_obj_pose'] = self._goal_obj_pose

        if 'stage' in obs:
            raise ValueError

        return obs

    def _sim_integrity(self):
        xyztheta = np.zeros(4)
        xyztheta[:3] = self.sim.data.get_body_xpos('hand')
        xyztheta[3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))

        if not all(np.logical_and(xyztheta <= high_bound[:4] + 0.05, xyztheta >= low_bound[:4] - 0.05)):
            print('robot', xyztheta)
            # pdb.set_trace()
            return False

        for i in range(self.num_objects):
            obj_xy = self._last_obs['object_poses_full'][i][:2]
            z = self._last_obs['object_poses_full'][i][2]
            if not all(np.logical_and(obj_xy <= high_bound[:2] + 0.05, obj_xy >= low_bound[:2] - 0.05)):
                # pdb.set_trace()
                return False
            if z >= 0.5 or z <= -0.1:
                # pdb.set_trace()
                return False

        return True

    def step(self, action):
        if not self._sim_integrity():
            print('Sim reset (integrity)')
            raise Environment_Exception

        target_qpos = np.clip(self._next_qpos(action), low_bound, high_bound)
        assert target_qpos.shape[0] == self._base_sdim
        finger_force = np.zeros(2)

        start_xyz = self.sim.data.get_body_xpos('hand')[:3]
        for st in range(self.substeps):
            alpha = 1.
            if not self.substeps == 1:
                alpha = st / (self.substeps - 1)

            double_alpha = min(2 * alpha, 1.)
            target_angle = (1 - double_alpha) * self._previous_target_qpos[3] + double_alpha * target_qpos[3]

            xyz_interp = (1 - alpha) * start_xyz + alpha * target_qpos[:3]

            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(target_angle))
            self.sim.data.set_mocap_pos('mocap', xyz_interp)

            self.sim.data.ctrl[0] = self._previous_target_qpos[-1]
            self.sim.data.ctrl[1] = -self._previous_target_qpos[-1]

            for _ in range(20):
                self._clip_gripper()
                if self.finger_sensors:
                    finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())
                try:
                    self.sim.step()
                except MujocoException:
                    print('Sim reset (bad contact) 1')
                    raise Environment_Exception
            if self._hp.verbose_dir is not None and st % 10 == 0:
                self._render_verbose()

        for st in range(1000):
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(target_qpos[3]))
            self.sim.data.set_mocap_pos('mocap', target_qpos[:3])
            if target_qpos[-1] == self._previous_target_qpos[-1]:
                self.sim.data.ctrl[0] = target_qpos[-1]
                self.sim.data.ctrl[1] = -target_qpos[-1]
            else:
                alpha = min(st / 599., 1)
                mag = (1 - alpha) * self._previous_target_qpos[-1] + alpha * target_qpos[-1]
                self.sim.data.ctrl[0] = mag
                self.sim.data.ctrl[1] = -mag
            self._clip_gripper()

            if self.finger_sensors:
                finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())

            try:
                self.sim.step()
            except MujocoException:
                print('Sim reset (bad contact) 2')
                raise Environment_Exception

            if self._hp.verbose_dir is not None and st % 200 == 0:
                self._render_verbose()

        finger_force /= self.substeps * 10
        if np.sum(finger_force) > 0:
            print(finger_force)

        self._previous_target_qpos = target_qpos

        obs = self._get_obs(finger_force)
        self._post_step()
        obs['control_delta'] = np.abs(obs['state'][:4] - self._previous_target_qpos[:4])
        
        if self._hp.verbose_dir is not None or self._hp.print_delta:
            print('delta xy: {}, delta z {}, delta theta: {}, quat: {}'.format(np.linalg.norm(obs['control_delta'][:2]), obs['control_delta'][2], np.rad2deg(obs['control_delta'][3]), self.sim.data.get_body_xquat('hand')))
        return obs

    def _post_step(self):
        """
        Add custom behavior in sub classes for post-step checks
        (eg if goal has been reached)
            -Occurs after _get_obs so last_obs is available...
        :return: None
        """
        return

    def valid_rollout(self):
        object_zs = self._last_obs['object_poses_full'][:, 2]
        return not any(object_zs < -2e-2) and self._sim_integrity()

    def _next_qpos(self, action):
        raise NotImplementedError

    def move_arm(self):
        """
        move_arm randomly, used to create startgoal-configurations
        """
        arm_disp_range = 0.1
        arm_disp = np.random.uniform(-arm_disp_range, arm_disp_range, 2)
        arm_disp = np.concatenate([arm_disp, np.zeros(1)])

        armpos = self.sim.data.get_body_xpos('hand')[:3]
        new_armpos = armpos + arm_disp
        new_armpos[:2] = np.clip(new_armpos[:2], low_bound[:2], high_bound[:2])
        self.sim.data.set_mocap_pos('mocap', new_armpos)

        print('old arm pos', armpos)
        print('new arm pos', new_armpos)

        # self.sim.forward()
        for _ in range(self.skip_first*10):
            self._clip_gripper()
            self.sim.step()

    def move_objects(self):
        """
        move objects randomly, used to create startgoal-configurations
        """
        def get_new_obj_pose(curr_pos, curr_quat):
            angular_disp = 0.0
            delta_alpha = np.random.uniform(-angular_disp, angular_disp)
            delta_rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_alpha)
            curr_quat = Quaternion(curr_quat)
            newquat = delta_rot * curr_quat

            pos_ok = False
            while not pos_ok:
                const_dist = True
                if const_dist:
                    alpha = np.random.uniform(-np.pi, np.pi, 1)
                    d = 0.25
                    delta_pos = np.array([d * np.cos(alpha), d * np.sin(alpha), 0.])
                else:
                    pos_disp = 0.1
                    delta_pos = np.concatenate([np.random.uniform(-pos_disp, pos_disp, 2), np.zeros([1])])
                newpos = curr_pos + delta_pos
                lift_object = False
                if lift_object:
                    newpos[2] = 0.15
                if np.any(newpos[:2] > high_bound[:2]) or np.any(newpos[:2] < low_bound[:2]):
                    pos_ok = False
                else:
                    pos_ok = True

            return newpos, newquat

        for i in range(self.num_objects):
            curr_pos = self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7]
            curr_quat = self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7]
            obji_xyz, obji_quat = get_new_obj_pose(curr_pos, curr_quat)
            self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = obji_xyz
            self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7] = obji_quat.elements

        sim_state = self.sim.get_state()
        # sim_state.qpos[:] = sim_state.qpos
        sim_state.qvel[:] = np.zeros_like(sim_state.qvel)
        self.sim.set_state(sim_state)
        self.sim.forward()


if __name__ == '__main__':
        env = BaseSawyerMujocoEnv('sawyer_grasp.xml', np.array([True, True, True, True, False]))
        avg_200 = 0.
        for _ in range(200):
            timer = time.time()
            env.sim.render(640, 480)
            avg_200 += time.time() - timer
        avg_200 /= 200
        print('avg_100', avg_200)
