from python_visual_mpc.visual_mpc_core.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import python_visual_mpc
from python_visual_mpc.visual_mpc_core.envs.mujoco_env.util.create_xml import create_object_xml, create_root_xml, clean_xml
import copy
import time

BASE_DIR = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/mjc_models/'
low_bound = np.array([-0.5, -0.5, -0.08, -np.pi*2, -1])
high_bound = np.array([0.5, 0.5, 0.15, np.pi*2, 1])


def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return np.array([np.cos(zangle/2), 0, 0, np.sin(zangle/2)])


def quat_to_zangle(quat):
    """
    :param quat: quaternion with only
    :return: zangle in rad
    """
    theta = np.arctan2(2 * quat[0] * quat[3], 1 - 2 * quat[3] ** 2)
    return np.array([theta])


class BaseCartgripperEnv(BaseMujocoEnv):
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
        clean_xml(gen_xml)
        self._base_sdim, self._base_adim, self.mode_rel = 5, 5, _hp.mode_rel
        self.num_objects, self.skip_first, self.substeps = _hp.num_objects, _hp.skip_first, _hp.substeps
        self.randomize_initial_pos = _hp.randomize_initial_pos
        self.finger_sensors, self._maxlen = _hp.finger_sensors, _hp.maxlen

        self._previous_target_qpos, self._n_joints = None, 9
        self._read_reset_state = reset_state

        if self._hp.verbose_dir is not None:
            self._verbose_vid = []
            self._ctr = 0


    def _default_hparams(self):
        default_dict = {
                        'filename': None,
                        'mode_rel': [True, True, True, True, False],
                        'base_sdim':5,
                        'base_adim':5,
                        'num_objects':1,
                        'skip_first':50,
                        'substeps':50,
                        'sample_objectpos':True,
                        'object_object_mindist':-1,
                        'randomize_initial_pos':True,
                        'arm_obj_initdist':-1,
                        'arm_start_lifted':True,
                        'verbose_dir': None,
                        'object_mass': 1.0,
                        'friction': 1.0,
                        'finger_sensors': True,
                        'maxlen': 0.12,
                        'minlen': 0.01,
                        'obj_classname': 'freejoint',
                        'block_height': 0.02,
                        'block_width': 0.02}

        parent_params = super()._default_hparams()
        parent_params.set_hparam('ncam', 2)
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params


    def step(self, action):
        target_qpos = np.clip(self._next_qpos(action), low_bound, high_bound)
        assert target_qpos.shape[0] == self._base_sdim
        finger_force = np.zeros(2)

        for st in range(self.substeps):
            if self.finger_sensors:
                finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())

            alpha = st / float(self.substeps)
            self.sim.data.ctrl[:] = alpha * target_qpos + (1 - alpha) * self._previous_target_qpos
            self.sim.step()
        finger_force /= self.substeps

        self._previous_target_qpos = target_qpos
        return self._get_obs(finger_force)

    def render(self):
        return super().render()[:, ::-1]

    def reset(self):
        #clear our observations from last rollout
        self._last_obs = None

        # create random starting poses for objects
        def create_pos():
            poses = []
            for i in range(self.num_objects):
                pos = np.random.uniform(-.35, .35, 2)
                alpha = np.random.uniform(0, np.pi * 2)
                ori = np.array([np.cos(alpha / 2), 0, 0, np.sin(alpha / 2)])
                poses.append(np.concatenate((pos, np.array([0]), ori), axis=0))
            return poses

        if self._hp.sample_objectpos:  # if object pose explicit do not sample poses
            if self._hp.object_object_mindist:
                assert self.num_objects == 2
                ob_ob_dist = 0.
                while ob_ob_dist < self._hp.object_object_mindist:
                    object_pos_l = create_pos()
                    ob_ob_dist = np.linalg.norm(object_pos_l[0][:3] - object_pos_l[1][:3])
                object_pos = np.concatenate(object_pos_l)
            else:
                object_pos_l = create_pos()
                object_pos = np.concatenate(object_pos_l)
        else:
            object_pos = self.object_pos0[:self.num_objects]

        xpos0 = np.zeros(self._base_sdim + 1)
        if self.randomize_initial_pos:
            assert not self._hp.arm_obj_initdist
            xpos0[:2] = np.random.uniform(-.4, .4, 2)
            xpos0[2] = np.random.uniform(-0.08, .14)
        elif self._hp.arm_obj_initdist:
            d = self._hp.arm_obj_initdist
            alpha = np.random.uniform(-np.pi, np.pi, 1)
            delta_pos = np.array([d * np.cos(alpha), d * np.sin(alpha)])
            xpos0[:2] = object_pos[:2] + delta_pos.squeeze()
            xpos0[2] = np.random.uniform(-0.08, .14)
        else:
            xpos0_true_len = (self.sim.get_state().qpos.shape[0] - self.num_objects * 7)
            len_xpos0 = self._hp.xpos0.shape[0]

            if len_xpos0 != xpos0_true_len:
                xpos0 = np.concatenate([self._hp.xpos0, np.zeros(xpos0_true_len - len_xpos0)],
                                       0)  # testing in setting with updown rot, while data has only xyz
                print("appending zeros to initial robot configuration!!!")
            else:
                xpos0 = self._hp.xpos0
            assert xpos0.shape[0] == self._base_sdim + 1

        if self._hp.arm_start_lifted:
            xpos0[2] = self._hp.arm_start_lifted
        xpos0[-1] = low_bound[-1]    #start with gripper open
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = np.concatenate((xpos0, object_pos.flatten()), 0)
        sim_state.qvel[:] = np.zeros_like(sim_state.qvel)
        self.sim.set_state(sim_state)
        self.sim.forward()
        finger_force = np.zeros(2)
        for t in range(self.skip_first):
            for _ in range(self.substeps):
                self.sim.data.ctrl[:] = xpos0[:-1]
                self.sim.step()
                if self.finger_sensors:
                    finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())

        self._previous_target_qpos = copy.deepcopy(self.sim.data.qpos[:self._base_adim].squeeze())
        self._previous_target_qpos[-1] = low_bound[-1]
        self._init_dynamics()

        return self._get_obs(finger_force / self.skip_first / self.substeps)

    def _get_obs(self, finger_sensors):
        obs, touch_offset = {}, 0
        #report finger sensors as needed
        if self.finger_sensors:
            obs['finger_sensors'] = finger_sensors
            touch_offset = 2

        #joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:self._n_joints].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:self._n_joints].squeeze())

        #control state
        obs['state'] = np.zeros(self._base_sdim)
        obs['state'][:4] = self.sim.data.qpos[:4].squeeze()
        obs['state'][-1] = self._previous_target_qpos[-1]

        #report object poses
        obs['object_poses_full'] = np.zeros((self.num_objects, 7))
        obs['object_poses'] = np.zeros((self.num_objects, 3))
        for i in range(self.num_objects):
            fullpose = self.sim.data.qpos[i * 7 + self._n_joints:(i + 1) * 7 + self._n_joints].squeeze().copy()

            if self.object_sensors:
                fullpose[:3] = self.sim.data.sensordata[touch_offset + i * 3:touch_offset + (i + 1) * 3].copy()
            obs['object_poses_full'][i] = fullpose
            obs['object_poses'][i, :2] = fullpose[:2]
            obs['object_poses'][i, 2] = quat_to_zangle(fullpose[3:])

        #copy non-image data for environment's use (if needed)
        self._last_obs = copy.deepcopy(obs)

        #get images
        obs['images'] = self.render()

        obj_image_locations = np.zeros((2, self.num_objects + 1, 2))
        for i, cam in enumerate(['maincam', 'leftcam']):
            obj_image_locations[i, 0] = self.project_point(self.sim.data.get_body_xpos('hand')[:3], cam)
            for j in range(self.num_objects):
                obj_image_locations[i, j + 1] = self.project_point(obs['object_poses_full'][j, :3], cam)
        obj_image_locations[:, :, 0] = self._frame_height - obj_image_locations[:, :, 0]
        obs['obj_image_locations'] = obj_image_locations

        return obs

    def valid_rollout(self):
        object_zs = self._last_obs['object_poses_full'][:, 2]
        return not any(object_zs < -2e-2)

    def _init_dynamics(self):
        raise NotImplementedError

    def _next_qpos(self, action):
        raise NotImplementedError

    def snapshot_noarm(self):
        qpos = copy.deepcopy(self.sim.data.qpos)
        qpos[2] -= 10
        sim_state = self.sim.get_state()
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()
        image = self.render('maincam').squeeze()
        qpos[2] += 10
        sim_state.qpos[:] = qpos
        self.sim.set_state(sim_state)
        self.sim.forward()

        return image

if __name__ == '__main__':
        env = BaseCartgripperEnv('cartgripper_grasp.xml', 1, 0.1, 1, np.array([True, True, True, True, False]))
        avg_100 = 0.
        for _ in range(100):
            timer = time.time()
            env.sim.render(640, 480)
            avg_100 += time.time() - timer
        avg_100 /= 100
        print('avg_100', avg_100)