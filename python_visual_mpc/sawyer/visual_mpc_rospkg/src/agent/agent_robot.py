import numpy as np
import rospy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_wsg_controller import WSGRobotController
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_dualcam_recorder import RobotDualCamRecorder, Trajectory, render_bbox

from python_visual_mpc.visual_mpc_core.agent.utils.target_qpos_utils import get_target_qpos
import copy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.primitives_regintervals import zangle_to_quat
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils import inverse_kinematics
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.misc.camera_calib.calibrated_camera import CalibratedCamera
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.visual_mpc_client import Getdesig
import moviepy.editor as  mpy
import pdb
import os
import cv2

class AgentSawyer:
    def __init__(self, agent_params):
        print('CREATING AGENT FOR ROBOT: {}'.format(agent_params['robot_name']))
        self._hyperparams = agent_params
        self.img_height, self.img_width = agent_params['image_height'], agent_params['image_width']

        # initializes node and creates interface with Sawyer
        self._controller = WSGRobotController(agent_params['control_rate'], agent_params['robot_name'])
        self._recorder = RobotDualCamRecorder(agent_params, self._controller)

        self._controller.reset_with_impedance()

        if 'rpn_objects' in agent_params:
            self._calibrated_camera = CalibratedCamera(agent_params['robot_name'])
        if 'save_large_gifs' in agent_params and 'opencv_tracking' in agent_params:
            self.track_save_dir = None



    def _select_points(self, front_cam, left_cam, fig_save_dir, clicks_per_desig = 2, n_desig = 1):
        assert clicks_per_desig == 1 or clicks_per_desig == 2, "CLICKS_PER_DESIG SHOULD BE 1 OR 2"

        start_pix = []
        if clicks_per_desig == 2:
            goal_pix = []
        cam_dicts = [front_cam, left_cam]
        for i, cam in enumerate(self._hyperparams['cameras']):
            c_main = Getdesig(cam_dicts[i]['crop'], fig_save_dir, 'goal_{}'.format(cam), n_desig=n_desig,
                              im_shape=[self.img_height, self.img_width], clicks_per_desig=clicks_per_desig)

            start_pos = c_main.desig.astype(np.int64)
            start_pix.append(start_pos.reshape(1, n_desig, 2))

            if clicks_per_desig == 2:
                goal_pos = c_main.goal.astype(np.int64)
                goal_pix.append(goal_pos.reshape(1, n_desig, 2))

        start_pix = np.concatenate(start_pix, 0)
        if clicks_per_desig == 2:
            goal_pix = np.concatenate(goal_pix, 0)
            return start_pix, goal_pix

        return  start_pix

    def sample(self, policy, itr):
        traj_ok = False
        max_tries = self._hyperparams.get('max_tries', 100)
        cntr = 0
        traj = None
        fig_save_dir = None

        if 'benchmark_exp' not in self._hyperparams:
            if itr % 30 == 0 and itr > 0:
                self._controller.redistribute_objects()
                print('Sampling {}'.format(itr))
            else:
                print('Sampling {}, redist in {}'.format(itr, 30 - itr % 30))
        else:
            print("BEGINNING BENCHMARK")

            save_dir = raw_input("Enter Experiment save_dir:")
            self._hyperparams['benchmark_exp'] = save_dir
            fig_save_dir = self._hyperparams['data_save_dir'] + '/' + save_dir
            record_dir = fig_save_dir + '/verbose'
            self._hyperparams['record'] = record_dir

            if not os.path.exists(record_dir):
                print("CREATING DIR: {}".format(record_dir))
                os.makedirs(record_dir)
            else:
                print("WARNING PATH EXISTS: {}".format(record_dir))

        while not traj_ok and cntr < max_tries:
            if 'benchmark_exp' in self._hyperparams:
                if cntr > 0:
                    if 'y' not in raw_input("would you like to retry benchmark (answer y/n)?"):
                        break
                ntasks = self._hyperparams.get('ntask', 1)
                if 'register_gtruth' in self._hyperparams and len(self._hyperparams['register_gtruth']) == 2:
                    self._controller.reset_with_impedance(duration=1.0)
                    self._controller.disable_impedance()
                    print("PLACE OBJECTS IN GOAL POSITION")
                    raw_input("When ready to annotate GOAL images press enter...")

                    read_ok, front_goal, left_goal = self._recorder.get_images()
                    if not read_ok:
                        print("CAMERA DESYNC")
                        break
                    goal_dir = fig_save_dir + '/goal'
                    if not os.path.exists(goal_dir):
                        os.makedirs(goal_dir)
                    if 'image_medium' in self._hyperparams:
                        front_goal_float, left_goal_float = front_goal['med'].astype(np.float32) / 255., \
                                                            left_goal['med'].astype(np.float32) / 255.
                    else:
                        front_goal_float, left_goal_float =front_goal['crop'].astype(np.float32) / 255., \
                                                           left_goal['crop'].astype(np.float32) / 255.

                    goal_images = np.concatenate((front_goal_float[None], left_goal_float[None]), 0)
                    print('goal_images shape', goal_images.shape)

                    goal_pix = self._select_points(front_goal, left_goal, goal_dir, clicks_per_desig=1, n_desig=ntasks)

                    start_dir = fig_save_dir + '/start'
                    if not os.path.exists(start_dir):
                        os.makedirs(start_dir)

                    raw_input("Robot in safe position? Hit enter when ready...")
                    self._controller.set_neutral()
                    self._controller.enable_impedance()
                    self._controller.reset_with_impedance(duration=2.0, reset_sitffness=60)
                    print("PLACE OBJECTS IN START POSITION")
                    raw_input("When ready to annotate START images press enter...")
                    read_ok, front_start, left_start = self._recorder.get_images()
                    if not read_ok:
                        print("CAMERA DESYNC")
                        break
                    start_pix = self._select_points(front_start, left_start, start_dir, clicks_per_desig=1, n_desig=ntasks)

                    traj, traj_ok = self.rollout(policy, start_pix, goal_pix, goal_images)
                    cntr += 1
                else:
                    self._controller.reset_with_impedance(duration=1.0)
                    print("PLACE OBJECTS IN START POSITION")
                    raw_input("When ready to annotate START/GOAL press enter...")
                    read_ok, front_cam, left_cam = self._recorder.get_images()
                    if not read_ok:
                        print("CAMERA DESYNC")
                        break
                    start_pix, goal_pix = self._select_points(front_cam, left_cam, fig_save_dir, n_desig=ntasks)
                    if 'opencv_tracking' in self._hyperparams:
                        self._recorder.start_tracking(start_pix)

                    traj, traj_ok = self.rollout(policy, start_pix, goal_pix)
                    cntr += 1

                    if 'opencv_tracking' in self._hyperparams:
                        self._recorder.end_tracking()
            else:
                self._controller.reset_with_impedance(duration=1.0, close_first=True)  # go to neutral
                traj, traj_ok = self.rollout(policy)

        return traj, traj_ok

    def rollout(self, policy, start_pix = None, goal_pix = None, goal_image = None):
        traj = Trajectory(self._hyperparams)
        traj_ok = True

        self.t_down = 0
        self.gripper_up, self.gripper_closed = False, False

        if 'rpn_objects' in self._hyperparams:
            if not self._recorder.store_recordings(traj, 0): #grab frame of robot in neutral
                return None, False
            neutral_img = traj.raw_images[0, 0]
            _, rbt_coords = self._calibrated_camera.object_points(neutral_img)

            traj.Object_pose = np.zeros((1, len(rbt_coords), 3))
            for i, r in enumerate(rbt_coords):
                traj.Object_pose[0, i] = r

        if 'randomize_initial_pos' in self._hyperparams:
            self._controller.reset_with_impedance(angles=self.random_start_angles(), open_gripper=False, duration= 1.0,
                                                  stiffness=self._hyperparams['impedance_stiffness'])
        else:
            self._controller.reset_with_impedance(open_gripper=False, duration=1.0,
                                                  stiffness=self._hyperparams['impedance_stiffness'])
        rospy.sleep(0.3)   #let things settle
        self._recorder.reset_recording()
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

                if 'opencv_tracking' in self._hyperparams:
                    start_pix = self._recorder.get_track()

            mj_U = policy.act(traj, t, start_pix, goal_pix, goal_image)
            traj.actions[t] = copy.deepcopy(mj_U)

            self.next_qpos, self.t_down, self.gripper_up, self.gripper_closed = get_target_qpos(
                self.next_qpos, self._hyperparams, mj_U, t, self.gripper_up, self.gripper_closed, self.t_down,
                traj.robot_states[t, 2], traj.touch_sensors)

            traj.target_qpos[t + 1] = copy.deepcopy(self.next_qpos)

            target_pose = self.state_to_pose(self.next_qpos)

            start_joints = self._controller.limb.joint_angles()
            try:
                target_ja = inverse_kinematics.get_joint_angles(target_pose, seed_cmd=start_joints,
                                                                    use_advanced_options=True)
            except ValueError:
                if 'benchmark_exp' in self._hyperparams:
                    print("ERR CAN'T APPLY")    #often gets triggered by robot twisting in corner during benchmark
                    pdb.set_trace()
                    continue
                else:
                    rospy.logerr('no inverse kinematics solution found, '
                                 'going to reset robot...')
                    current_joints = self._controller.limb.joint_angles()
                    self._controller.limb.set_joint_positions(current_joints)
                    return None, False

            wait_change = (self.next_qpos[-1] > 0.05) != (self.prev_qpos[-1] > 0.05)        #wait for gripper to ack change in status

            self._recorder.start_recording()
            if self.next_qpos[-1] > 0.05:
                self._controller.close_gripper(wait_change)
            else:
                self._controller.open_gripper(wait_change)

            self._controller.move_with_impedance_sec(target_ja, duration=self._hyperparams['step_duration'])
            self._recorder.stop_recording(traj)

        if not traj_ok:
            print("FAILED ROLLOUT RETRYING....")
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

