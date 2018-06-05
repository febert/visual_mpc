import rospy
from sensor_msgs.msg import Image as Image_msg
import cv2
import numpy as np
import intera_interface
from cv_bridge import CvBridge
import copy
from threading import Lock, Semaphore
from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)
from sensor_msgs.msg import JointState
import os
import cPickle as pkl
import shutil
import moviepy.editor as mpy
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.primitives_regintervals import quat_to_zangle
NUM_JOINTS = 7 #Sawyer has 7 dof arm

class Trajectory:
    def __init__(self, agentparams):
        T = agentparams['T']
        self.sequence_length = T

        self.actions = np.zeros((T, agentparams['adim']))
        self.joint_angles = np.zeros((T, NUM_JOINTS))
        self.joint_velocities = np.zeros((T, NUM_JOINTS))
        self.endeffector_poses = np.zeros((T, 7))    #x,y,z + quaternion
        self.robot_states = np.zeros((T, agentparams['sdim']))

        cam_height = agentparams.get('cam_image_height', 401)
        cam_width = agentparams.get('cam_image_width', 625)
        self.raw_images = np.zeros((T, 2, cam_height, cam_width, 3), dtype = np.uint8)

        img_height, img_width = agentparams['image_height'], agentparams['image_width']
        self.images = np.zeros((T, 2, img_height, img_width, 3), dtype = np.uint8)
        self.touch_sensors = np.zeros((T, 2))   #2 fingers

        self.target_qpos = np.zeros((T + 1, agentparams['sdim']))
        self.mask_rel = copy.deepcopy(agentparams['mode_rel'])

        self._save_raw = 'no_raw_images' not in agentparams

    def save(self, file_path):
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                raise IOError("Path {}, refers to an existing file!".format(file_path))
            print("Traj folder exists! Deleting...")
            shutil.rmtree(file_path)
        print("Creating traj folder: {}".format(file_path))
        os.makedirs(file_path)

        with open(file_path + '/state_action.pkl', 'wb') as f:
            sa_dict = {'qpos' : self.joint_angles,
                       'qvel' : self.joint_velocities,
                       'finger_sensors' : self.touch_sensors,
                       'actions' : self.actions,
                       'states' : self.robot_states,
                       'target_qpos' : self.target_qpos,
                       'mask_rel' : self.mask_rel}
            pkl.dump(sa_dict, f)

        image_folders = [file_path +'/images{}'.format(i) for i in range(self.raw_images.shape[1])]

        for f, folder in enumerate(image_folders):
            os.makedirs(folder)
            clip = []
            for i in range(self.sequence_length):
                cv2.imwrite('{}/im{}.png'.format(folder, i), self.images[i, f, :, :, ::-1],
                            [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
                clip.append(self.images[i, f])
                if self._save_raw:
                    cv2.imwrite('{}/im_med{}.png'.format(folder, i), self.raw_images[i, f, :, :, ::-1],
                                [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
            clip = mpy.ImageSequenceClip(clip, fps = 5)
            clip.write_gif('{}/diag.gif'.format(folder))

class Latest_observation(object):
    def __init__(self):
        self.img_cv2 = None
        self.img_cropped = None
        self.tstamp_img = None
        self.img_msg = None
        self.mutex = Lock()

class RobotDualCamRecorder:
    def __init__(self, agent_params, robot_controller, OFFSET_TOL = 0.03):
        self.agent_params = agent_params
        self.data_conf = agent_params['data_conf']
        self._ctrl = robot_controller

        self.front_limage = Latest_observation()
        self.left_limage = Latest_observation()

        self.bridge =  CvBridge()
        self.front_first, self.front_sem = False, Semaphore(value=0)
        self.left_first, self.left_sem = False, Semaphore(value=0)
        rospy.Subscriber("/camera0/undistort/output/image", Image_msg, self.store_latest_f_im)
        rospy.Subscriber("/camera1/undistort/output/image", Image_msg, self.store_latest_l_im)
        self.left_sem.acquire()
        self.front_sem.acquire()
        print("Cameras subscribed")

        self.name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
        self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)

        self.obs_tol = OFFSET_TOL

    def store_recordings(self, traj, t):
        assert t < traj.sequence_length, "time t is larger than traj.sequence_length={}".format(t, traj.sequence_length)
        assert t >= 0, "t = {}, must be non-negative!".format(t)

        self.front_limage.mutex.acquire()
        self.left_limage.mutex.acquire()

        state, jv, ja, ee_pose, _, force_sensors = self.get_state(return_full=True)

        if self.front_limage.img_cv2 is not None and self.left_limage.img_cv2 is not None:
            read_ok = np.abs(self.front_limage.tstamp_img - self.left_limage.tstamp_img) <= self.obs_tol
            traj.raw_images[t, 0] = self.front_limage.img_cv2[:, :, ::-1]
            traj.images[t, 0] = self.front_limage.img_cropped[:, :, ::-1]
            traj.raw_images[t, 1] = self.left_limage.img_cv2[:, :, ::-1]
            traj.images[t, 1] = self.left_limage.img_cropped[:, :, ::-1]
        else:
            read_ok = False

        traj.joint_angles[t] = ja
        traj.joint_velocities[t] = jv
        traj.endeffector_poses[t] = ee_pose
        traj.robot_states[t] = state
        traj.touch_sensors[t] = force_sensors

        self.front_limage.mutex.release()
        self.left_limage.mutex.release()

        return read_ok

    def get_joint_angles(self):
        return np.array([self._ctrl.limb.joint_angle(j) for j in self._ctrl.limb.joint_names()])

    def get_gripper_state(self):
        g_width, g_force = self._ctrl.get_gripper_status(integrate_force=True)
        close_thresh, open_thresh = self._ctrl.get_limits()

        t = (g_width - close_thresh) / (open_thresh - close_thresh)  #t = 1 --> open, and t = 0 --> closed
        agent_open, agent_closed = self.agent_params['targetpos_clip'][0][-1], self.agent_params['targetpos_clip'][1][-1]
        gripper_status = agent_open * t + agent_closed * (1 - t)

        return gripper_status, np.array([g_force, g_force])

    def get_joint_angles_velocity(self):
        return np.array([self._ctrl.limb.joint_velocity(j) for j in self._ctrl.limb.joint_names()])

    def get_state(self, return_full = False):
        joint_velocity = self.get_joint_angles_velocity()
        joint_angles = self.get_joint_angles()
        ee_pose = self.get_endeffector_pose()
        gripper_state, force_sensors = self.get_gripper_state()

        z_angle = quat_to_zangle(ee_pose[3:])

        robot_state = np.array([ee_pose[0], ee_pose[1], ee_pose[2], z_angle, gripper_state])

        if return_full:
            return robot_state, joint_velocity, joint_angles, ee_pose, gripper_state, force_sensors

        return robot_state

    def get_endeffector_pose(self):
        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = self._ctrl.limb.joint_names()
        joints.position = [self._ctrl.limb.joint_angle(j)
                           for j in joints.name]

        # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
        fkreq.tip_names.append('right_hand')
        try:
            rospy.wait_for_service(self.name_of_service, 5)
            resp = self.fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        pos = np.array([resp.pose_stamp[0].pose.position.x,
                        resp.pose_stamp[0].pose.position.y,
                        resp.pose_stamp[0].pose.position.z,
                        resp.pose_stamp[0].pose.orientation.x,
                        resp.pose_stamp[0].pose.orientation.y,
                        resp.pose_stamp[0].pose.orientation.z,
                        resp.pose_stamp[0].pose.orientation.w])

        return pos

    def _proc_image(self, latest_obsv, data, cam_conf):
        latest_obsv.img_msg = data
        latest_obsv.tstamp_img = rospy.get_time()

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgra8")[:, :, :3]
        latest_obsv.img_cv2 = copy.deepcopy(cv_image)
        latest_obsv.img_cropped = self._crop_resize(cv_image, cam_conf)

    def store_latest_f_im(self, data):
        self.front_limage.mutex.acquire()
        front_conf = self.data_conf['front_cam']
        self._proc_image(self.front_limage, data, front_conf)
        self.front_limage.mutex.release()

        if not self.front_first:
            self.front_first = True
            self.front_sem.release()

    def store_latest_l_im(self, data):
        self.left_limage.mutex.acquire()
        left_conf = self.data_conf['left_cam']
        self._proc_image(self.left_limage, data, left_conf)
        self.left_limage.mutex.release()

        if not self.left_first:
            self.left_first = True
            self.left_sem.release()

    def _crop_resize(self, image, cam_conf):
        target_img_height, target_img_width = self.agent_params['image_height'], self.agent_params['image_width']

        crop_left, crop_right = cam_conf.get('crop_left', 0), cam_conf.get('crop_right', 0)
        crop_top, crop_bot = cam_conf.get('crop_top', 0), cam_conf.get('crop_bot', 0)

        if crop_right > 0:
            crop_img = image[:, crop_left:-crop_right]
        else:
            crop_img = image[:, crop_left:]

        if crop_bot > 0:
            crop_img = crop_img[crop_top:-crop_bot]
        else:
            crop_img = crop_img[crop_top:]

        return cv2.resize(crop_img, (target_img_width, target_img_height), interpolation=cv2.INTER_AREA)