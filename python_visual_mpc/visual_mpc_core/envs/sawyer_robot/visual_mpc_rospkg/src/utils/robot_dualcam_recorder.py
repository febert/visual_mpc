#!/usr/bin/env python
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
import imageio

from python_visual_mpc.sawyer.visual_mpc_rospkg.src.primitives_regintervals import quat_to_zangle
NUM_JOINTS = 7 #Sawyer has 7 dof arm

def low2high(point, cam_conf, cam_height, cam_width, low_height, low_width):
    crop_left, crop_right = cam_conf.get('crop_left', 0), cam_conf.get('crop_right', 0)
    crop_top, crop_bot = cam_conf.get('crop_top', 0), cam_conf.get('crop_bot', 0)
    cropped_width, cropped_height = cam_width - crop_left - crop_right, cam_height - crop_bot - crop_top

    scale_height, scale_width = float(cropped_height) / low_height, \
                                float(cropped_width) / low_width
    high_point = np.array([scale_height, scale_width]) * point + np.array([crop_top, crop_left])

    return np.round(high_point).astype(np.int64)

def crop_resize(image, cam_conf, target_img_height, target_img_width):
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

def low2high(point, cam_conf, cam_height, cam_width, low_height, low_width):
    crop_left, crop_right = cam_conf.get('crop_left', 0), cam_conf.get('crop_right', 0)
    crop_top, crop_bot = cam_conf.get('crop_top', 0), cam_conf.get('crop_bot', 0)
    cropped_width, cropped_height = cam_width - crop_left - crop_right, cam_height - crop_bot - crop_top

    scale_height, scale_width = float(cropped_height) / low_height, \
                                float(cropped_width) / low_width
    high_point = np.array([scale_height, scale_width]) * point + np.array([crop_top, crop_left])

    return np.round(high_point).astype(np.int64)

def high2low(point, cam_conf, cam_height, cam_width, low_height, low_width):
    crop_left, crop_right = cam_conf.get('crop_left', 0), cam_conf.get('crop_right', 0)
    crop_top, crop_bot = cam_conf.get('crop_top', 0), cam_conf.get('crop_bot', 0)
    cropped_width, cropped_height = cam_width - crop_right - crop_left, cam_height - crop_bot - crop_top

    y = float(min(max(point[0] - crop_top, 0), cropped_height))
    x = float(min(max(point[1] - crop_left, 0), cropped_width))
    scale_height, scale_width = low_height / float(cropped_height), \
                                low_width / float(cropped_width)
    low_point = np.array([scale_height, scale_width]) * np.array([y, x])

    return np.round(low_point).astype(np.int64)

def render_bbox(img, bbox):
    rect_img = img[:, :, ::-1].copy()
    p1 = (bbox[0], bbox[1])
    p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    cv2.rectangle(rect_img, p1, p2, (0, 0, 255))
    return rect_img[:, :, ::-1]

class Trajectory:
    def __init__(self, agentparams):
        self._agent_conf = agentparams

        T = agentparams['T']
        self.sequence_length = T

        if 'autograsp' in agentparams:
            self.min = np.array(agentparams['targetpos_clip'][0][:3])
            self.delta = np.array(agentparams['targetpos_clip'][1][:3]) - self.min
            self.X_full = np.zeros((T, 5))

        self.actions = np.zeros((T, agentparams['adim']))
        self.joint_angles = np.zeros((T, NUM_JOINTS))
        self.joint_velocities = np.zeros((T, NUM_JOINTS))
        self.endeffector_poses = np.zeros((T, 7))    #x,y,z + quaternion
        self.robot_states = np.zeros((T, agentparams['sdim']))

        if 'opencv_tracking' in self._agent_conf:
            self.track_bbox = np.zeros((T, 2, 4), dtype=np.int32)

        cam_height = agentparams.get('cam_image_height', 401)
        cam_width = agentparams.get('cam_image_width', 625)
        self.raw_images = np.zeros((T, 2, cam_height, cam_width, 3), dtype = np.uint8)

        if 'image_medium' in self._agent_conf:
            img_med_height, img_med_width = self._agent_conf['image_medium']
            self.images = np.zeros((T, 2, img_med_height, img_med_width, 3), dtype=np.uint8)
        else:
            img_height, img_width = agentparams['image_height'], agentparams['image_width']
            self.images = np.zeros((T, 2, img_height, img_width, 3), dtype=np.uint8)

        self.touch_sensors = np.zeros((T, 2))   #2 fingers
        self.target_qpos = np.zeros((T + 1, agentparams['sdim']))
        self.mask_rel = copy.deepcopy(agentparams['mode_rel'])

        self._save_raw = 'no_raw_images' not in agentparams

        if 'save_videos' in self._agent_conf:
            self.frames = [[],[]]

    @property
    def i_tr(self):
        return 0



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
            if 'save_videos' in self._agent_conf:
                print('saving: {}/clip.mp4'.format(folder))
                writer = imageio.get_writer('{}/clip.mp4'.format(folder), fps=30)
                for frame in self.frames[f]:
                    writer.append_data(frame)
                writer.close()
            clip = []
            for i in range(self.sequence_length):
                cv2.imwrite('{}/im{}.png'.format(folder, i), self.images[i, f, :, :, ::-1],
                            [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
                if 'save_large_gifs' in self._agent_conf:
                    if 'opencv_tracking' in self._agent_conf:
                        raw_images_bgr = copy.deepcopy(self.raw_images[i, f])
                        clip.append(render_bbox(raw_images_bgr, self.track_bbox[i, f]))
                    else:
                        clip.append(self.raw_images[i, f])
                else:
                    clip.append(self.images[i, f])
                if self._save_raw:
                    cv2.imwrite('{}/im_med{}.png'.format(folder, i), self.raw_images[i, f, :, :, ::-1],
                                [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
            clip = mpy.ImageSequenceClip(clip, fps = 5)
            clip.write_gif('{}/diag.gif'.format(folder))

class Latest_observation(object):
    def __init__(self, create_tracker = False, save_buffer = False, medium_images = False):
        self.img_cv2 = None
        self.img_cropped = None
        self.tstamp_img = None
        self.img_msg = None
        self.mutex = Lock()
        self._medium = medium_images
        if save_buffer:
            self.save_itr = 0
        if create_tracker:
            self.reset_tracker()
        if medium_images:
            self.img_medium = None


    def reset_tracker(self):
        self.cv2_tracker = cv2.TrackerMIL_create()
        self.bbox = None
        self.track_itr = 0

    def to_dict(self):
        img_crop = self.img_cropped[:, :, ::-1].copy()
        img_raw = self.img_cv2[:, :, ::-1].copy()
        if not self._medium:
            return {'crop': img_crop, 'raw' : img_raw}
        img_med = self.img_medium[:, :, ::-1].copy()
        return {'crop': img_crop, 'raw': img_raw, 'med' : img_med}

class RobotDualCamRecorder:
    TRACK_SKIP = 2        #the camera publisher works at 60 FPS but camera itself only goes at 30
    def __init__(self, agent_params, robot_controller, OFFSET_TOL = 0.06):
        self.agent_params = agent_params
        self.data_conf = agent_params['data_conf']
        self._ctrl = robot_controller

        self.front_limage = Latest_observation('opencv_tracking' in agent_params,
                                               'save_videos' in self.agent_params, 'image_medium' in self.agent_params)
        self.left_limage = Latest_observation('opencv_tracking' in agent_params,
                                              'save_videos' in self.agent_params,  'image_medium' in self.agent_params)

        self._is_tracking = False
        if 'opencv_tracking' in agent_params:
            self.box_height = 80

        self.bridge =  CvBridge()
        self.front_first, self.front_sem = False, Semaphore(value=0)
        self.left_first, self.left_sem = False, Semaphore(value=0)
        if 'save_videos' in self.agent_params:
            self._buffers = [[],[]]
            self._saving = False

        rospy.Subscriber("/camera0/undistort/output/image", Image_msg, self.store_latest_f_im)
        rospy.Subscriber("/camera1/undistort/output/image", Image_msg, self.store_latest_l_im)
        self.left_sem.acquire()
        self.front_sem.acquire()
        print("Cameras subscribed")

        self.name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
        self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)

        if 'opencv_tracking' in agent_params:
            self.obs_tol = 0.1
        else: self.obs_tol = OFFSET_TOL



    def _low2high(self, point, cam_conf):
        return low2high(point, cam_conf, self.cam_height,
                        self.cam_width, self.agent_params['image_height'], self.agent_params['image_width'])

    def _high2low(self, point, cam_conf):
        return high2low(point, cam_conf, self.cam_height,
                        self.cam_width, self.agent_params['image_height'], self.agent_params['image_width'])

    def _cam_start_tracking(self, lt_ob, cam_conf, point):
        point = self._low2high(point, cam_conf)
        lt_ob.bbox = np.array([int(point[1] - self.box_height / 2.),
                               int(point[0] - self.box_height / 2.),
                               self.box_height, self.box_height]).astype(np.int64)

        lt_ob.cv2_tracker.init(lt_ob.img_cv2, tuple(lt_ob.bbox))
        lt_ob.track_itr = 0

    def start_tracking(self, start_points):
        assert 'opencv_tracking' in self.agent_params
        n_cam, n_desig, xy_dim = start_points.shape
        if n_cam != 2 or n_desig != 1:
            raise NotImplementedError("opencv_tracking requires 2 cameras and 1 designated pixel")
        if xy_dim != 2:
            raise ValueError("Requires XY pixel location")

        self.front_limage.mutex.acquire()
        self.left_limage.mutex.acquire()
        self._cam_start_tracking(self.front_limage, self.data_conf['front_cam'], start_points[0, 0])
        self._cam_start_tracking(self.left_limage, self.data_conf['left_cam'], start_points[1, 0])
        self._is_tracking = True
        self.front_limage.mutex.release()
        self.left_limage.mutex.release()
        rospy.sleep(2)   #sleep a bit for first few messages to initialize tracker

        print("TRACKING INITIALIZED")

    def end_tracking(self):
        self.front_limage.mutex.acquire()
        self.left_limage.mutex.acquire()
        self._is_tracking = False
        self.front_limage.reset_tracker()
        self.left_limage.reset_tracker()
        self.front_limage.mutex.release()
        self.left_limage.mutex.release()
    def _bbox2point(self, bbox):
        point = np.array([int(bbox[1]), int(bbox[0])]) \
                  + np.array([self.box_height / 2, self.box_height / 2])
        return point.astype(np.int32)
    def get_track(self):
        assert 'opencv_tracking' in self.agent_params
        assert self._is_tracking, "RECORDER IS NOT TRACKING"

        points = np.zeros((2, 1, 2), dtype=np.int64)
        self.front_limage.mutex.acquire()
        self.left_limage.mutex.acquire()
        points[0, 0] = self._high2low(self._bbox2point(self.front_limage.bbox), self.data_conf['front_cam'])
        points[1, 0] = self._high2low(self._bbox2point(self.left_limage.bbox), self.data_conf['left_cam'])
        self.front_limage.mutex.release()
        self.left_limage.mutex.release()

        return points.astype(np.int64)

    def get_images(self):
        self.front_limage.mutex.acquire()
        self.left_limage.mutex.acquire()
        front_stamp, left_stamp = self.front_limage.tstamp_img, self.left_limage.tstamp_img
        read_ok = np.abs(front_stamp - left_stamp) <= self.obs_tol
        if read_ok:
            front_dict = self.front_limage.to_dict()
            left_dict = self.left_limage.to_dict()
        self.front_limage.mutex.release()
        self.left_limage.mutex.release()
        if not read_ok:
            return False, None, None
        return True, front_dict, left_dict

    def store_recordings(self, traj, t):
        assert t < traj.sequence_length, "time t is larger than traj.sequence_length={}".format(t, traj.sequence_length)
        assert t >= 0, "t = {}, must be non-negative!".format(t)

        self.front_limage.mutex.acquire()
        self.left_limage.mutex.acquire()

        state, jv, ja, ee_pose, _, force_sensors = self.get_state(return_full=True)

        if self.front_limage.img_cv2 is not None and self.left_limage.img_cv2 is not None:
            read_ok = np.abs(self.front_limage.tstamp_img - self.left_limage.tstamp_img) <= self.obs_tol
            traj.raw_images[t, 0] = self.front_limage.img_cv2[:, :, ::-1]
            traj.raw_images[t, 1] = self.left_limage.img_cv2[:, :, ::-1]

            if 'image_medium' in self.agent_params:
                traj.images[t, 0] = self.front_limage.img_medium[:, :, ::-1]
                traj.images[t, 1] = self.left_limage.img_medium[:, :, ::-1]
            else:
                traj.images[t, 0] = self.front_limage.img_cropped[:, :, ::-1]
                traj.images[t, 1] = self.left_limage.img_cropped[:, :, ::-1]


            if self._is_tracking:
                traj.track_bbox[t, 0] = self.front_limage.bbox.copy()
                traj.track_bbox[t, 1] = self.left_limage.bbox.copy()
            if not read_ok:
                print("FRONT TIME {} VS LEFT TIME {}".format(self.front_limage.tstamp_img, self.left_limage.tstamp_img))
        else:
            read_ok = False

        traj.joint_angles[t] = ja
        traj.joint_velocities[t] = jv
        traj.endeffector_poses[t] = ee_pose
        traj.robot_states[t] = state
        traj.touch_sensors[t] = force_sensors

        if 'autograsp' in self.agent_params:
            norm_states = copy.deepcopy(np.concatenate((state[:-1], force_sensors), axis=0)[:-1])
            norm_states[:3] -= traj.min
            norm_states[:3] /= traj.delta
            print('norm_states at {}'.format(t), norm_states)
            traj.X_full[t] = norm_states

        self.front_limage.mutex.release()
        self.left_limage.mutex.release()

        return read_ok
    def start_recording(self, reset_buffer = True):
        if 'save_videos' in self.agent_params:
            self.front_limage.mutex.acquire()
            self.left_limage.mutex.acquire()
            self._saving = True
            self.front_limage.save_itr = 0
            self.left_limage.save_itr = 0
            if reset_buffer:
                self.reset_recording()
            self.front_limage.mutex.release()
            self.left_limage.mutex.release()

    def stop_recording(self, traj):
        if 'save_videos' in self.agent_params:
            self.front_limage.mutex.acquire()
            self.left_limage.mutex.acquire()

            for c, frames in enumerate(self._buffers):
                for f in frames:
                    traj.frames[c].append(f)
            self._buffers = [[], []]

            self.front_limage.mutex.release()
            self.left_limage.mutex.release()
    def reset_recording(self):
        if 'save_videos' in self.agent_params:
            self._buffers = [[], []]

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

        if 'image_medium' in self.agent_params:
            medium_height, medium_width  = self.agent_params['image_medium']
            latest_obsv.img_medium = crop_resize(cv_image, cam_conf, medium_height, medium_width)

        if 'opencv_tracking' in self.agent_params and self._is_tracking:
            if latest_obsv.track_itr % self.TRACK_SKIP == 0:
                _, bbox = latest_obsv.cv2_tracker.update(latest_obsv.img_cv2)
                latest_obsv.bbox = np.array(bbox).astype(np.int32).reshape(-1)
            latest_obsv.track_itr += 1



    def store_latest_f_im(self, data):
        self.front_limage.mutex.acquire()
        front_conf = self.data_conf['front_cam']
        self._proc_image(self.front_limage, data, front_conf)

        if not self.front_first:
            if not os.path.exists(self.agent_params['data_save_dir']):
                os.makedirs(self.agent_params['data_save_dir'])
            cv2.imwrite(self.agent_params['data_save_dir'] + '/front_test.png', self.front_limage.img_cropped)
            self.cam_height, self.cam_width = self.front_limage.img_cv2.shape[:2]
            self.front_first = True
            self.front_sem.release()
        if 'save_videos' in self.agent_params and self._saving:
            if self.front_limage.save_itr % self.TRACK_SKIP == 0:
                self._buffers[0].append(copy.deepcopy(self.front_limage.img_cv2)[:, :, ::-1])
            self.front_limage.save_itr += 1
        self.front_limage.mutex.release()


    def store_latest_l_im(self, data):
        self.left_limage.mutex.acquire()
        left_conf = self.data_conf['left_cam']
        self._proc_image(self.left_limage, data, left_conf)

        if not self.left_first:
            if not os.path.exists(self.agent_params['data_save_dir']):
                os.makedirs(self.agent_params['data_save_dir'])
            cv2.imwrite(self.agent_params['data_save_dir'] + '/left_test.png', self.left_limage.img_cropped)
            self.left_first = True
            self.left_sem.release()
        if 'save_videos' in self.agent_params and self._saving:
            if self.left_limage.save_itr % self.TRACK_SKIP == 0:
                self._buffers[1].append(copy.deepcopy(self.left_limage.img_cv2)[:, :, ::-1])
            self.left_limage.save_itr += 1

        self.left_limage.mutex.release()


    def _crop_resize(self, image, cam_conf):
        return crop_resize(image, cam_conf, self.agent_params['image_height'], self.agent_params['image_width'])


if __name__ == '__main__':
    from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_wsg_controller import WSGRobotController
    controller = WSGRobotController(800, 'vestri')

    dummy_agent = {'T': 1, 'image_height': 64, 'image_width': 48, 'sdim': 5, 'adim': 5, 'mode_rel': True,
                   'data_conf': {'left_cam': {}, 'front_cam': {}, },
                   'targetpos_clip' : [[0.375, -0.22, 0.184, -0.5 * np.pi, 0], [0.825, 0.24, 0.32, 0.5 * np.pi, 0.1]]}
    recorder = RobotDualCamRecorder(dummy_agent, controller)
    traj = Trajectory(dummy_agent)
    recorder.store_recordings(traj, 0)

    for c in range(2):
        cv2.imwrite('cam_{}_calib.png'.format(c), traj.raw_images[0, c][:,:, ::-1])
