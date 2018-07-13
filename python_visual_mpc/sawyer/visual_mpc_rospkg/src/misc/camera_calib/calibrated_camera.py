#!/usr/bin/python

import rospy
import numpy as np
import cPickle as pkl

from python_visual_mpc.region_proposal_networks.rpn_tracker import RPN_Tracker
import cv2

import scipy.spatial

CAM_FUZZ = np.array([0, 133])      #I'm not 100% sure why we need this. Maybe correction in principal point  for ar alvar vs image_undistort?

class CalibratedCamera:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        calib_base = __file__.split('/')[:-1]
        self._calib_folder = '/'.join(calib_base + [self.robot_name])
        cam_K_path = '/'.join(calib_base + ['K_cam.npy'])

        self.H_fcam = np.load('{}/H_fcam.npy'.format(self._calib_folder))
        self.t_fcam = np.load('{}/t_fcam.npy'.format(self._calib_folder))
        self.H_lcam = np.load('{}/H_lcam.npy'.format(self._calib_folder))
        self.t_lcam = np.load('{}/t_lcam.npy'.format(self._calib_folder))
        self.K = np.load(cam_K_path)

        self._p2w_dict = pkl.load(open('{}/{}_point_to_world.pkl'.format(self._calib_folder, self.robot_name), 'rb'))

        self._camera_points = np.array([self._p2w_dict['top_left'], self._p2w_dict['top_right'],
                                  self._p2w_dict['bot_left'], self._p2w_dict['bot_right']])

        self._robot_points = np.array([self._p2w_dict['robot_top_left'], self._p2w_dict['robot_top_right'],
                                  self._p2w_dict['robot_bot_left'], self._p2w_dict['robot_bot_right']])

        self._cam_tri = scipy.spatial.Delaunay(self._camera_points)

        self.rpn_tracker = RPN_Tracker()

    def object_points(self, img, img_save_file = None):
        center_coords = self.rpn_tracker.get_boxes(img, valid_box= self._p2w_dict['valid_box'], im_save_dir=img_save_file)

        return center_coords, self.camera_to_robot(center_coords)
    def robot_to_camera(self, robot_points, cam_name = 'front'):
        if cam_name == 'front' or cam_name == 0:
            H, t = self.H_fcam, self.t_fcam
        else:
            H, t = self.H_lcam, self.t_lcam
        camera_coords = []

        for r in robot_points:
            c = np.linalg.solve(H, r.reshape((3, 1)) - t)
            c_norm = c / c[2, 0]
            raw_xy = self.K.dot(c_norm).reshape(-1)
            pixel_xy = raw_xy[:2]
            pixel_xy -= CAM_FUZZ
            camera_coords.append(pixel_xy)
        return camera_coords

    def camera_to_robot(self, camera_coord, name = 'front'):
        assert name == 'front', "calibration for camera_to_object not performed for left cam"
        robot_coords = []

        targets = np.array([c[::-1] for c in camera_coord])
        target_triangle = self._cam_tri.find_simplex(targets)
        for i, t in enumerate(target_triangle):
            b = self._cam_tri.transform[t, :2].dot((targets[i].reshape(1, 2) - self._cam_tri.transform[t, 2]).T).T
            bcoord = np.c_[b, 1 - b.sum(axis=1)]

            points_robot_space = self._robot_points[self._cam_tri.simplices[t]]
            robot_coords.append(np.sum(points_robot_space * bcoord.T, axis=0))
        return robot_coords


def main():
    rospy.init_node('test_rpn')

    test_img = cv2.imread('test.png')

    calibrated_system = CalibratedCamera('vestri')
    center_coords, robot_coords = calibrated_system.object_points(test_img[:, :, ::-1].copy())

    print('camera point', center_coords)
    print('robot point', robot_coords)

    shifted_up = robot_coords[0].copy()

    shifted_up[2] += 0.18

    translated_point = calibrated_system.robot_to_camera([shifted_up])
    print('translated camera point', translated_point)

    cv2.circle(test_img, (int(center_coords[0][0]), int(center_coords[0][1])), 3, (0, 0, 255), -1)
    cv2.circle(test_img, (int(translated_point[0][0]), int(translated_point [0][1])), 3, (0, 0, 255), -1)
    cv2.imshow('found points', test_img)
    cv2.waitKey(-1)

if __name__ == '__main__':
    main()