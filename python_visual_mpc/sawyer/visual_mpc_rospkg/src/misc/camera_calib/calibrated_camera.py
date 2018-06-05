#!/usr/bin/python

import rospy
import numpy as np
import cPickle as pkl

from python_visual_mpc.region_proposal_networks.rpn_tracker import RPN_Tracker
import cv2

import scipy.spatial

class CalibratedCamera:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self._calib_folder = '/'.join(__file__.split('/')[:-1] + [self.robot_name])

        self.H_fcam = np.load('{}/H_fcam.npy'.format(self._calib_folder))
        self.t_fcam = np.load('{}/t_fcam.npy'.format(self._calib_folder))
        self.H_lcam = np.load('{}/H_lcam.npy'.format(self._calib_folder))
        self.t_lcam = np.load('{}/t_lcam.npy'.format(self._calib_folder))

        self._p2w_dict = pkl.load(open('{}/{}_point_to_world.pkl'.format(self._calib_folder, self.robot_name), 'rb'))

        self._camera_points = np.array([self._p2w_dict['top_left'], self._p2w_dict['top_right'],
                                  self._p2w_dict['bot_left'], self._p2w_dict['bot_right']])

        self._robot_points = np.array([self._p2w_dict['robot_top_left'], self._p2w_dict['robot_top_right'],
                                  self._p2w_dict['robot_bot_left'], self._p2w_dict['robot_bot_right']])

        self._cam_tri = scipy.spatial.Delaunay(self._camera_points)

        self.rpn_tracker = RPN_Tracker()

    def object_points(self, img, img_save_file = None):
        center_coords = self.rpn_tracker.get_boxes(img, valid_box= self._p2w_dict['valid_box'], im_save_dir=img_save_file)
        robot_coords = []

        targets = np.array([c[::-1] for c in center_coords])
        target_triangle = self._cam_tri.find_simplex(targets)
        for i, t in enumerate(target_triangle):
            b = self._cam_tri.transform[t, :2].dot((targets[i].reshape(1,2) - self._cam_tri.transform[t, 2]).T).T
            bcoord = np.c_[b, 1 - b.sum(axis=1)]

            points_robot_space = self._robot_points[self._cam_tri.simplices[t]]
            robot_coords.append(np.sum(points_robot_space * bcoord.T, axis = 0))
        return center_coords, robot_coords


def main():
    rospy.init_node('test_rpn')

    test_img = cv2.imread('test.png')[:, :, ::-1]

    calibrated_system = CalibratedCamera('vestri')
    center_coords, robot_coords = calibrated_system.object_points(test_img)

    print('robot point', robot_coords)

if __name__ == '__main__':
    main()