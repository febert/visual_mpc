#!/usr/bin/env python
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_controller import RobotController
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_dualcam_recorder import RobotDualCamRecorder, high2low, crop_resize
from python_visual_mpc.video_prediction.basecls.utils.get_designated_pix import Getdesig
import pdb
import os
import cv2
import numpy as np
import cPickle as pkl
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.benckmarking.annotate_benchmark import get_folders
sudri_crop = {'left_cam': {'crop_bot': 70, 'crop_left': 130, 'crop_right': 120},
              'front_cam': {'crop_bot': 70, 'crop_left': 90, 'crop_right': 160}}
sudri_dict = {
    'robot_name': 'sudri',
    'targetpos_clip': [[0.375, -0.22, 0.184, -0.5 * np.pi, 0], [0.825, 0.24, 0.32, 0.5 * np.pi, 0.1]],
    'data_conf': sudri_crop
}

def main():
    agent_params = sudri_dict
    agent_params['image_height'] = 96
    agent_params['image_width'] = 128
    agent_params['data_save_dir'] = 'test_imgs'
    controller = RobotController()
    recorder = RobotDualCamRecorder(agent_params, controller)

    while True:
        pdb.set_trace()
        name = raw_input("Enter bench name:")
        goal_points = np.zeros((2, 2), dtype = np.int64)
        start_points = np.zeros((2, 2), dtype=np.int64)

        if not os.path.exists(name):
            os.mkdir(name)

        raw_input("Hit enter when ready to take GOAL image")
        status, front, left = recorder.get_images()
        if not status:
            print("DESYNC")
            continue

        cv2.imwrite('{}/front_goal_full.png'.format(name), front['raw'][:,:,::-1])
        cv2.imwrite('{}/left_goal_full.png'.format(name), left['raw'][:,:,::-1])
        cv2.imwrite('{}/front_goal.png'.format(name), front['crop'][:, :, ::-1])
        cv2.imwrite('{}/left_goal.png'.format(name), left['crop'][:, :, ::-1])
        #cam_conf, cam_height, cam_width, low_height, low_width
        point = Getdesig(front['raw']).coords
        print('point', point)
        point_low = high2low(point, agent_params['data_conf']['front_cam'],
                                  recorder.cam_height, recorder.cam_width,
                                  agent_params['image_height'], agent_params['image_width'])
        print('low', point_low)
        goal_points[0] = point_low
        goal_points[1] = high2low(Getdesig(left['raw']).coords, agent_params['data_conf']['left_cam'],
                                  recorder.cam_height, recorder.cam_width,
                                  agent_params['image_height'], agent_params['image_width'])
        raw_input("Hit enter when ready to reset")
        controller.set_neutral()

        raw_input("Hit enter when ready to take START image")
        status, front_s, left_s = recorder.get_images()
        if not status:
            print("DESYNC")
            continue
        cv2.imwrite('{}/front_start_full.png'.format(name), front_s['raw'][:,:,::-1])
        cv2.imwrite('{}/left_start_full.png'.format(name), left_s['raw'][:,:,::-1])
        cv2.imwrite('{}/front_start.png'.format(name), front_s['crop'][:,:,::-1])
        cv2.imwrite('{}/left_start.png'.format(name), left_s['crop'][:,:,::-1])

        start_points[0] = high2low(Getdesig(front_s['raw']).coords, agent_params['data_conf']['front_cam'],
                                  recorder.cam_height, recorder.cam_width,
                                  agent_params['image_height'], agent_params['image_width'])
        start_points[1] = high2low(Getdesig(left_s['raw']).coords, agent_params['data_conf']['left_cam'],
                                  recorder.cam_height, recorder.cam_width,
                                  agent_params['image_height'], agent_params['image_width'])


        pkl.dump({'goal' : goal_points, 'start' : start_points}, open('{}/start_goal_points.pkl'.format(name), 'wb'))

def copy_folders(folders, dest, new_height = 96, new_width = 128, data_conf = sudri_crop):
    for f in folders:
        exp_name = f.split('/')[-2]
        print(exp_name)
        if not os.path.exists(dest + '/{}'.format(exp_name)):
            os.makedirs(dest + '/{}'.format(exp_name))

        points = pkl.load(open(f + '/tracker_annotations.pkl', 'rb'))['points']
        new_points = np.zeros_like(points, np.int64)
        # copyfile(, dest + '/{}/tracker_annotations.pkl'.format(exp_name))

        for cam, name in enumerate(['front_cam', 'left_cam']):
            if not os.path.exists(dest + '/{}/images{}'.format(exp_name, cam)):
                os.makedirs(dest + '/{}/images{}'.format(exp_name, cam))
            for i in range(50):
                img = cv2.imread('{}/images{}/im_med{}.png'.format(f, cam, i))
                new_points[i, cam] = high2low(points[i, cam], data_conf[name],
                         img.shape[0], img.shape[1],
                         new_height, new_width)
                small_img = crop_resize(img, data_conf[name], new_height, new_width)
                cv2.imwrite(dest + '/{}/images{}/im{}.png'.format(exp_name, cam, i), small_img)
        pkl.dump({'points' : new_points},  open(dest + '/{}/tracker_annotations.pkl'.format(exp_name), 'wb'))



if __name__ =='__main__':
    # main()
    source = '/home/sudeep/Documents/ros_ws/src/visual_mpc/experiments/cem_exp/grasping_sawyer/indep_views_reuseaction_opencvtrack/exp_50'
    dest =  '/home/sudeep/Desktop/track_bench/gdn_runs'

    source_folders = []
    folders = get_folders(source)
    for f in folders:
        pkl_path = f + '/traj_data/tracker_annotations.pkl'
        if os.path.exists(pkl_path):
            loaded = pkl.load(open(pkl_path, 'rb'))
            print(f)
            np.sum(loaded['points'])   #test to ensure no Nones
            print(loaded['points'].shape)
            source_folders.append(f + '/traj_data')

    copy_folders(source_folders, dest)