#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image as Image_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import shutil
import copy
import socket
import thread
from thread import start_new_thread

import numpy as np
import imutils
import pdb
from visual_mpc_rospkg.srv import *

from PIL import Image
import pickle
import imageio


import argparse

import moviepy.editor as mpy

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg


class Latest_observation(object):
    def __init__(self):
        #color image:
        self.img_cv2 = None
        self.img_cropped = None
        self.img_cropped_medium = None
        self.tstamp_img = None  # timestamp of image
        self.img_msg = None

class Trajectory(object):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.action_list = []
        self.joint_angle_list = []
        self.endeffector_pos_list = []
        self.desig_hpos_list = []
        self.highres_imglist = []
        self.track_desig_list = []

class RobotRecorder(object):
    def __init__(self, agent_params, dataconf = None, save_dir = None, seq_len = None, use_aux=True, save_video=False,
                 save_actions=True, save_images = True, image_shape=None, save_lowres=False):

        self.save_lowres = save_lowres
        self.save_actions = save_actions
        self.save_images = save_images
        self.dataconf = dataconf
        self.agent_params = agent_params
        """
        Records joint data to a file at a specified rate.
        rate: recording frequency in Hertz
        :param save_dir  where to save the recordings
        :param rate
        :param start_loop whether to start recording in a loop
        :param whether the recorder instance is an auxiliary recorder
        """

        side = "right"
        self.state_sequence_length = seq_len
        self.overwrite = True
        self.use_aux = use_aux

        self.save_video = save_video

        self.itr = 0

        if image_shape == None:
            self.img_height = 64
            self.img_width = 64
        else:
            self.img_height = image_shape[0]
            self.img_width = image_shape[1]

        if __name__ !=  '__main__':
            # the main instance one also records actions and joint angles
            self.instance_type = 'main'
            self._gripper = None
            self.gripper_name = '_'.join([side, 'gripper'])
            import intera_interface
            self._limb_right = intera_interface.Limb(side)
        else:
            # auxiliary recorder
            rospy.init_node('aux_recorder1')
            rospy.loginfo("init node aux_recorder1")
            self.instance_type = 'aux1'

        print('init recorder with instance type', self.instance_type)

        if self.dataconf is not None:
            self.crop_lowres = True
        else: self.crop_lowres = False

        prefix = self.instance_type

        rospy.Subscriber("/kinect2/hd/image_color", Image_msg, self.store_latest_im)

        self.save_dir = save_dir
        self.image_folder = save_dir
        self.ltob = Latest_observation()
        self.ltob_aux1 = Latest_observation()

        self.bridge = CvBridge()
        self.ngroup = 1000
        self.igrp = 0

        #for timing analysis:
        self.t_finish_save = []

        # if it is an auxiliary node advertise services
        if self.instance_type == 'aux1':
            # initializing the server:
            rospy.Service('save_kinectdata', save_kinectdata, self.save_kinect_handler)
            rospy.Service('get_kinectdata', get_kinectdata, self.get_kinect_handler)
            rospy.Service('init_traj', init_traj, self.init_traj_handler)
            rospy.Service('delete_traj', delete_traj, self.delete_traj_handler)

            self.t_get_request = []
            rospy.spin()

        elif self.instance_type == 'main':
            # initializing the client:
            self.get_kinectdata_func = rospy.ServiceProxy('get_kinectdata', get_kinectdata)
            self.save_kinectdata_func = rospy.ServiceProxy('save_kinectdata', save_kinectdata)
            self.init_traj_func = rospy.ServiceProxy('init_traj', init_traj)
            self.delete_traj_func = rospy.ServiceProxy('delete_traj', delete_traj)

            def spin_thread():
                rospy.spin()
            start_new_thread(spin_thread, ())
            print("Recorder intialized.")
            print("started spin thread")

        self.curr_traj = Trajectory(self.state_sequence_length)

    def save_kinect_handler(self, req):
        self.t_savereq = rospy.get_time()
        self.t_get_request.append(self.t_savereq)
        self._save_img_local(req.itr)
        return save_kinectdataResponse()

    def get_kinect_handler(self, req):
        print("handle get_kinect_request")

        img = np.asarray(self.ltob.img_cropped)
        img = self.bridge.cv2_to_imgmsg(img)
        return get_kinectdataResponse(img)

    def init_traj_handler(self, req):
        self.igrp = req.igrp
        self._init_traj_local(req.itr)
        return init_trajResponse()

    def delete_traj_handler(self, req):
        self.igrp = req.igrp
        try:
            self._delete_traj_local(req.itr)
        except:
            pass
        return delete_trajResponse()


    def store_latest_im(self, data):
        self.ltob.img_msg = data
        self.ltob.tstamp_img = rospy.get_time()

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  #(1920, 1080)
        self.ltob.img_cv2 = self.crop_highres(cv_image)

        if self.crop_lowres:
            self.ltob.img_cropped, self.ltob.img_cropped_medium = self._crop_lowres(self.ltob.img_cv2)  # use the cropped highres image


    def crop_highres(self, cv_image):
        colstart = 180
        rowstart = 0
        endcol = colstart + 1500
        endrow = rowstart + 1500
        cv_image = copy.deepcopy(cv_image[rowstart:endrow, colstart:endcol])

        shrink_after_crop = .75
        cv_image = cv2.resize(cv_image, (0, 0), fx=shrink_after_crop, fy=shrink_after_crop, interpolation=cv2.INTER_AREA)
        if self.instance_type == 'main':
            cv_image = imutils.rotate_bound(cv_image, 180)

        self.crop_highres_params = {'colstart':colstart, 'rowstart':rowstart,'shrink_after_crop':shrink_after_crop}
        return cv_image

    def _crop_lowres(self, cv_image):
        rowstart = self.dataconf['rowstart']  #10
        colstart = self.dataconf['colstart']  # 28
        shrink_before_crop = self.dataconf['shrink_before_crop']
        img = cv2.resize(cv_image, (0, 0), fx=shrink_before_crop, fy=shrink_before_crop, interpolation=cv2.INTER_AREA)
        img = img[rowstart:rowstart + self.img_height, colstart:colstart + self.img_width]
        assert img.shape == (self.img_height, self.img_width, 3)
        self.crop_lowres_params = {'colstart': colstart, 'rowstart': rowstart, 'shrink_before_crop': shrink_before_crop}

        if 'image_medium' in self.agent_params:
            rowstart = self.dataconf['rowstart']*2
            colstart = self.dataconf['colstart']*2
            shrink_before_crop = self.dataconf['shrink_before_crop']*2
            img_med = cv2.resize(cv_image, (0, 0), fx=shrink_before_crop, fy=shrink_before_crop, interpolation=cv2.INTER_AREA)
            img_med = img_med[rowstart:rowstart + self.img_height*2, colstart:colstart + self.img_width*2]
        else:
            img_med = None
        return img, img_med

    def init_traj(self, itr):
        assert self.instance_type == 'main'
        # request init service for auxiliary recorders
        if self.use_aux:
            try:
                rospy.wait_for_service('init_traj', timeout=1)
                resp1 = self.init_traj_func(itr, self.igrp)
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logerr("Service call failed: %s" % (e,))
                raise ValueError('get_kinectdata service failed')

        self._init_traj_local(itr)

        if ((itr+1) % self.ngroup) == 0:
            self.igrp += 1

    def _init_traj_local(self, itr):
        """
        :param itr: number of current trajecotry
        :return:
        """
        self.itr = itr
        self.group_folder = self.save_dir + '/traj_group{}'.format(self.igrp)

        rospy.loginfo("Init trajectory {} in group {}".format(itr, self.igrp))

        self.traj_folder = self.group_folder + '/traj{}'.format(itr)
        self.image_folder = self.traj_folder + '/images'
        self.depth_image_folder = self.traj_folder + '/depth_images'

        if os.path.exists(self.traj_folder):
            print("################################")
            print('trajectory folder {} already exists, deleting the folder'.format(self.traj_folder))
            shutil.rmtree(self.traj_folder)
        os.makedirs(self.traj_folder)
        os.makedirs(self.image_folder)
        os.makedirs(self.depth_image_folder)

        if self.instance_type == 'main':
            self.state_action_data_file = self.traj_folder + '/joint_angles_traj{}.txt'.format(itr)
            self.state_action_pkl_file = self.traj_folder + '/joint_angles_traj{}.pkl'.format(itr)

            self.curr_traj = Trajectory(self.state_sequence_length)

            joints_right = self._limb_right.joint_names()
            with open(self.state_action_data_file, 'w+') as f:
                f.write('time,')
                action_names = ['move','val_move_x','val_move_y','close','val_close','up','val_up']
                captions = joints_right + action_names
                f.write(','.join(captions) + ',' + '\n')


    def delete_traj(self, tr):
        assert self.instance_type == 'main'
        if self.use_aux:
            try:
                rospy.wait_for_service('delete_traj', 0.1)
                resp1 = self.delete_traj_func(tr, self.igrp)
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logerr("Service call failed: %s" % (e,))
                raise ValueError('delete traj service failed')
        self._delete_traj_local(tr)

    def _delete_traj_local(self, i_tr):
        self.group_folder = self.save_dir + '/traj_group{}'.format(self.igrp)
        traj_folder = self.group_folder + '/traj{}'.format(i_tr)
        shutil.rmtree(traj_folder)
        print('deleted {}'.format(traj_folder))

    def save(self, i_save, action, endeffector_pose, desig_hpos_main= None, track_desig_pos=None, goal_pos=None):
        self.t_savereq = rospy.get_time()
        assert self.instance_type == 'main'

        if self.use_aux:
            # request save at auxiliary recorders
            try:
                rospy.wait_for_service('get_kinectdata', 0.1)
                resp1 = self.save_kinectdata_func(i_save)
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logerr("Service call failed: %s" % (e,))
                raise ValueError('get_kinectdata service failed')

        if self.save_images:
            self._save_img_local(i_save)

        if self.save_actions:
            self._save_state_actions(i_save, action, endeffector_pose, track_desig_pos, goal_pos)

        if self.save_video:
            highres = cv2.cvtColor(self.ltob.img_cv2, cv2.COLOR_BGR2RGB)
            if desig_hpos_main is not None:
                self.curr_traj.desig_hpos_list.append(desig_hpos_main)
            self.curr_traj.highres_imglist.append(highres)

    def add_cross_hairs(self, images, desig_pos):
        out = []
        for im, desig in zip(images, desig_pos):
            for p in range(self.agent_params['ndesig']):
                desig = desig.astype(np.int64)

                im[:, desig[p, 1]] = np.array([0, 255., 255.])
                im[desig[p, 0],:] = np.array([0, 255., 255.])

            out.append(im)
        return out


    def save_highres(self):

        if 'opencv_tracking' in self.agent_params:
            highres_imglist = self.add_cross_hairs(self.curr_traj.highres_imglist, self.curr_traj.desig_hpos_list)
        else:
            highres_imglist = self.curr_traj.highres_imglist

        if 'make_final_vid' in self.agent_params:
            writer = imageio.get_writer(self.image_folder + '/highres_traj{}.mp4'.format(self.itr), fps=10)
            # add crosshairs to images in case of tracking:
            print('shape highres:', highres_imglist[0].shape)
            for im in highres_imglist:
                writer.append_data(im)
            writer.close()

        if 'make_final_gif' in self.agent_params:
            im_list = [cv2.resize(im, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for im in highres_imglist]

            clip = mpy.ImageSequenceClip(im_list, fps=4)
            clip.write_gif(self.image_folder + '/highres_traj{}.gif'.format(self.itr))

    def get_aux_img(self):
        try:
            rospy.wait_for_service('get_kinectdata', 0.1)
            resp1 = self.get_kinectdata_func()
            self.ltob_aux1.img_msg = resp1.image
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            raise ValueError('get_kinectdata service failed')

    def _save_state_actions(self, i_save, action, endeff_pose, track=None, goal_pos = None):
        joints_right = self._limb_right.joint_names()
        with open(self.state_action_data_file, 'a') as f:
            angles_right = [self._limb_right.joint_angle(j)
                            for j in joints_right]

            f.write("%f," % (i_save))
            f.write("%i," % (rospy.get_time(),))

            # values = np.concatenate([angles_right, action])
            # only writing actions to text file
            values = action
            f.write(','.join([str(x) for x in values]) + '\n')

        self.curr_traj.joint_angle_list.append(angles_right)
        self.curr_traj.action_list.append(action)
        self.curr_traj.endeffector_pos_list.append(endeff_pose)
        if track is not None:
            self.curr_traj.track_desig_list.append(track)

        if i_save == self.state_sequence_length-1:
            joint_angles = np.stack(self.curr_traj.joint_angle_list)
            actions = np.stack(self.curr_traj.action_list)
            endeffector_pos = np.stack(self.curr_traj.endeffector_pos_list)
            if track is not None:
                track_desig = np.stack(self.curr_traj.track_desig_list)
            assert joint_angles.shape[0] == self.state_sequence_length
            assert actions.shape[0] == self.state_sequence_length
            assert endeffector_pos.shape[0] == self.state_sequence_length
            assert endeffector_pos.shape[0] == self.state_sequence_length

            with open(self.state_action_pkl_file, 'wb') as f:
                dict= {'jointangles': joint_angles,
                       'actions': actions,
                       'endeffector_pos':endeffector_pos}
                if track is not None:
                    dict['track_desig'] = track_desig
                    dict['goal_pos'] = goal_pos
                pickle.dump(dict, f)

    def _save_img_local(self, i_save):

        pref = self.instance_type

        if self.save_lowres:
            image_name = self.image_folder + "/small{}.jpg".format(i_save)
            cv2.imwrite(image_name, self.ltob.img_cropped,[cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])

        if self.ltob.img_cv2 is not None:
            image_name = self.image_folder+ "/" + pref + "_full_cropped_im{0}.jpg".format(i_save)
            cv2.imwrite(image_name, self.ltob.img_cv2, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        else:
            raise ValueError('img_cv2 no data received')

        self.t_finish_save.append(rospy.get_time())
        if i_save == (self.state_sequence_length-1):
            with open(self.image_folder+'/{}_snapshot_timing.pkl'.format(pref), 'wb') as f:
                dict = {'t_finish_save': self.t_finish_save }
                if pref == 'aux1':
                    dict['t_get_request'] = self.t_get_request
                pickle.dump(dict, f)

    def low_res_to_highres(self, inp):
        h = self.crop_highres_params
        l = self.crop_lowres_params

        if 'wristrot' in self.agent_params:
            highres = (inp + np.array([l['rowstart'], l['colstart']])).astype(np.float) / l['shrink_before_crop']
        else:
            orig = (inp + np.array([l['rowstart'], l['colstart']])).astype(np.float) / l['shrink_before_crop']
            highres = (orig - np.array([h['rowstart'], h['colstart']])) * h['shrink_after_crop']

        highres = highres.astype(np.int64)
        return highres

    def high_res_to_lowres(self, inp):
        h = self.crop_highres_params
        l = self.crop_lowres_params

        if 'wristrot' in self.agent_params:
            lowres = inp.astype(np.float) * l['shrink_before_crop'] - np.array([l['rowstart'], l['colstart']])
        else:
            orig = inp.astype(np.float) / h['shrink_after_crop'] + np.array([h['rowstart'], h['colstart']])
            lowres = orig.astype(np.float) * l['shrink_before_crop'] - np.array([l['rowstart'], l['colstart']])

        lowres = lowres.astype(np.int64)
        return lowres


if __name__ ==  '__main__':
    print('started')
    rec = RobotRecorder('/home/guser/Documents/sawyer_data/newrecording', seq_len=48)
