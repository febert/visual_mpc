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
import numpy as np
import imutils
import pdb
from berkeley_sawyer.srv import *
from PIL import Image
import cPickle
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
        self.tstamp_img = None  # timestamp of image
        self.img_msg = None

        #depth image:
        self.d_img_raw_npy = None  # 16 bit raw data
        self.d_img_cropped_npy = None
        self.d_img_cropped_8bit = None
        self.tstamp_d_img = None  # timestamp of image
        self.d_img_msg = None


class RobotRecorder(object):
    def __init__(self, save_dir, seq_len = None, use_aux=True, save_video=False,
                 save_actions=True, save_images = True):

        self.save_actions = save_actions
        self.save_images = save_images

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

        if save_video:
            self.save_gif = True
        else:
            self.save_gif = False

        self.image_folder = save_dir
        self.itr = 0
        self.highres_imglist = []

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

        print 'init recorder with instance type', self.instance_type

        prefix = self.instance_type

        rospy.Subscriber(prefix + "/kinect2/hd/image_color", Image_msg, self.store_latest_im)
        rospy.Subscriber(prefix + "/kinect2/sd/image_depth_rect", Image_msg, self.store_latest_d_im)

        self.save_dir = save_dir
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
            thread.start_new(spin_thread, ())
            print "Recorder intialized."
            print "started spin thread"
            self.action_list, self.joint_angle_list, self.cart_pos_list = [], [], []


    def save_kinect_handler(self, req):
        self.t_savereq = rospy.get_time()
        self.t_get_request.append(self.t_savereq)
        self._save_img_local(req.itr)
        return save_kinectdataResponse()

    def get_kinect_handler(self, req):
        print "handle get_kinect_request"

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

    def store_latest_d_im(self, data):
        # if self.ltob.tstamp_img != None:
            # rospy.loginfo("time difference to last stored dimg: {}".format(
            #     rospy.get_time() - self.ltob.tstamp_d_img
            # ))

        self.ltob.tstamp_d_img = rospy.get_time()

        self.ltob.d_img_msg = data
        cv_image = self.bridge.imgmsg_to_cv2(data, '16UC1')

        self.ltob.d_img_raw_npy = np.asarray(cv_image)
        img = cv2.resize(cv_image, (0, 0), fx=1 /5.5, fy=1 / 5.5, interpolation=cv2.INTER_AREA)

        img = np.clip(img,0, 1400)

        startcol = 7
        startrow = 0
        endcol = startcol + 64
        endrow = startrow + 64
        #crop image:
        img = img[startrow:endrow, startcol:endcol]

        self.ltob.d_img_cropped_npy = img
        img = img.astype(np.float32)/ np.max(img) *256
        img = img.astype(np.uint8)
        img = np.squeeze(img)
        self.ltob.d_img_cropped_8bit = img


    def store_latest_im(self, data):

        self.ltob.img_msg = data
        self.ltob.tstamp_img = rospy.get_time()
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  #(1920, 1080)

        self.ltob.img_cv2 = self.crop_highres(cv_image)
        self.ltob.img_cropped = self.crop_lowres(cv_image)

    def crop_highres(self, cv_image):
        startcol = 180
        startrow = 0
        endcol = startcol + 1500
        endrow = startrow + 1500
        cv_image = copy.deepcopy(cv_image[startrow:endrow, startcol:endcol])

        cv_image = cv2.resize(cv_image, (0, 0), fx=.75, fy=.75, interpolation=cv2.INTER_AREA)
        if self.instance_type == 'main':
            cv_image = imutils.rotate_bound(cv_image, 180)
        return cv_image

    def crop_lowres(self, cv_image):
        self.ltob.d_img_raw_npy = np.asarray(cv_image)
        if self.instance_type == 'main':
            img = cv2.resize(cv_image, (0, 0), fx=1 / 16., fy=1 / 16., interpolation=cv2.INTER_AREA)
            startrow = 3
            startcol = 27

            img = imutils.rotate_bound(img, 180)
        else:
            img = cv2.resize(cv_image, (0, 0), fx=1 / 15., fy=1 / 15., interpolation=cv2.INTER_AREA)
            startrow = 2
            startcol = 27
        endcol = startcol + 64
        endrow = startrow + 64

        # crop image:
        img = img[startrow:endrow, startcol:endcol]
        assert img.shape == (64,64,3)
        return img


    def init_traj(self, itr):
        assert self.instance_type == 'main'
        # request init service for auxiliary recorders
        if self.use_aux:
            try:
                rospy.wait_for_service('init_traj', timeout=1)
                resp1 = self.init_traj_func(itr, self.igrp)
            except (rospy.ServiceException, rospy.ROSException), e:
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


        traj_folder = self.group_folder + '/traj{}'.format(itr)
        self.image_folder = traj_folder + '/images'
        self.depth_image_folder = traj_folder + '/depth_images'

        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)
        else:
            if not self.overwrite:
                raise ValueError("trajectory {} already exists".format(traj_folder))
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        if not os.path.exists(self.depth_image_folder):
            os.makedirs(self.depth_image_folder)

        if self.instance_type == 'main':
            self.state_action_data_file = traj_folder + '/joint_angles_traj{}.txt'.format(itr)
            self.state_action_pkl_file = traj_folder + '/joint_angles_traj{}.pkl'.format(itr)
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
            except (rospy.ServiceException, rospy.ROSException), e:
                rospy.logerr("Service call failed: %s" % (e,))
                raise ValueError('delete traj service failed')
        self._delete_traj_local(tr)

    def _delete_traj_local(self, i_tr):
        self.group_folder = self.save_dir + '/traj_group{}'.format(self.igrp)
        traj_folder = self.group_folder + '/traj{}'.format(i_tr)
        shutil.rmtree(traj_folder)
        print 'deleted {}'.format(traj_folder)

    def save(self, i_save, action, endeffector_pose):
        self.t_savereq = rospy.get_time()
        assert self.instance_type == 'main'

        if self.use_aux:
            # request save at auxiliary recorders
            try:
                rospy.wait_for_service('get_kinectdata', 0.1)
                resp1 = self.save_kinectdata_func(i_save)
            except (rospy.ServiceException, rospy.ROSException), e:
                rospy.logerr("Service call failed: %s" % (e,))
                raise ValueError('get_kinectdata service failed')

        if self.save_images:
            self._save_img_local(i_save)

        if self.save_actions:
            self._save_state_actions(i_save, action, endeffector_pose)

        if self.save_gif:
            highres = cv2.cvtColor(self.ltob.img_cv2, cv2.COLOR_BGR2RGB)
            print 'highres dim',highres.shape
            self.highres_imglist.append(highres)

    def save_highres(self):
        # clip = mpy.ImageSequenceClip(self.highres_imglist, fps=10)
        # clip.write_gif(self.image_folder + '/highres_traj{}.mp4'.format(self.itr))
        writer = imageio.get_writer(self.image_folder + '/highres_traj{}.mp4'.format(self.itr), fps=10)
        print 'shape highes:', self.highres_imglist[0].shape
        for im in self.highres_imglist:
            writer.append_data(im)
        writer.close()

    def get_aux_img(self):
        try:
            rospy.wait_for_service('get_kinectdata', 0.1)
            resp1 = self.get_kinectdata_func()
            self.ltob_aux1.img_msg = resp1.image
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            raise ValueError('get_kinectdata service failed')

    def _save_state_actions(self, i_save, action, endeff_pose):
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

        self.joint_angle_list.append(angles_right)
        self.action_list.append(action)
        self.cart_pos_list.append(endeff_pose)


        if i_save == self.state_sequence_length-1:
            joint_angles = np.stack(self.joint_angle_list)
            actions = np.stack(self.action_list)
            endeffector_pos = np.stack(self.cart_pos_list)

            with open(self.state_action_pkl_file, 'wb') as f:
                dict= {'jointangles': joint_angles,
                       'actions': actions,
                       'endeffector_pos':endeffector_pos}
                cPickle.dump(dict, f)
            self.action_list = []
            self.joint_angle_list = []
            self.cart_pos_list = []

    def _save_img_local(self, i_save):

        pref = self.instance_type

        #saving image
        # saving the full resolution image
        if self.ltob.img_cv2 is not None:
            image_name = self.image_folder+ "/" + pref + "_full_cropped_im{0}".format(str(i_save).zfill(2))
            image_name += "_time{1}.jpg".format(self.ltob.tstamp_img)

            cv2.imwrite(image_name, self.ltob.img_cv2, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        else:
            raise ValueError('img_cv2 no data received')

        # saving the cropped and downsized image
        if self.ltob.img_cropped is not None:
            image_name = self.image_folder + "/" + pref +"_cropped_im{0}_time{1}.png".format(i_save, self.ltob.tstamp_img)
            cv2.imwrite(image_name, self.ltob.img_cropped, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT,1])
            print 'saving small image to ', image_name
        else:
            raise ValueError('img_cropped no data received')

        # saving the depth data
        # saving the cropped depth data in a Pickle file
        if self.ltob.d_img_cropped_npy is not None:
            file = self.depth_image_folder + "/" + pref +"_depth_im{0}_time{1}.pkl".format(i_save, self.ltob.tstamp_d_img)
            cPickle.dump(self.ltob.d_img_cropped_npy, open(file, 'wb'))
        else:
            raise ValueError('d_img_cropped_npy no data received')

        # saving downsampled 8bit images
        if self.ltob.d_img_cropped_8bit is not None:
            image_name = self.depth_image_folder + "/" + pref + "_cropped_depth_im{0}_time{1}.png".format(i_save, self.ltob.tstamp_d_img)
            cv2.imwrite(image_name, self.ltob.d_img_cropped_8bit, [cv2.IMWRITE_PNG_STRATEGY_DEFAULT, 1])
        else:
            raise ValueError('d_img_cropped_8bit no data received')

        self.t_finish_save.append(rospy.get_time())
        if i_save == (self.state_sequence_length-1):
            with open(self.image_folder+'/{}_snapshot_timing.pkl'.format(pref), 'wb') as f:
                dict = {'t_finish_save': self.t_finish_save }
                if pref == 'aux1':
                    dict['t_get_request'] = self.t_get_request
                cPickle.dump(dict, f)


if __name__ ==  '__main__':
    print 'started'
    rec = RobotRecorder('/home/guser/Documents/sawyer_data/newrecording', seq_len=48)
