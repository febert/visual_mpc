""" This file defines the sample class. """
import numpy as np

class Trajectory(object):
    def __init__(self, conf):
        self.T = conf['T']

        img_channels = 3

        if conf is None:
            img_height = 64
            img_width = 64
        else:
            img_height = conf['image_height']
            img_width = conf['image_width']

        if 'cameras' in conf:
            self.images = np.zeros((self.T, len(conf['cameras']),
                                    img_height,
                                    img_width,
                                    img_channels), dtype='uint8')
        else:
            self.images = np.zeros((self.T,
                                    img_height,
                                    img_width,
                                    img_channels), dtype='uint8')

        if 'image_medium' in conf:
            self._image_medium = np.zeros((self.T,
                                            conf['image_medium'][0],
                                            conf['image_medium'][1],
                                            img_channels), dtype='uint8')
        if 'first_last_noarm' in conf:
            self.first_last_noarm = np.zeros((2,img_height,
                                                img_width,
                                                img_channels), dtype='uint8')

        # for storing the terminal predicted images of the K best actions at each time step:
        self.final_predicted_images = []
        self.predicted_images = None
        self.gtruth_images = None

        if 'adim' in conf:
            self.actions = np.zeros([self.T, conf['adim']])
        else:
            self.actions = np.zeros([self.T, 2])

        if 'sdim' in conf:
            state_dim = conf['sdim']
        else:
            state_dim = 2

        self.X_full = np.zeros([self.T, state_dim//2])
        self.Xdot_full = np.zeros([self.T, state_dim//2])
        self.X_Xdot_full = np.zeros([self.T, state_dim])

        if 'posmode' in conf:
            self.target_qpos = np.zeros([self.T + 1, conf['adim']])

        if 'num_objects' in conf:
            self.Object_pose = np.zeros([self.T, conf['num_objects'], 3])  # x,y rot of  block
            self.Object_full_pose = np.zeros([self.T, conf['num_objects'], 7])  # xyz and quaternion pose

        self.desig_pos = np.zeros([self.T, 2])
        self.score = np.zeros([self.T])

        self.goal_mask = None

        if 'make_gtruth_flows' in conf:
            self.large_ob_masks = np.zeros([self.T, conf['num_objects'], conf['viewer_image_height'], conf['viewer_image_width']])  # x,y rot of  block
            self.large_arm_masks = np.zeros([self.T, conf['viewer_image_height'],
                                            conf['viewer_image_width']])  # x,y rot of  block

            self.ob_masks = np.zeros([self.T, conf['num_objects'], conf['image_height'],
                                             conf['image_width']])  # x,y rot of  block
            self.arm_masks = np.zeros([self.T, conf['image_height'],
                                             conf['image_width']])  # x,y rot of  block

            self.largeimage = np.zeros((self.T,
                                        conf['viewer_image_height'],
                                        conf['viewer_image_width'], 3), dtype='uint8')

            self.largedimage = np.zeros((self.T,
                                         conf['viewer_image_height'],
                                         conf['viewer_image_width']), dtype=np.float32)

            self.bwd_flow = np.zeros((self.T-1,
                                        conf['image_height'],
                                        conf['image_width'], 2), dtype=np.float32)


        # world coordinates including the arm
        if 'num_objects' in conf:
            self.obj_world_coords = np.zeros([self.T, conf['num_objects'] + 1, 7])  # xyz and quaternion pose

        self.plan_stat_l = []  # statistics about the plan

        self.goal_dist = []

