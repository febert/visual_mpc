import Tkinter
from PIL import Image, ImageTk
from sys import argv

import argparse
import numpy as np
import sys
from python_visual_mpc.region_proposal_networks.Featurizer import BBProposer, AlexNetFeaturizer

import pdb

class RPN_Tracker(object):
    def __init__(self):
        self.pix2boxes = {}
        self.feats = {}
        self.proposer = BBProposer()
        self.featurizer = AlexNetFeaturizer()

    def get_regions(self, image):
        self.clone = np.array(image)[:, :, :3].astype(np.float64)
        self.clone[:,:] -= np.array([122.7717, 102.9801, 115.9465 ])
        boxes = [tuple(b) for b in self.proposer.extract_proposal(self.clone)]
        self.clone[:,:] += np.array([122.7717, 102.9801, 115.9465 ])
        crops = [self.proposer.get_crop(b, self.clone) for b in boxes]
        feats = {b: self.featurizer.getFeatures(c) for b,c in zip(boxes,crops)}



        return boxes, feats


    def plot(self):
        window = Tkinter.Tk(className="Image")
        image = Image.fromarray(np.uint8(self.clone))
        canvas = Tkinter.Canvas(window, width=image.size[0], height=image.size[1])
        canvas.pack()
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvasimage = canvas.create_image(image.size[0]//2,
                                               image.size[1]//2,
                                               image=self.image_tk)


    def draw_cross(self, coord, im):
        """
        :param coord: x,y coordiante
        :param im:
        :return:
        """
        coord = coord.astype(np.int)
        r = coord[1]
        c = coord[0]

        h = np.array([50])
        rmin = r - h
        rmax = r + h
        cmin = c - h
        cmax = c + h


        img_h = im.shape[0]
        img_w = im.shape[1]
        np.clip(rmin, 0, img_h)
        np.clip(rmax, 0, img_h)
        np.clip(cmin, 0, img_w)
        np.clip(cmax, 0, img_w)

        color = np.array([0, 255, 255])
        im[int(rmin):int(rmax), c] = color
        im[r, int(cmin):int(cmax)] = color

        return im

    def get_task(self, image):
        """
        :param image:
        :return:
        box format is x1, y1, x2, y2 = colstart, rowstart, colend, rowend
        """


        boxes, feats = self.get_regions(image)

        # for b in boxes:
        #     self.proposer.draw_box(b, self.clone, 0)  # red

        valid_region = np.array([300,180,850,650])
        self.proposer.draw_box(list(valid_region), self.clone, 2)  # blue

        valid_boxes = []
        valid_center_coords = []
        for b in boxes:
            center_coord = np.array([(b[0] + b[2])/2.,   #col, row
                                     (b[1] + b[3])/2.])
            if  valid_region[0] < center_coord[0] < valid_region[2] and \
                valid_region[1] < center_coord[1] < valid_region[3]:
                valid_boxes.append(b)
                valid_center_coords.append(center_coord)

        # for b in valid_boxes:
        #     self.proposer.draw_box(b, self.clone, 1)  # green

        sel_ind = np.random.choice(range(len(valid_boxes)))
        desig_point = valid_center_coords[sel_ind]
        self.proposer.draw_box(valid_boxes[sel_ind], self.clone, 1)

        while True:
            disp_vec = np.random.uniform(-100,100,2)

            goal_point = desig_point + disp_vec
            if  valid_region[0] < goal_point[0] < valid_region[2] and \
                valid_region[1] < goal_point[1] < valid_region[3]:
                break


        self.draw_cross(desig_point, self.clone)
        self.draw_cross(goal_point, self.clone)

        self.plot()
        #TODO: filter out boxes