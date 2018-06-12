try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
except ImportError:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
from PIL import Image
from sys import argv

import argparse
import numpy as np
import sys
################3333
from python_visual_mpc.region_proposal_networks.Featurizer import BBProposer, AlexNetFeaturizer
import cv2
import pdb
import os

import glob
import moviepy.editor as mpy

def make_gif(im_list):
    clip = mpy.ImageSequenceClip(im_list, fps=4)
    clip.write_gif('modeldata/tracking.gif')

class Too_few_objects_found_except(Exception):
    pass

class RPN_Tracker(object):
    def __init__(self, savedir= None, recorder=None):
        self.savedir = savedir
        self.pix2boxes = {}
        self.feats = {}
        self.proposer = BBProposer()
        self.featurizer = AlexNetFeaturizer()

        self.recorder = recorder

        self.sel_feature_vec = None

    def get_regions(self, image):
        self.clone = np.array(image)[:, :, :3].astype(np.float64)
        self.clone[:,:] -= np.array([122.7717, 102.9801, 115.9465 ])
        self.boxes = [tuple(b) for b in self.proposer.extract_proposal(self.clone)]
        self.clone[:,:] += np.array([122.7717, 102.9801, 115.9465 ])
        crops = [self.proposer.get_crop(b, self.clone) for b in self.boxes]
        self.feats = {b: self.featurizer.getFeatures(c) for b,c in zip(self.boxes,crops)}

    def plot(self, interactive = False):
        window = tkinter.Tk(className="Image")
        image = Image.fromarray(np.uint8(self.clone))
        canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
        canvas.pack()
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvasimage = canvas.create_image(image.size[0]//2,
                                               image.size[1]//2,
                                               image=self.image_tk)

        if interactive:
            canvas.bind("<Button-1>", self.click)
            self.canvas = canvas
            self.canvas.bind("<Right>", self.nextbox)
            self.canvas.bind("<Left>", self.prevbox)
            self.canvas.bind("<Return>", self.savefeats)
            self.canvas.focus_set()
            self.canvas.pack()

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

    def get_boxes(self, image, valid_box = np.array([170,80,450,280]), im_save_dir = None):

        self.get_regions(image)
        boxes = self.boxes

        valid_region = valid_box

        valid_boxes = []
        valid_center_coords = []

        min_heigh_or_width = 50
        for b in boxes:
            center_coord = np.array([(b[0] + b[2]) / 2.,  # col, row
                                     (b[1] + b[3]) / 2.])

            box_height = b[3] - b[1]
            box_width = b[2] - b[0]

            if  valid_region[0] < center_coord[0] < valid_region[2] and \
                valid_region[1] < center_coord[1] < valid_region[3] and \
                    (box_height > min_heigh_or_width or box_width > min_heigh_or_width):
                valid_boxes.append(b)
                valid_center_coords.append(center_coord)

        if im_save_dir is not None:
            import rospy
            for b in valid_boxes:
                self.proposer.draw_box(b, self.clone, 0)  # red

            self.proposer.draw_box(list(valid_region), self.clone, 2)  # blue

            im = Image.fromarray(np.array(self.clone).astype(np.uint8))
            im.save(im_save_dir + '/task_{}.png'.format(rospy.get_time()))
        return valid_center_coords

    def get_task(self, image, im_save_dir):
        """
        :param image:
        :return:
        box format is x1, y1, x2, y2 = colstart, rowstart, colend, rowend
        """

        self.get_regions(image)
        boxes = self.boxes

        # for b in boxes:
        #     self.proposer.draw_box(b, self.clone, 0)  # red

        # valid_region = np.array([340,210,830,570])  #big
        valid_region = np.array([340,240,780,520])  #small
        self.proposer.draw_box(list(valid_region), self.clone, 2)  # blue

        valid_boxes = []
        valid_center_coords = []

        min_heigh_or_width = 50
        for b in boxes:
            center_coord = np.array([(b[0] + b[2])/2.,   #col, row
                                     (b[1] + b[3])/2.])

            box_height = b[3] - b[1]
            box_width = b[2] - b[0]

            if  valid_region[0] < center_coord[0] < valid_region[2] and \
                valid_region[1] < center_coord[1] < valid_region[3] and \
                    (box_height > min_heigh_or_width or box_width > min_heigh_or_width):
                print('box', b)
                print('h', box_height)
                print('w', box_width)
                valid_boxes.append(b)
                valid_center_coords.append(center_coord)

        for b in valid_boxes:
            self.proposer.draw_box(b, self.clone, 0)  # red

        if len(valid_boxes) < 4:
            raise Too_few_objects_found_except

        sel_ind = np.random.choice(list(range(len(valid_boxes))))
        desig_point = valid_center_coords[sel_ind]
        self.proposer.draw_box(valid_boxes[sel_ind], self.clone, 1)

        while True:
            disp_vec = np.random.uniform(-300,300,2)

            goal_point = desig_point + disp_vec
            if  valid_region[0] < goal_point[0] < valid_region[2] and \
                valid_region[1] < goal_point[1] < valid_region[3]:
                break

        self.draw_cross(desig_point, self.clone)
        self.draw_cross(goal_point, self.clone)

        # self.plot()
        import rospy
        im  = Image.fromarray(np.array(self.clone).astype(np.uint8))
        im.save(im_save_dir + '/task_{}.png'.format(rospy.get_time()))

        #convert format x, y to r,c
        desig_point = np.array([desig_point[1], desig_point[0]])
        goal_point = np.array([desig_point[1], goal_point[0]])

        desig_point = self.recorder.high_res_to_lowres(desig_point)
        goal_point = self.recorder.high_res_to_lowres(goal_point)

        return desig_point, goal_point

    ### for tracking:
    def test_tracking(self, files):

        im_list = []
        for t in range(95):
            [imfile] = glob.glob(files + "/main_full_cropped_im{}_*.jpg".format(str(t).zfill(2)))
            img = np.array(Image.open(imfile))

            self.get_regions(img)
            if t == 0:
                for y in range(self.clone.shape[0]):
                    for x in range(self.clone.shape[1]):
                        self.pix2boxes[(x, y)] = []
                for b in self.boxes:
                    x1, y1, x2, y2 = (int(bi) for bi in b)
                    for y in range(y1, y2):
                        for x in range(x1, x2):
                            self.pix2boxes[(x, y)].append(b)
                    self.proposer.draw_box(b, self.clone, 0)

                self.plot(interactive=True)
                tkinter.mainloop()
            else:
                self.get_regions(img)
                dist_box_dict = {}
                dists = []
                for box in list(self.feats.keys()):
                    dist = np.linalg.norm(self.sel_feature_vec - self.feats[box])
                    dist_box_dict[dist] = box
                    dists.append(dist)

                least_dist = min(dists)
                best_box = dist_box_dict[least_dist]

                self.proposer.draw_box(best_box, self.clone, 1)
                cv2.imshow("Tracking", cv2.cvtColor(self.clone.astype(np.uint8), cv2.COLOR_BGR2RGB))
                k = cv2.waitKey(1) & 0xff
                if k == 27: break

                small_rgb = cv2.resize(self.clone, (0, 0), fx=0.6, fy=0.6)
                if t % 5 == 0:
                    im_list.append(small_rgb)

        make_gif(im_list)

    def drawbox(self,box):
        self.proposer.draw_box(box, self.clone, 1)
        image = Image.fromarray(np.uint8(self.clone))
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvasimage, image=self.image_tk)

    def refreshboxes(self):
        for b in self.boxes:
            self.proposer.draw_box(b, self.clone, 0)
        image = Image.fromarray(np.uint8(self.clone))
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.canvasimage, image=self.image_tk)

    def nextbox(self, event):
        self.listpos+=1
        if self.listpos == len(self.boxlist):
            self.listpos = 0
        box = self.boxlist[self.listpos]
        self.refreshboxes()
        self.drawbox(box)

    def prevbox(self, event):
        self.listpos-=1
        if self.listpos == -1:
            self.listpos = len(self.boxlist)-1
        box = self.boxlist[self.listpos]
        self.refreshboxes()
        self.drawbox(box)

    def savefeats(self, event):
        print("You selected the box at", self.boxlist[self.listpos])
        self.sel_feature_vec = self.feats[self.boxlist[self.listpos]]

    def click(self,event):
        x,y = event.x, event.y
        self.refreshboxes()
        self.boxlist = self.pix2boxes[(x,y)]
        self.listpos = 0
        if len(self.boxlist) > 0:
            print("There are", len(self.boxlist), "boxes around this pixel.")
            self.drawbox(self.boxlist[self.listpos])

        else:
            print("No box was selected, please try again")



if __name__ == '__main__':
    r = RPN_Tracker()
    import python_visual_mpc
    image_files = os.path.dirname(python_visual_mpc.__file__) + '/flow/sparse_tracking/testdata/img2'
    r.test_tracking(image_files)
