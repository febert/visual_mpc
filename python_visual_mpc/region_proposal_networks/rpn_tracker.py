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

        for y in range(self.clone.shape[0]):
            for x in range(self.clone.shape[1]):
                self.pix2boxes[(x,y)] = []
        for b in boxes:
            self.proposer.draw_box(b, self.clone, 0)

        window = Tkinter.Tk(className="Image")
        image = Image.fromarray(np.uint8(self.clone))
        canvas = Tkinter.Canvas(window, width=image.size[0], height=image.size[1])
        canvas.pack()
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvasimage = canvas.create_image(image.size[0]//2,
                                               image.size[1]//2,
                                               image=self.image_tk)

        return boxes, feats

    def get_task(self, image):

        boxes, feats = self.get_regions(image)
        pdb.set_trace()
        #TODO: filter out boxes