#!/usr/bin/env python
import numpy as np
import argparse
import Tkinter as Tk
from Tkinter import Button, Frame, Canvas, Scrollbar, Label
import Tkconstants
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import cv2
import copy
import os
import imp
import imutils
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from visual_mpc_rospkg.msg import intarray, floatarray
from sensor_msgs.msg import Image as Image_msg
import rospy
import python_visual_mpc

from python_visual_mpc import __file__ as base_filepath

IMAGE_WIDTH = 64 * 12
IMAGE_HEIGHT = 56 * 12
CANVAS_WIDTH = IMAGE_WIDTH + 100
CANVAS_HEIGHT = IMAGE_HEIGHT + 100
RATIO = IMAGE_WIDTH / 64
OFFSET_X = (CANVAS_WIDTH - IMAGE_WIDTH) / 2
OFFSET_Y = (CANVAS_HEIGHT - IMAGE_HEIGHT) / 2
COLORS = ["#f11", "#fb0", "#05f"]

SEQUENCE_LENGTH = 14


class Visualizer(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Run benchmarks')
        parser.add_argument('benchmark', type=str, help='the name of the folder with agent setting for the benchmark')
        parser.add_argument('ndesig', type=int, default=1, help='allowed number of designated points')

        args = parser.parse_args()

        self.base_dir = '/'.join(str.split(base_filepath, '/')[:-2])
        cem_exp_dir = self.base_dir + '/experiments/cem_exp/benchmarks_sawyer'
        benchmark_name = args.benchmark
        bench_dir = cem_exp_dir + '/' + benchmark_name
        if not os.path.exists(bench_dir):
            raise ValueError('benchmark directory does not exist')
        bench_conf = imp.load_source('mod_hyper', bench_dir + '/mod_hyper.py')
        self.policyparams = bench_conf.policy
        self.agentparams = bench_conf.agent
        hyperparams = imp.load_source('hyperparams', self.policyparams['netconf'])
        self.netconf = hyperparams.configuration
        dataconf_file = self.base_dir + '/'.join(str.split(self.netconf['data_dir'], '/')[:-1]) + '/conf.py'
        self.dataconf = imp.load_source('hyperparams', dataconf_file).configuration

        self.num_predictions = 3
        self.image_ratio = int((1. / self.dataconf['shrink_before_crop']) / 0.75)
        if 'img_height' in self.netconf and 'img_width' in self.netconf:
            self.prediction_height = self.netconf['img_height']
            self.prediction_width = self.netconf['img_width']
        else:
            self.prediction_height = 64
            self.prediction_width = 64
        self.prediction_ratio = (self.prediction_height * self.image_ratio - 20) / (3. * self.prediction_height)
        print self.prediction_height, self.image_ratio, self.prediction_ratio

        self.prediction_length = self.netconf['sequence_length'] - 1

        self.ndesig = args.ndesig

        self.num_pairs = 0
        self.pairs = []
        self.pixel1, self.pixel2 = None, None
        self.selPixels = True
        self.receivingPreds = True
        self.receivingDistribs = True

        self.bridge = CvBridge()
        rospy.init_node('visual_mpc_demo')
        rospy.loginfo("init node visual mpc demo")
        self.visual_mpc_cmd_publisher = rospy.Publisher('visual_mpc_cmd', numpy_msg(intarray), queue_size=10)
        self.visual_mpc_reset_cmd_publisher = rospy.Publisher('visual_mpc_reset_cmd', numpy_msg(intarray), queue_size=10)
        rospy.Subscriber("main/kinect2/hd/image_color", Image_msg, self.update_image)
        rospy.Subscriber('gen_image', numpy_msg(floatarray), self.update_pred_photos)
        rospy.Subscriber('gen_pix_distrib', numpy_msg(floatarray), self.update_distrib_photos)

        self.assetsdir = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-1]) + '/sawyer/visual_mpc_rospkg/src/assets'

        self.cmap = plt.cm.get_cmap('jet')

        self.root = Tk.Tk()
        self.root.config(bg="white")

        self.left_panel = Frame(self.root)
        self.left_panel.grid(row=0, column=0)
        self.left_panel.config(bg="white")
        self.right_panel = Frame(self.root)
        self.right_panel.grid(row=0, column=1)
        self.right_panel.config(bg="white")

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        self.canvasPhoto = ImageTk.PhotoImage(Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white"))
        self.canvas = Canvas(self.left_panel)
        self.canvas.bind("<Button-1>", self.input_pixel)
        self.canvas.grid(row=0, column=0, columnspan=3, sticky=Tkconstants.NSEW)

        self.canvas.config(bg="white", width=CANVAS_WIDTH, height=CANVAS_HEIGHT, borderwidth=0, highlightthickness=0)
        self.canvasImage = self.canvas.create_image(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, image=self.canvasPhoto)

        labelPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/label.png"))
        self.label = Label(self.right_panel, image=labelPhoto)
        self.label.image = labelPhoto
        self.label.grid(row=0, column=0, columnspan=2, pady=(50, 0), padx=(0, 50))
        self.label.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.predictionPhotos = []
        self.distributionPhotos = []
        self.predictions = []
        self.distributions = []

        self.emptyImage = Image.new("RGB", (int(self.prediction_width * self.prediction_ratio),
                                            int(self.prediction_height * self.prediction_ratio)), "white")
        self.emptyPhoto = ImageTk.PhotoImage(self.emptyImage)

        for i in range(self.num_predictions):
            predVideo = Label(self.right_panel, image=self.emptyPhoto)
            predVideo.image = self.emptyPhoto
            predVideo.grid(row=i+1, column=0, padx=(0, 10), pady=(10, 0))
            predVideo.config(bg="white")

            distribVideo = Label(self.right_panel, image=self.emptyPhoto)
            distribVideo.image = self.emptyPhoto
            distribVideo.grid(row=i+1, column=1, padx=(0, 50), pady=(10, 0))
            distribVideo.config(bg="white")

            self.predictionPhotos.append([self.emptyImage])
            self.predictions.append(predVideo)

            self.distributionPhotos.append([self.emptyImage])
            self.distributions.append(distribVideo)

        self.predictions[0].grid(pady=(50, 0))
        self.distributions[0].grid(pady=(50, 0))
        self.predictions[-1].grid(pady=(10, 45))
        self.distributions[-1].grid(pady=(10, 45))

        addPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/add.png"))
        self.addButton = Button(self.left_panel, image=addPhoto, command=self.begin_input)
        self.addButton.image = addPhoto
        self.addButton.grid(column=2, row=1, pady=(0, 50), padx=(0, 50), sticky=Tkconstants.E)
        self.addButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)
        if self.ndesig == 1:
            self.addButton.config(state=Tkconstants.DISABLED)

        startPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/start.png"))
        self.startButton = Button(self.left_panel, image=startPhoto, command=self.start)
        self.startButton.image = startPhoto
        self.startButton.grid(column=0, row=1, pady=(0, 50), padx=(50, 0), sticky=Tkconstants.W)
        self.startButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0,
                                state=Tkconstants.DISABLED)

        resetPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/reset.png"))
        self.resetButton = Button(self.left_panel, image=resetPhoto, command=self.reset_demo)
        self.resetButton.image = resetPhoto
        self.resetButton.grid(column=1, row=1, pady=(0, 50), sticky=Tkconstants.W)
        self.resetButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.iter = 0
        self.video_loop()

    def video_loop(self):
        self.canvas.itemconfig(self.canvasImage, image=self.canvasPhoto)
        self.canvas.copy_image = self.canvasPhoto
        self.iter = (self.iter + 1) % len(self.predictionPhotos[0])
        for i in range(self.num_predictions):
            if len(self.distributionPhotos[i]) < self.prediction_length or len(self.predictionPhotos[i]) < self.prediction_length:
                predictionPhoto = self.emptyPhoto
                distributionPhoto = self.emptyPhoto
            else:
                predictionPhoto = ImageTk.PhotoImage(self.predictionPhotos[i][self.iter])
                distributionPhoto = ImageTk.PhotoImage(self.distributionPhotos[i][self.iter])
            self.predictions[i].config(image=predictionPhoto)
            self.predictions[i].image = predictionPhoto
            self.distributions[i].config(image=distributionPhoto)
            self.distributions[i].image = distributionPhoto
        self.root.after(200, self.video_loop)

    def start(self):
        if self.num_pairs == 0:
            print "please select a pair of points"
        elif self.pixel1 and not self.pixel2:
            print "please select second pixel"
        else:
            print "starting"
            self.visual_mpc_cmd_publisher.publish(np.array(self.pairs, dtype=np.uint32))

    def reset_demo(self):
        self.pairs = []
        self.pixel1, self.pixel2 = None, None
        self.selPixels = True
        self.canvas.delete("points")
        self.receivingPreds = True
        self.receivingDistribs = True
        self.predictionPhotos = []
        self.distributionPhotos = []
        self.num_pairs = 0
        for i in range(self.num_predictions):
            self.predictionPhotos.append([self.emptyImage])
            self.distributionPhotos.append([self.emptyImage])

        self.startButton.config(state=Tkconstants.DISABLED)
        if self.ndesig == 1:
            self.addButton.config(state=Tkconstants.DISABLED)
        else:
            self.addButton.config(state=Tkconstants.NORMAL)
        self.visual_mpc_reset_cmd_publisher.publish(np.array([1]))

    def update_image(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        colstart = 180
        rowstart = 0
        endcol = colstart + 1500
        endrow = rowstart + 1500
        cv_image = copy.deepcopy(cv_image[rowstart:endrow, colstart:endcol])
        cv_image = imutils.rotate_bound(cv_image, 180)

        rowstart = self.dataconf['rowstart']
        colstart = self.dataconf['colstart']
        cv_image = cv_image[rowstart * 12 : (rowstart + 56) * 12, colstart * 12 : (colstart + 64) * 12]
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        self.canvasPhoto = ImageTk.PhotoImage(pil_image)

    def update_pred_photos(self, data):
        if self.receivingPreds:
            print "receiving p"
            self.receivingPreds = False
            data = 255 * data.data.reshape((self.num_predictions, self.prediction_length, self.prediction_height,
                                            self.prediction_width, 3))
            data = data.astype(np.uint8)
            for i in range(self.num_predictions):
                tempPhotos = []
                for j in range(self.prediction_length):
                    pil_image = Image.fromarray(data[i, j]).resize([int(self.prediction_width * self.prediction_ratio),
                                                                    int(self.prediction_height * self.prediction_ratio)],
                                                                   resample=Image.LANCZOS)
                    tempPhotos.append(pil_image)
                self.predictionPhotos[i] = tempPhotos
            print "processed p"
        else:
            self.receivingPreds = True

    def update_distrib_photos(self, data):
        if self.receivingDistribs:
            print "receiving d"
            self.receivingDistribs = False
            data = data.data.reshape((self.num_pairs, self.num_predictions, self.prediction_length,
                                      self.prediction_height, self.prediction_width))
            for i in range(self.num_predictions):
                tempPhotos = []
                for j in range(self.prediction_length):
                    colored_distrib = 255 * self.cmap(np.squeeze(data[0, i, j]))[:, :, :3]
                    colored_distrib = colored_distrib.astype(np.uint8)
                    pil_image = Image.fromarray(colored_distrib).resize([int(self.prediction_width * self.prediction_ratio),
                                                                         int(self.prediction_height * self.prediction_ratio)],
                                                                        resample=Image.LANCZOS)
                    tempPhotos.append(pil_image)
                self.distributionPhotos[i] = tempPhotos
            print "processed d"
        else:
            self.receivingDistribs = True

    def input_pixel(self, event):
        if self.selPixels and event.x >= OFFSET_X and event.y >= OFFSET_Y and event.x <= OFFSET_X + IMAGE_WIDTH and event.y <= OFFSET_Y + IMAGE_HEIGHT:
            x = int(round((self.canvas.canvasx(event.x) - OFFSET_X) / RATIO))
            y = int(round((self.canvas.canvasy(event.y) - OFFSET_Y) / RATIO))

            if self.pixel1:
                self.canvas.create_oval(event.x - 6, event.y - 6, event.x + 6, event.y + 6,
                                        outline=COLORS[self.num_pairs % len(COLORS)],
                                        width=4,
                                        tags="points")

                print "pixel 2: ", y, x
                self.pixel2 = [y, x]
                self.selPixels = False

                self.pairs.extend(self.pixel1)
                self.pairs.extend(self.pixel2)
                self.num_pairs += 1
                self.pixel1 = None
                self.pixel2 = None

                self.startButton.config(state=Tkconstants.NORMAL)
            else:
                self.canvas.create_oval(event.x - 6, event.y - 6, event.x + 6, event.y + 6,
                                        outline=COLORS[self.num_pairs % len(COLORS)],
                                        fill=COLORS[self.num_pairs % len(COLORS)],
                                        width=4,
                                        tags="points")

                print "pixel 1: ", y, x
                self.pixel1 = [y, x]
                self.startButton.config(state=Tkconstants.DISABLED)

    def begin_input(self):
        print "ready for inputs"
        self.selPixels = True
        if self.pixel1 and not self.pixel2:
            print "please select second pixel"
        if self.num_pairs + 1 >= self.ndesig:
            self.addButton.config(state=Tkconstants.DISABLED)


if __name__ == '__main__':
    v = Visualizer()