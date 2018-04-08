#!/usr/bin/env python
import numpy as np
import argparse
import tkinter as Tk
from tkinter import Button, Frame, Canvas, Scrollbar, Label
import tkinter.constants
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
import pdb

from python_visual_mpc import __file__ as base_filepath


class Visualizer(object):
    def __init__(self):
        rospy.init_node('visual_mpc_demo')
        rospy.loginfo("init node visual mpc demo")
        benchmark_name = rospy.get_param('~exp')
        self.ndesig = rospy.get_param('~ndesig')
        print('ndesig', self.ndesig)

        self.base_dir = '/'.join(str.split(base_filepath, '/')[:-2])
        cem_exp_dir = self.base_dir + '/experiments/cem_exp/benchmarks_sawyer'
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
        self.image_height = int(self.prediction_height * self.image_ratio)
        self.image_width = int(self.prediction_width * self.image_ratio)
        self.canvas_height = self.image_height + 100
        self.canvas_width = self.image_width + 100
        self.offset_y = (self.canvas_height - self.image_height) / 2
        self.offset_x = (self.canvas_width - self.image_width) / 2
        self.prediction_ratio = (self.prediction_height * self.image_ratio - 20) / (3. * self.prediction_height)
        self.prediction_length = self.netconf['sequence_length'] - 1

        # self.ndesig = args.ndesig

        self.num_pairs = 0
        # self.pairs = []
        self.startPixels = []
        self.goalPixels = []
        self.pixel1, self.pixel2 = None, None
        self.selPixels = True
        self.receivingPreds = True
        self.receivingDistribs = True
        self.receivingScores = True
        self.colors = ["#f11", "#05f", "#fb0"]

        self.bridge = CvBridge()

        self.display_publisher = rospy.Publisher('/robot/head_display', Image_msg, queue_size=2)
        self.visual_mpc_cmd_publisher = rospy.Publisher('visual_mpc_cmd', numpy_msg(intarray), queue_size=10)
        self.visual_mpc_reset_cmd_publisher = rospy.Publisher('visual_mpc_reset_cmd', numpy_msg(intarray), queue_size=10)
        rospy.Subscriber("/kinect2/hd/image_color", Image_msg, self.update_image)
        rospy.Subscriber('gen_image', numpy_msg(floatarray), self.update_pred_photos)
        rospy.Subscriber('gen_pix_distrib', numpy_msg(floatarray), self.update_distrib_photos)
        rospy.Subscriber('gen_score', numpy_msg(floatarray), self.update_score_texts)

        self.assetsdir = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-1]) + '/sawyer/visual_mpc_rospkg/src/assets'

        self.ready_splash = cv2.imread(self.assetsdir + '/mpc_start.png')
        self.exec_splash = cv2.imread(self.assetsdir + '/mpc_exec.png')
        self.publish_to_head(self.ready_splash)

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
        self.canvasPhoto = ImageTk.PhotoImage(Image.new("RGB", (self.image_width, self.image_height), "white"))
        self.canvas = Canvas(self.left_panel)
        self.canvas.bind("<Button-1>", self.input_pixel)
        self.canvas.grid(row=0, column=0, columnspan=3, sticky=tkinter.constants.NSEW)

        self.canvas.config(bg="white", width=self.canvas_width, height=self.canvas_height, borderwidth=0, highlightthickness=0)
        self.canvasImage = self.canvas.create_image(self.canvas_width / 2, self.canvas_height / 2, image=self.canvasPhoto)

        labelPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/label.png"))
        self.label = Label(self.right_panel, image=labelPhoto)
        self.label.image = labelPhoto
        self.label.grid(row=0, column=0, columnspan=2, pady=(50, 0), padx=(0, 50))
        self.label.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.predictionPhotos = []
        self.distributionPhotos = []
        self.scoreTexts = []
        self.predictions = []
        self.distributions = []
        self.scores = []

        self.emptyImage = Image.new("RGB", (int(self.prediction_width * self.prediction_ratio),
                                            int(self.prediction_height * self.prediction_ratio)), "white")
        self.emptyPhoto = ImageTk.PhotoImage(self.emptyImage)

        for i in range(self.num_predictions):
            predVideo = Label(self.right_panel, image=self.emptyPhoto)
            predVideo.image = self.emptyPhoto
            predVideo.grid(row=i+1, column=0, padx=(0, 10), pady=(10, 0))
            predVideo.config(bg="white")

            distributionPhotosCol = []
            distributionsCol = []
            for j in range(self.ndesig):
                distribVideo = Label(self.right_panel, image=self.emptyPhoto)
                distribVideo.image = self.emptyPhoto
                distribVideo.grid(row=i+1, column=j+1, padx=(0, 10), pady=(10, 0))
                distribVideo.config(bg="white")
                distributionPhotosCol.append([self.emptyImage])
                distributionsCol.append(distribVideo)

            score = Label(self.right_panel, text="", font=("Helvetica", 20))
            score.grid(row=i+1, column=3, padx=(50, 50), pady=(10, 0))
            score.config(bg="white")

            self.predictionPhotos.append([self.emptyImage])
            self.predictions.append(predVideo)

            self.distributionPhotos.append(distributionPhotosCol)
            self.distributions.append(distributionsCol)

            self.scoreTexts.append("")
            self.scores.append(score)

        self.predictions[0].grid(pady=(50, 0))
        # self.distributions[0].grid(pady=(50, 0))
        self.scores[0].grid(pady=(50, 0))
        self.predictions[-1].grid(pady=(10, 45))
        # self.distributions[-1].grid(pady=(10, 45))
        self.scores[-1].grid(pady=(10, 45))

        print(len(self.distributions), len(self.distributions[0]))
        for j in range(self.ndesig):
            self.distributions[0][j].grid(pady=(50, 0))
            self.distributions[-1][j].grid(pady=(10, 45))

        addPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/add.png"))
        self.addButton = Button(self.left_panel, image=addPhoto, command=self.begin_input)
        self.addButton.image = addPhoto
        self.addButton.grid(column=2, row=1, pady=(0, 50), padx=(0, 50), sticky=tkinter.constants.E)
        self.addButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)
        if self.ndesig == 1:
            self.addButton.config(state=tkinter.constants.DISABLED)

        startPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/start.png"))
        self.startButton = Button(self.left_panel, image=startPhoto, command=self.start)
        self.startButton.image = startPhoto
        self.startButton.grid(column=0, row=1, pady=(0, 50), padx=(50, 0), sticky=tkinter.constants.W)
        self.startButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0,
                                state=tkinter.constants.DISABLED)

        resetPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/reset.png"))
        self.resetButton = Button(self.left_panel, image=resetPhoto, command=self.reset_demo)
        self.resetButton.image = resetPhoto
        self.resetButton.grid(column=1, row=1, pady=(0, 50), sticky=tkinter.constants.W)
        self.resetButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.iter = 0
        self.video_loop()

    def video_loop(self):
        self.canvas.itemconfig(self.canvasImage, image=self.canvasPhoto)
        self.canvas.copy_image = self.canvasPhoto
        self.iter = (self.iter + 1) % len(self.predictionPhotos[0])
        for i in range(self.num_predictions):
            distributionPhotosCol = []
            if len(self.distributionPhotos[i][-1]) < self.prediction_length or len(self.predictionPhotos[i]) < self.prediction_length:
                predictionPhoto = self.emptyPhoto
                for j in range(self.ndesig):
                    distributionPhotosCol.append(self.emptyPhoto)
            else:
                predictionPhoto = ImageTk.PhotoImage(self.predictionPhotos[i][self.iter])
                for j in range(self.ndesig):
                    distributionPhotosCol.append(ImageTk.PhotoImage(self.distributionPhotos[i][j][self.iter]))
                self.scores[i].config(text=self.scoreTexts[i])
            self.predictions[i].config(image=predictionPhoto)
            self.predictions[i].image = predictionPhoto
            for j in range(self.ndesig):
                self.distributions[i][j].config(image=distributionPhotosCol[j])
                self.distributions[i][j].image = distributionPhotosCol[j]
        self.root.after(200, self.video_loop)

    def start(self):
        if self.num_pairs == 0:
            print("please select a pair of points")
        elif self.pixel1 and not self.pixel2:
            print("please select second pixel")
        else:
            print("starting")
            self.visual_mpc_cmd_publisher.publish(np.array(self.startPixels + self.goalPixels, dtype=np.uint32))
            self.publish_to_head(self.exec_splash)

    def reset_demo(self):
        # self.pairs = []
        self.startPixels = []
        self.goalPixels = []
        self.pixel1, self.pixel2 = None, None
        self.selPixels = True
        self.canvas.delete("points")
        self.receivingPreds = True
        self.receivingDistribs = True
        self.receivingScores = True
        self.predictionPhotos = []
        self.distributionPhotos = []
        self.scoreTexts = []
        self.num_pairs = 0
        for i in range(self.num_predictions):
            self.predictionPhotos.append([self.emptyImage])
            distributionPhotosCol = []
            for j in range(self.ndesig):
                distributionPhotosCol.append([self.emptyImage])
            self.distributionPhotos.append(distributionPhotosCol)
            self.scoreTexts.append("")
            self.scores[i].config(text="")

        for i in range(self.num_predictions):
            for j in range(self.ndesig):
                self.distributions[i][j].config(highlightthickness=0)

        self.startButton.config(state=tkinter.constants.DISABLED)
        if self.ndesig == 1:
            self.addButton.config(state=tkinter.constants.DISABLED)
        else:
            self.addButton.config(state=tkinter.constants.NORMAL)
        self.visual_mpc_reset_cmd_publisher.publish(np.array([1]))
        self.publish_to_head(self.ready_splash)

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
        cv_image = cv_image[rowstart * self.image_ratio : (rowstart + self.prediction_height) * self.image_ratio,
                            colstart * self.image_ratio : (colstart + self.prediction_width) * self.image_ratio]
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        self.canvasPhoto = ImageTk.PhotoImage(pil_image)

    def update_pred_photos(self, data):
        if self.receivingPreds:
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
        else:
            self.receivingPreds = True

    def update_distrib_photos(self, data):
        if self.receivingDistribs:
            self.receivingDistribs = False
            data = data.data.reshape((self.num_predictions, self.prediction_length, self.num_pairs,
                                      self.prediction_height, self.prediction_width))
            for i in range(self.num_predictions):
                for j in range(self.ndesig):
                    tempPhotos = []
                    for k in range(self.prediction_length):
                        renormalized_data = data[i, k, j]/np.max(data[i, k, j])
                        colored_distrib = 255 * self.cmap(np.squeeze(renormalized_data))[:, :, :3]
                        colored_distrib = colored_distrib.astype(np.uint8)
                        pil_image = Image.fromarray(colored_distrib).resize([int(self.prediction_width * self.prediction_ratio),
                                                                             int(self.prediction_height * self.prediction_ratio)],
                                                                            resample=Image.LANCZOS)
                        tempPhotos.append(pil_image)
                    self.distributionPhotos[i][j] = tempPhotos

            for i in range(self.num_predictions):
                for j in range(self.ndesig):
                    self.distributions[i][j].config(highlightbackground=self.colors[j], highlightthickness=2)
        else:
            self.receivingDistribs = True

    def update_score_texts(self, data):
        if self.receivingScores:
            self.receivingScores = False
            data = data.data
            for i in range(self.num_predictions):
                self.scoreTexts[i] = str(data[i])
        else:
            self.receivingScores = True

    def input_pixel(self, event):
        if self.selPixels and event.x >= self.offset_x and event.y >= self.offset_y and event.x <= self.offset_x + self.image_width and event.y <= self.offset_y + self.image_height:
            x = int(round((self.canvas.canvasx(event.x) - self.offset_x) / self.image_ratio))
            y = int(round((self.canvas.canvasy(event.y) - self.offset_y) / self.image_ratio))

            if self.pixel1:
                self.canvas.create_oval(event.x - 10, event.y - 10, event.x + 10, event.y + 10,
                                        outline=self.colors[self.num_pairs % len(self.colors)],
                                        width=8,
                                        tags="points")

                print("pixel 2: ", y, x)
                self.pixel2 = [y, x]
                # self.selPixels = False

                self.goalPixels.extend(self.pixel2)

                # self.pairs.extend(self.pixel1)
                # self.pairs.extend(self.pixel2)
                self.num_pairs += 1
                self.pixel1 = None
                self.pixel2 = None

                if self.num_pairs == self.ndesig:
                    self.startButton.config(state=tkinter.constants.NORMAL)
                    self.selPixels = False
            else:
                self.canvas.create_oval(event.x - 10, event.y - 10, event.x + 10, event.y + 10,
                                        outline=self.colors[self.num_pairs % len(self.colors)],
                                        fill=self.colors[self.num_pairs % len(self.colors)],
                                        width=8,
                                        tags="points")

                print("pixel 1: ", y, x)
                self.pixel1 = [y, x]

                self.startPixels.extend(self.pixel1)
                self.startButton.config(state=tkinter.constants.DISABLED)

    def begin_input(self):
        print("ready for inputs")
        # self.selPixels = True
        if self.pixel1 and not self.pixel2:
            print("please select second pixel")
        if self.num_pairs + 1 >= self.ndesig:
            self.addButton.config(state=tkinter.constants.DISABLED)

    def publish_to_head(self, img):
        num_rows, num_cols = img.shape[:2]
        rotation_matrix = np.array([[0., 1., 0., ], [-1., 0., 1024., ]])
        img = cv2.warpAffine(img, rotation_matrix, (num_rows, num_cols))[:, ::-1, :]
        img = np.swapaxes(img, 0, 1)[::-1, ::-1, :]
        # I don't know why the transforms above work either
        img_message = self.bridge.cv2_to_imgmsg(img)
        self.display_publisher.publish(img_message)


if __name__ == '__main__':
    v = Visualizer()