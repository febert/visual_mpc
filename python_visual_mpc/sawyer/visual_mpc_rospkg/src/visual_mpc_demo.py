#!/usr/bin/env python
import numpy as np
from matplotlib import animation
import Tkinter as Tk
from Tkinter import Button, Frame, Canvas, Scrollbar, Label
import Tkconstants
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import cv2
import imutils
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from visual_mpc_rospkg.msg import intarray, floatarray
from sensor_msgs.msg import Image as Image_msg
import rospy
import python_visual_mpc

CANVAS_WIDTH = 1144
CANVAS_HEIGHT = 888
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
RATIO = IMAGE_WIDTH / 64
PREDICTION_WIDTH = 256
PREDICTION_HEIGHT = 192
OFFSET_X = (CANVAS_WIDTH - IMAGE_WIDTH) / 2
OFFSET_Y = (CANVAS_HEIGHT - IMAGE_HEIGHT) / 2
COLORS = ["#f11", "#fb0", "#05f"]

SEQUENCE_LENGTH = 14


class Visualizer(object):
    def __init__(self):
        self.num_pairs = 0
        self.pairs = []
        self.pixel1, self.pixel2 = None, None
        self.selPixels = False
        self.receivingPreds = True
        self.receivingDistribs = True

        self.prediction_length = 14
        self.prediction_width = 256
        self.prediction_height = 192

        self.bridge = CvBridge()

        rospy.init_node('visual_mpc_demo')
        rospy.loginfo("init node visual mpc demo")
        self.visual_mpc_cmd_publisher = rospy.Publisher('visual_mpc_cmd', numpy_msg(intarray), queue_size=10)
        rospy.Subscriber("main/kinect2/hd/image_color", Image_msg, self.update_image)
        rospy.Subscriber('gen_image', numpy_msg(floatarray), self.update_pred_photos)
        rospy.Subscriber('gen_pix_distrib', numpy_msg(floatarray), self.update_distrib_photos)

        self.assetsdir = '/'.join(str.split(python_visual_mpc.__file__, '/')[:-1]) + '/sawyer/visual_mpc_rospkg/src/assets'

        self.cmap = plt.cm.get_cmap('jet')

        self.root = Tk.Tk()
        self.root.config(bg="white")

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        self.canvasPhoto = ImageTk.PhotoImage(Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white"))
        self.canvas = Canvas(self.root)
        self.canvas.bind("<Button-1>", self.input_pixel)
        self.canvas.grid(row=0, column=0, rowspan=4, columnspan=3, sticky=Tkconstants.NSEW)

        self.canvas.config(bg="white", width=CANVAS_WIDTH, height=CANVAS_HEIGHT, borderwidth=0, highlightthickness=0)
        self.canvasImage = self.canvas.create_image(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, image=self.canvasPhoto)

        labelPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/label.png"))
        self.label = Label(self.root, image=labelPhoto)
        self.label.image = labelPhoto
        self.label.grid(row=0, column=3, pady=(30, 0), padx=(0, 50), sticky=Tkconstants.W)
        self.label.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.num_predictions = 1
        self.predictionPhotos = []
        self.distributionPhotos = []
        self.predictions = []

        self.emptyImage = Image.new("RGB", (PREDICTION_WIDTH, PREDICTION_HEIGHT), "white")
        self.emptyPhoto = ImageTk.PhotoImage(self.emptyImage)

        for i in range(self.num_predictions):
            video = Label(self.root, image=self.emptyPhoto)
            video.image = self.emptyPhoto
            video.grid(row=i + 1, column=3, padx=(40, 0), sticky=Tkconstants.W)
            video.config(bg="white")

            self.predictionPhotos.append([self.emptyImage])
            self.distributionPhotos.append([self.emptyImage])
            self.predictions.append(video)

        self.predictions[-1].grid(pady=(0, 35))

        addPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/add.png"))
        self.addButton = Button(self.root, image=addPhoto, command=self.begin_input)
        self.addButton.image = addPhoto
        self.addButton.grid(column=2, row=4, pady=(0, 50))
        self.addButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        startPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/start.png"))
        self.startButton = Button(self.root, image=startPhoto, command=self.start)
        self.startButton.image = startPhoto
        self.startButton.grid(column=0, row=4, pady=(0, 50))
        self.startButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0,
                                state=Tkconstants.DISABLED)

        resetPhoto = ImageTk.PhotoImage(Image.open(self.assetsdir + "/reset.png"))
        self.resetButton = Button(self.root, image=resetPhoto, command=self.reset_demo)
        self.resetButton.image = resetPhoto
        self.resetButton.grid(column=1, row=4, pady=(0, 50), sticky=Tkconstants.W)
        self.resetButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.iter = 0
        self.video_loop()

    def video_loop(self):
        self.canvas.itemconfig(self.canvasImage, image=self.canvasPhoto)
        self.canvas.copy_image = self.canvasPhoto
        self.iter = (self.iter + 1) % len(self.predictionPhotos[0])
        for i in range(self.num_predictions):
            if len(self.distributionPhotos[i]) < self.prediction_length or len(self.predictionPhotos[i]) < self.prediction_length:
                blendedPhoto = self.emptyPhoto
            else:
                blendedPhoto = ImageTk.PhotoImage(self.distributionPhotos[i][self.iter])
                # blendedPhoto = ImageTk.PhotoImage(Image.blend(self.predictionPhotos[i][self.iter], self.distributionPhotos[i][self.iter], 0.6))
            self.predictions[i].config(image=blendedPhoto)
            self.predictions[i].image = blendedPhoto
        self.root.after(200, self.video_loop)

    def start(self):
        if self.num_pairs == 0:
            print "please select a pair of points"
        elif self.pixel1 and not self.pixel2:
            print "please select second pixel"
        else:
            print "starting"
            self.visual_mpc_cmd_publisher.publish(np.array(self.pairs, dtype=np.uint32))
            # self.clear_inputs()
    #
    # def clear_inputs(self):
    #     # self.num_pairs = 0
    #     # self.pairs = []
    #     # self.pixel1, self.pixel2 = None, None
    #     # self.selPixels = True
    #     self.canvas.delete("points")

    def reset_demo(self):
        self.num_pairs = 0
        self.pairs = []
        self.pixel1, self.pixel2 = None, None
        self.selPixels = True
        self.canvas.delete("points")
        self.receivingPreds = True
        self.receivingDistribs = True
        self.predictionPhotos = []
        self.distributionPhotos = []
        for i in range(self.num_predictions):
            self.predictionPhotos.append([self.emptyImage])
            self.distributionPhotos.append([self.emptyImage])

    def update_image(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv_image = imutils.rotate_bound(cv_image, 180)
        startrow = 48           # 3x16
        startcol = 432          # 27x16
        cv_image = cv_image[startrow:startrow+IMAGE_HEIGHT, startcol:startcol+IMAGE_WIDTH]
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        self.canvasPhoto = ImageTk.PhotoImage(pil_image)

    def update_pred_photos(self, data):
        if self.receivingPreds:
            self.receivingPreds = False
            data = 255 * data.data.reshape((self.num_predictions, self.prediction_length, 56, 64, 3))
            data = data.astype(np.uint8)
            for i in range(self.num_predictions):
                for j in range(self.prediction_length):
                    pil_image = Image.fromarray(data[i, j]).resize([self.prediction_width, self.prediction_height], resample=Image.LANCZOS)
                    self.predictionPhotos[i].append(pil_image)
                self.predictionPhotos[i].pop(0)
            print len(self.predictionPhotos[0])

    def update_distrib_photos(self, data):
        if self.receivingDistribs:
            self.receivingDistribs = False
            data = 255 * data.data.reshape((self.num_pairs, self.num_predictions, self.prediction_length, 56, 64))
            # data = data.astype(np.uint8)
            for i in range(self.num_predictions):
                for j in range(self.prediction_length):
                    colored_distrib = self.cmap(np.squeeze(data[0, i, j]))[:, :, :3]
                    colored_distrib = colored_distrib.astype(np.uint8)
                    pil_image = Image.fromarray(colored_distrib).resize([self.prediction_width, self.prediction_height],
                                                                resample=Image.LANCZOS)
                    self.distributionPhotos[i].append(pil_image)
                self.distributionPhotos[i].pop(0)
            print len(self.distributionPhotos[0])

    def input_pixel(self, event):
        if self.selPixels and event.x >= OFFSET_X and event.y >= OFFSET_Y and event.x <= OFFSET_X + IMAGE_WIDTH and event.y <= OFFSET_Y + IMAGE_HEIGHT:
            self.canvas.create_oval(event.x-6, event.y-6, event.x+6, event.y+6,
                                    outline=COLORS[self.num_pairs % len(COLORS)],
                                    fill=COLORS[self.num_pairs % len(COLORS)],
                                    width=2,
                                    tags="points")

            x = int(round((self.canvas.canvasx(event.x) - OFFSET_X) / RATIO))
            y = int(round((self.canvas.canvasy(event.y) - OFFSET_Y) / RATIO))

            if self.pixel1:
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
                print "pixel 1: ", y, x
                self.pixel1 = [y, x]
                self.startButton.config(state=Tkconstants.DISABLED)

    def begin_input(self):
        print "ready for inputs"
        self.selPixels = True
        if self.pixel1 and not self.pixel2:
            print "please select second pixel"


if __name__ == '__main__':
    v = Visualizer()