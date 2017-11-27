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
# from cv_bridge import CvBridge
#
# from rospy.numpy_msg import numpy_msg
# from visual_mpc_rospkg.msg import intarray, floatarray
# from sensor_msgs.msg import Image as Image_msg
# import rospy

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


class Visualizer(object):
    def __init__(self):
        self.num_pairs = 0
        self.pairs = []
        self.pixel1, self.pixel2 = None, None
        self.selPixels = False

        # self.bridge = CvBridge()
        # self.visual_mpc_cmd_publisher = rospy.Publisher('visual_mpc_cmd', numpy_msg(intarray), queue_size=10)
        # rospy.Subscriber("main/kinect2/hd/image_color", Image_msg, self.update_image)
        # rospy.Subscriber('gen_image', numpy_msg(floatarray), self.update_preds)
        # rospy.Subscriber('gen_pix_distrib', numpy_msg(floatarray), self.update_distribs)

        self.root = Tk.Tk()
        self.root.config(bg="white")

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        self.canvasPhoto = ImageTk.PhotoImage(Image.open("assets/frames0/frame0.png").resize([IMAGE_WIDTH, IMAGE_HEIGHT], Image.ANTIALIAS))
        self.canvas = Canvas(self.root)
        self.canvas.bind("<Button-1>", self.input_pixel)
        self.canvas.grid(row=0, column=0, rowspan=4, columnspan=3, sticky=Tkconstants.NSEW)

        self.canvas.config(bg="white", width=CANVAS_WIDTH, height=CANVAS_HEIGHT, borderwidth=0, highlightthickness=0)
        self.canvasImage = self.canvas.create_image(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, image=self.canvasPhoto)

        labelPhoto = ImageTk.PhotoImage(Image.open("assets/label.png"))
        self.label = Label(self.root, image=labelPhoto)
        self.label.image = labelPhoto
        self.label.grid(row=0, column=3, pady=(30, 0), padx=(0, 50), sticky=Tkconstants.W)
        self.label.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.num_predictions = 3
        self.predictionPhotos = []
        self.predictions = []
        for i in range(self.num_predictions):
            photo = ImageTk.PhotoImage(Image.open("assets/frames%d/frame0.png" % i).resize([PREDICTION_WIDTH, PREDICTION_HEIGHT], Image.ANTIALIAS))
            video = Label(self.root, image=photo)
            video.image = photo
            video.grid(row=i+1, column=3, padx=(40, 0), sticky=Tkconstants.W)
            video.config(bg="white")

            self.predictionPhotos.append(photo)
            self.predictions.append(video)

        self.predictions[-1].grid(pady=(0, 35))

        # self.distribPhotos = []
        # self.distribs = []
        # for i in range(self.num_predictions):
        #     photo = ImageTk.PhotoImage(Image.open("assets/frames%d/frame0.png" % i).resize([128, 96]))
        #     video = Label(self.root, image=photo)
        #     video.image = photo
        #     video.grid(row=3, column=i, pady=(0, 20), sticky=Tkconstants.NSEW)
        #     video.config(bg="white")
        #
        #     self.distribPhotos.append(photo)
        #     self.distribs.append(video)
        #
        # self.distribs[0].grid(padx=(37, 0))
        # self.distribs[-1].grid(padx=(0, 37))

        addPhoto = ImageTk.PhotoImage(Image.open("assets/add.png"))
        self.addButton = Button(self.root, image=addPhoto, command=self.begin_input)
        self.addButton.image = addPhoto
        self.addButton.grid(column=2, row=4, pady=(0, 50))
        self.addButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        startPhoto = ImageTk.PhotoImage(Image.open("assets/start.png"))
        self.startButton = Button(self.root, image=startPhoto, command=self.start)
        self.startButton.image = startPhoto
        self.startButton.grid(column=0, row=4, pady=(0, 50))
        self.startButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        resetPhoto = ImageTk.PhotoImage(Image.open("assets/reset.png"))
        self.resetButton = Button(self.root, image=resetPhoto, command=self.reset)
        self.resetButton.image = resetPhoto
        self.resetButton.grid(column=1, row=4, pady=(0, 50), sticky=Tkconstants.W)
        self.resetButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.iter = 0
        self.video_loop()

    def video_loop(self):
        self.canvas.itemconfig(self.canvasImage, image=self.canvasPhoto)

        self.update_preds([Image.open("assets/frames0/frame%d.png" % self.iter).resize([PREDICTION_WIDTH, PREDICTION_HEIGHT], Image.ANTIALIAS),
                           Image.open("assets/frames1/frame%d.png" % self.iter).resize([PREDICTION_WIDTH, PREDICTION_HEIGHT], Image.ANTIALIAS),
                           Image.open("assets/frames2/frame%d.png" % self.iter).resize([PREDICTION_WIDTH, PREDICTION_HEIGHT], Image.ANTIALIAS)])

        for i in range(self.num_predictions):
            self.predictions[i].config(image=self.predictionPhotos[i])
        self.iter = (self.iter + 1) % 14
        self.root.after(500, self.video_loop)

    def start(self):
        if self.num_pairs == 0:
            print "please select a pair of points"
        elif self.pixel1 and not self.pixel2:
            print "please select second pixel"
        else:
            print "starting"
            # self.visual_mpc_cmd_publisher.publish(self.pairs)
            self.reset()

    def reset(self):
        self.num_pairs = 0
        self.pairs = []
        self.canvas.delete("points")
        self.selPixels = False

    def update_image(self, image):
        self.canvas.image = ImageTk.PhotoImage(image)

    def update_preds(self, pred_photos):
        for i in range(self.num_predictions):
            self.predictionPhotos[i] = ImageTk.PhotoImage(pred_photos[i])

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
            else:
                print "pixel 1: ", y, x
                self.pixel1 = [y, x]

    def begin_input(self):
        print "ready for inputs"
        self.selPixels = True
        if self.pixel1 and not self.pixel2:
            print "please select second pixel"

    # def update_image(self, data):
    #     cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #     cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #     pil_im = Image.fromarray(cv_image)
    #     self.panel.image = ImageTk.PhotoImage(pil_im)

    # def update_preds(self, data):
    #     for i in range(self.num_predictions):
    #         cv_image = self.bridge.imgmsg_to_cv2(data[i], "bgr8")
    #         cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #         pil_im = Image.fromarray(cv_image)
    #         self.predictionPhotos[i] = ImageTk.PhotoImage(pil_im)

    # def update_distribs(self, data):
    #     for i in range(self.num_predictions):
    #         cv_image = self.bridge.imgmsg_to_cv2(data[i], "bgr8")
    #         cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #         pil_im = Image.fromarray(cv_image)
    #         self.distribPhotos[i] = ImageTk.PhotoImage(pil_im)

    # def crop_lowres(self, cv_image):
    #     self.ltob.d_img_raw_npy = np.asarray(cv_image)
    #
    #
    #     shrink_before_crop = 1 / 16.
    #     img = cv2.resize(cv_image, (0, 0), fx=shrink_before_crop, fy=shrink_before_crop, interpolation=cv2.INTER_AREA)
    #     startrow = 3
    #     startcol = 27
    #
    #     img = imutils.rotate_bound(img, 180)
    #     endcol = startcol + self.img_width
    #     endrow = startrow + self.img_height
    #
    #     # crop image:
    #     img = img[startrow:endrow, startcol:endcol]
    #     assert img.shape == (self.img_height, self.img_width, 3)
    #
    #     self.crop_lowres_params = {'startcol':startcol,'startrow':startrow,'shrink_before_crop':shrink_before_crop}
    #     return img


if __name__ == '__main__':
    v = Visualizer()