#!/usr/bin/env python
import numpy as np
from matplotlib import animation
import Tkinter as Tk
from Tkinter import Button, Frame, Canvas, Scrollbar, Label
import Tkconstants
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import cv2
# from cv_bridge import CvBridge
#
# from rospy.numpy_msg import numpy_msg
# from visual_mpc_rospkg.msg import intarray, floatarray
# from sensor_msgs.msg import Image as Image_msg
# import rospy


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
        self.canvasPhoto = ImageTk.PhotoImage(Image.open("assets/frames0/frame0.png").resize([480, 480]))
        self.canvas = Canvas(self.root)
        self.canvas.bind("<Button-1>", self.input_pixel)
        self.canvas.grid(row=0, column=0, rowspan=2, columnspan=3, sticky=Tkconstants.NSEW)
        self.canvas.config(bg="white", width=600, height=525, borderwidth=0, highlightthickness=0)
        self.canvasImage = self.canvas.create_image(300, 262, image=self.canvasPhoto)

        self.num_predictions = 3
        self.predictionPhotos = []
        self.predictions = []
        for i in range(self.num_predictions):
            photo = ImageTk.PhotoImage(Image.open("assets/frames%d/frame0.png" % i).resize([128, 96]))
            video = Label(self.root, image=photo)
            video.image = photo
            video.grid(row=2, column=i, pady=(0, 20), sticky=Tkconstants.NSEW)
            video.config(bg="white")

            self.predictionPhotos.append(photo)
            self.predictions.append(video)

        self.predictions[0].grid(padx=(37, 0))
        self.predictions[-1].grid(padx=(0, 37))

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
        self.addButton.grid(column=3, row=0, padx=(0, 50), pady=(80, 0), columnspan=2)
        self.addButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        startPhoto = ImageTk.PhotoImage(Image.open("assets/start.png"))
        self.startButton = Button(self.root, image=startPhoto, command=self.start)
        self.startButton.image = startPhoto
        self.startButton.grid(column=3, row=1, padx=(0, 10), pady=80, sticky=Tkconstants.S)
        self.startButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        resetPhoto = ImageTk.PhotoImage(Image.open("assets/reset.png"))
        self.resetButton = Button(self.root, image=resetPhoto, command=self.reset)
        self.resetButton.image = resetPhoto
        self.resetButton.grid(column=4, row=1, padx=(10, 50), pady=80, sticky=Tkconstants.S)
        self.resetButton.config(bg="white", activebackground="white", borderwidth=0, highlightthickness=0)

        self.iter = 0
        self.video_loop()

    def video_loop(self):
        self.canvas.itemconfig(self.canvasImage, image=self.canvasPhoto)

        self.update_preds([Image.open("assets/frames0/frame%d.png" % self.iter).resize([128, 96]),
                           Image.open("assets/frames1/frame%d.png" % self.iter).resize([128, 96]),
                           Image.open("assets/frames2/frame%d.png" % self.iter).resize([128, 96])])

        for i in range(self.num_predictions):
            self.predictions[i].config(image=self.predictionPhotos[i])
        self.iter = (self.iter + 1) % 14
        self.root.after(500, self.video_loop)

    def start(self):
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
        if self.selPixels:
            x = event.x
            y = event.y

            self.canvas.create_oval(x-4, y-4, x+4, y+4, outline="#f11",
                               fill="#f11", width=2, tags="points")

            if self.pixel1:
                self.pixel2 = (y, x)
                self.selPixels = False
            else:
                self.pixel1 = (y, x)

    def begin_input(self):
        self.selPixels = True
        if self.pixel1 and self.pixel2:
            self.pairs.append([self.pixel1, self.pixel2])
            self.num_pairs += 1
            self.pixel1 = None
            self.pixel2 = None
        elif self.pixel1 and not self.pixel2:
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


if __name__ == '__main__':
    v = Visualizer()