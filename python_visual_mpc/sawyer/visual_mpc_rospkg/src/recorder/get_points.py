#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import cv2
import robot_recorder

class Getdesig(object):
    def __init__(self, img):
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)
        plt.imshow(img)

        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.top_left = np.zeros((2, 2))  #idesig, (r,c)
        self.top_right = np.zeros((2, 2))
        self.bottom_left = np.zeros((2, 2))
        self.bottom_right = np.zeros((2, 2))

        self.i_desig = 0

        plt.show()

    def onclick(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        self.ax.set_xlim(0, 63)
        self.ax.set_ylim(63, 0)

        print 'click', self.i_desig

        if self.i_desig == 0:
            self.top_left = np.array([event.ydata, event.xdata])
            print self.top_left
        if self.i_desig == 1:
            self.top_right = np.array([event.ydata, event.xdata])
            print self.top_right
        if self.i_desig == 2:
            self.bottom_left = np.array([event.ydata, event.xdata])
            print self.bottom_left
        if self.i_desig == 3:
            self.bottom_right = np.array([event.ydata, event.xdata])
            print self.bottom_right
            plt.savefig('corners.png')

            plt.close()
            # with open(self.basedir +'/desig_goal_pix{}.pkl'.format(self.suf), 'wb') as f:
            with open('desig_corners.pkl', 'wb') as f:
                dict = {'top_left': self.top_left,
                        'top_right': self.top_right,
                        'bottom_left': self.bottom_left,
                        'bottom_right': self.bottom_right}
                cPickle.dump(dict, f)

        self.ax.scatter(event.ydata, event.xdata, s=100, marker="D", facecolors='r', edgecolors='r')
        plt.draw()

        self.i_desig += 1

if __name__ == '__main__':
    import rospy
    rospy.init_node('get_points')
    rospy.loginfo("init get points node")
    recorder = robot_recorder.RobotRecorder(save_dir='',
                                            seq_len=60,
                                            use_aux=False,
                                            save_video=True,
                                            save_actions=False,
                                            save_images=False,
                                            image_shape=(60,60))
    imagemain = recorder.ltob.img_cv2
    print 'imagemain:', imagemain
    imagemain = cv2.cvtColor(imagemain, cv2.COLOR_BGR2RGB)
    c_main = Getdesig(imagemain)