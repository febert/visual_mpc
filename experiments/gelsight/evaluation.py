import cv2
import numpy as np
import os
import glob
import imageio as io
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser(description='Convert full trajectory folders to .tfrecord')
parser.add_argument('goal_image_dir', metavar='goal_image_dir', type=str, help='directory containing goal images')
parser.add_argument('traj_images_dir', metavar='traj_image_dir', type=str, help='directory containing trajectories')
parser.add_argument('baseline_traj_dir', metavar='baseline_traj_dir', type=str, help='directory containing baseline trajectories')
parser.add_argument('output_dir', metavar='output_dir', type=str, help='output location')

args = parser.parse_args()

GOAL_IMAGE_DIR = 'goal_images/traj_group0/'
TRAJ_IMAGES_DIR = 'exp_2018-09-12:15:46:52/trajectories/gelsight/train/traj_group0/'
OUTPUT_DIR = 'postprocess-baseline/'


def get_goal_image(traj_idx, img_idx):
    image_file = args.goal_image_dir + 'traj{}/images0/im_{}.jpg'.format(traj_idx, img_idx)
    if not os.path.isfile(image_file):
        raise ValueError("Image not found!")
    else:
        return cv2.imread(image_file)


def get_empty_image(idx):
    return get_goal_image(idx, 1) # Discard the 0th image for adjustment


def get_final_goal_image(idx):
    return get_goal_image(idx, 35)


def get_trajectory_images(dir, traj_idx):
    image_file_search = dir + 'traj{}/images0/*'.format(traj_idx)
    files = glob.glob(image_file_search)
    for file in files:
        yield cv2.imread(file)


def hand_annotate(img):
    plt.figure()
    implot = plt.imshow(img)
    cent = None

    def onclick(event):
        nonlocal cent
        cent = (int(event.xdata), int(event.ydata))
    cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()
    implot.figure.canvas.mpl_disconnect(cid)
    return cent


def auto_centroid(img, empty):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray, cmap='gray')
    # plt.show()
    # params = cv2.SimpleBlobDetector_Params()
    # params.minThreshold = 10
    # params.maxThreshold = 200
    #
    # params.minCircularity = 0.05
    # params.blobColor = 255
    # detector = cv2.SimpleBlobDetector_create(params)
    # kp = detector.detect(gray)
    # if not kp:
    #     return 0, 0
    # return int(kp[0].pt[0]), int(kp[0].pt[1])

    # ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    # im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    # cnt = contours[0]
    # M = cv2.moments(cnt)
    # if M['m00'] == 0:
    #     return 0, 0
    # return int(M['m10']/M['m00']), int(M['m01']/M['m00'])


    difference_image = np.abs(img.astype(float) - empty.astype(float))
    difference_image = difference_image.astype('uint8')
    diff_img = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(diff_img, cmap='gray')
    # plt.colorbar()
    # plt.show()
    ret, diff_img = cv2.threshold(diff_img, 70, 255, cv2.THRESH_TOZERO)
    # plt.imshow(diff_img, cmap='gray')
    # plt.show()
    cent = ndimage.measurements.center_of_mass(diff_img)
    return int(cent[1]), int(cent[0])
    # im2, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # pdb.set_trace()
    # cv2.drawContours(diff_img, contours, -1, (0, 255, 0), 3)
    # show(diff_img)
    # cnt = contours[0]
    # M = cv2.moments(cnt)
    if M['m00'] == 0:
        print("ERROR CALCULATING MOMENT")
        return 0, 0
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])


def show(img):
    # plt.imshow(img)
    # plt.show()
    pass


def centroids_traj(traj_idx):


    print('Goal cent: {}'.format(goal_cent))
    gif_modified_imgs = []
    cents = []
    for image in tqdm(get_trajectory_images(traj_idx)):
        im_copy = np.copy(image)
        show(im_copy)
        cent = centroid(im_copy, empty_img)
        print('Image cent: {}'.format(cent))
        cents.append(cent)
        cv2.circle(im_copy, cent, 3, (0, 0, 0))
        cv2.circle(im_copy, goal_cent, 3, (255, 255, 255))
        show(im_copy)
        gif_modified_imgs.append(im_copy)
    return gif_modified_imgs, cents, goal_cent



def run_stats(traj_dir):
    eval_methods =
    traj_folders = glob.glob(traj_dir + 'traj*')

    for trajectory in traj_folders:
        goal_img = get_final_goal_image(traj_idx)
        show(goal_img)
        empty_img = get_empty_image(traj_idx)
        goal_cent = centroid(goal_img, empty_img)






def make_gif(output_dir, images):
    io.mimsave(output_dir, images, duration=0.2)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        traj_indices = [sys.argv[1]]
    else:
        traj_indices = range(3)
    for traj_index in traj_indices:
        gif_images, cents, goal_cent = centroids_traj(traj_index)
        gif_dir = OUTPUT_DIR
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
        make_gif(gif_dir + '/traj_{}.gif'.format(traj_index), gif_images)
