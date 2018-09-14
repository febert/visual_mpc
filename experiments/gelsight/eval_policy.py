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
import pickle
import datetime

parser = argparse.ArgumentParser(description='Convert full trajectory folders to .tfrecord')
parser.add_argument('goal_image_dir', metavar='goal_image_dir', type=str, help='directory containing goal images')
parser.add_argument('traj_images_dir', metavar='traj_image_dir', type=str, help='directory containing trajectories')
parser.add_argument('baseline_traj_dir', metavar='baseline_traj_dir', type=str, help='directory containing baseline trajectories')
parser.add_argument('zero_image', metavar='zero_image', type=str, help='zero image to reference')
parser.add_argument('output_dir', metavar='output_dir', type=str, help='output location')
parser.add_argument('--full_gif', metavar='full_gif', type=bool, default=False, help='generate gifs?')
parser.add_argument('--grayscale', metavar='grayscale', type=bool, default=False, help='generate gifs?')
parser.add_argument('--normalize', metavar='normalize', type=bool, default=False, help='generate gifs?')

args = parser.parse_args()

def normalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    img = img / np.sum(img)
    return img

zero_image = normalize(cv2.imread(args.zero_image)[:, :, ::-1])

def make_gif(output_dir, images):
    # Convert list of np arrays into a gif
    io.mimsave(output_dir, images, duration=0.2)

def auto_centroid(img):
    # Given a full color image, return the centroid of the ball location using
    # blob finding.
    # Return: tuple of x, y

    #difference_image = np.abs(img.astype(float) - zero_image.astype(float))
    diff_image = np.abs(img - zero_image)
    diff_image = np.clip(diff_image, 1e-5, 1)
    print('diff img')
    # plt.imshow(diff_image, cmap='gray')
    # plt.colorbar()
    # plt.show()
    # ret, diff_img = cv2.threshold(diff_img, 70, 255, cv2.THRESH_TOZERO)
    # plt.imshow(diff_img, cmap='gray')
    # plt.show()
    # print(diff_img)
    cent = list(ndimage.measurements.center_of_mass(diff_image))
    print(cent)
    if np.isnan(cent[0]):
        cent[0] = 0
    if np.isnan(cent[1]):
        cent[1] = 0
    return int(cent[1]), int(cent[0])

def hand_centroid(img):
    # Given a full color image, have the user click the location of the center of the
    # blob.
    # Return: tuple of x, y


    plt.figure(figsize=(6,4))
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


def centroid_metric(images, goal, color, metric, out_dir):

    # Give back euclidean norm of vector b/w final trajectory image and goal, using automatic centroid finding


    goal_cent = np.array(metric(goal))
    annotated = []
    distances = []
    for img in images:
        cent = np.array(metric(img))
        distances.append(np.linalg.norm(goal_cent - cent))
        # Make a copy and annotate it with the positions
        cpy = np.copy(img)
        cv2.circle(color, (goal_cent[0], goal_cent[1]), 3, (0, 0, 0))
        cv2.circle(color, (cent[0], cent[1]), 3, (255, 255, 255))
        annotated.append(color)

    if args.full_gif:
        make_gif(out_dir, annotated)
    else:
        cv2.imwrite(out_dir, annotated[-1])

    return distances[-1]


def mse(images, goal):
    # Mean squared error of two color images
    # fig = plt.figure(figsize=(1, 2))
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(images, cmap='gray')
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(goal, cmap='gray')
    # plt.show()
    mse = np.mean(np.square(images - goal))
    print('mse: {}'.format(mse))
    return mse


def run_metrics(trajectory_dir, out_dir, invert_goal=False):
    # Run metrics on the trajectories in trajectory_dir compared to goals
    trajs = sorted(glob.glob(trajectory_dir + '/traj*'))
    print(trajs)

    auto_err = []
    hand_err = []
    mse_err = []

    for i in range(len(trajs)):

        img_paths = glob.glob(trajs[i] + '/images0/im_17.jpg')
        images = [cv2.imread(path) for path in img_paths]
        goal_traj_paths = glob.glob(args.goal_image_dir + '/traj{}/images0/im_17.jpg'.format(i))
        goal_image = cv2.imread(goal_traj_paths[0])

        if not args.full_gif:
            images = [images[-1]]

        if invert_goal:
            # BGR to RGB conversion
            images[-1] = images[-1][:, :, ::-1]

        full_img_cpy = np.copy(images[-1])
        images[-1] = normalize(images[-1])

        # plt.imshow(images[-1], cmap='gray')
        # plt.colorbar()
        # plt.show()
        goal_image = normalize(goal_image)

        # plt.imshow(goal_image, cmap='gray')
        # plt.colorbar()
        # plt.show()

        if not args.full_gif:
            auto_met = centroid_metric(images, goal_image, full_img_cpy, auto_centroid, out_dir + '/auto{}.jpg'.format(i))
            #hand_met = centroid_metric(images, goal_image, hand_centroid, out_dir + '/hand{}.jpg'.format(i))
        else:
            auto_met = centroid_metric(images, goal_image, auto_centroid, out_dir + '/auto{}.gif'.format(i))
            #hand_met = centroid_metric(images, goal_image, hand_centroid, out_dir + '/hand{}.gif'.format(i))

        auto_err.append(auto_met)
        #hand_err.append(hand_met)
        #
        # plt.imshow(images[-1])
        # plt.title('image')
        # plt.show()
        #
        # plt.imshow(goal_image)
        # plt.title('goal')
        # plt.show()




        mse_err.append(mse(images[-1], goal_image))

    return auto_err, hand_err, mse_err


if __name__ == '__main__':

    dirs = [args.output_dir + '/baseline', args.output_dir + '/mpc']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    print(args.full_gif)
    bl_auto_err, bl_hand_err, bl_mse_err = run_metrics(args.baseline_traj_dir, args.output_dir + '/baseline') # TODO: Don't invert goal
    mpc_auto_err, mpc_hand_err, mpc_mse_err = run_metrics(args.traj_images_dir, args.output_dir + '/mpc', invert_goal=True)
    print(bl_auto_err)
    print(bl_hand_err)
    print(bl_mse_err)
    print(mpc_auto_err)
    print(mpc_hand_err)
    print(mpc_mse_err)

    ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

    dump = {
        'bl_auto_err': bl_auto_err,
        'bl_hand_err': bl_hand_err,
        'bl_mse_err': bl_mse_err,
        'mpc_auto_err': mpc_auto_err,
        'mpc_hand_err': mpc_hand_err,
        'mpc_mse_err': mpc_mse_err
    }
    with open(args.output_dir + '/pickle-stats' + ctimestr + '.pkl', 'wb') as f:
        pickle.dump(dump, f)

    plt.figure()
    bl_auto_err = sorted(bl_auto_err)
    bl_hand_err = sorted(bl_hand_err)
    bl_mse_err = sorted(bl_mse_err)
    mpc_auto_err = sorted(mpc_auto_err)
    mpc_hand_err = sorted(mpc_hand_err)
    mpc_mse_err = sorted(mpc_mse_err)

    plt.figure()
    plt.step(bl_auto_err, np.arange(len(bl_auto_err)))
    plt.step(mpc_auto_err, np.arange(len(mpc_auto_err)))
    plt.legend(['bl','mpc'])
    #plt.hist([bl_auto_err, mpc_auto_err], histtype='bar', color=['red', 'blue'], label=['baseline', 'mpc'])
    # plt.legend(loc='best')
    plt.show()
    plt.figure()
    plt.step(bl_hand_err, np.arange(len(bl_hand_err)))
    plt.step(mpc_hand_err, np.arange(len(mpc_hand_err)))
    plt.legend(['bl','mpc'])
    # plt.hist([bl_hand_err, mpc_hand_err], histtype='bar', color=['red', 'blue'], label=['baseline', 'mpc'])
    # plt.legend(loc='best')
    plt.show()
    plt.figure()
    plt.step(bl_mse_err, np.arange(len(bl_mse_err)))
    plt.step(mpc_mse_err, np.arange(len(mpc_mse_err)))
    plt.legend(['bl','mpc'])
    #plt.hist([bl_mse_err, mpc_mse_err], histtype='bar', color=['red', 'blue'], label=['baseline', 'mpc'])

    #plt.legend(loc='best')
    plt.show()


