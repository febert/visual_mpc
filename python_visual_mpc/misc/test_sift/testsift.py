import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob as glob


def test_sift():
    path = '/mnt/sda1/sawyerdata/weiss_gripper/vestri/main/traj_group0/traj0/images'
    im_file1 = glob(path + '/main_full_cropped_im00_time*')[0]
    im_file2 = glob(path + '/main_full_cropped_im10_time*')[0]

    img1 = cv2.imread(im_file1,0)          # queryImage
    img2 = cv2.imread(im_file2,0) # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
    plt.imshow(img3),plt.show()


if __name__ == '__main__':
    test_sift()