import cv2
import os


img_dir = '/home/guser/catkin_ws/src/lsdc/experiments/cem_exp/benchmarks_sawyer/predprop_1stimg_bckgd/recvideo/videos'

import glob


images = []

for i in range(0,40):
    imgname = glob.glob(img_dir + '/main_full_cropped_im{}_*'.format(i))
    if len(imgname) > 1:
        print 'num img for step {} greater than 1'.format(i)
    images.append(imgname[0])


output = '/'.join(str.split(img_dir, '/')[:-1]) + '/output.mp4'

# Determine the width and height from the first image
frame = cv2.imread(images[0])
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

for image in images:
    frame = cv2.imread(image)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))