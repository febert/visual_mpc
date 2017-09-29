import cv2
import sys
import glob
import pdb


import moviepy.editor as mpy

def make_gif(im_list):
    clip = mpy.ImageSequenceClip(im_list, fps=4)
    clip.write_gif('tracking.gif')

if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN

    # tracker = cv2.Tracker_create("MIL")
    # tracker = cv2.TrackerKCF_create()
    tracker = cv2.TrackerMIL_create()

    # get the frist image of the sequence
    image_folder = 'testdata/img'
    [imfile] = glob.glob(image_folder + "/main_full_cropped_im{}_*.jpg".format(str(0).zfill(2)))

    imread_flag = None
    frame = cv2.imread(imfile)
    # cv2.imshow("Tracking", frame)
    # cv2.waitKey()

    # Define an initial bounding box
    # bbox = (450, 100, 50, 50)  # for the arm
    bbox = (550, 240, 50, 50)  # for the small snow-man

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    video_len = 96

    im_with_bbox = []

    for i_save in range(1, video_len):

        [imfile] = glob.glob(image_folder + "/main_full_cropped_im{}_*.jpg".format(str(i_save).zfill(2)))

        interval = 1
        if i_save % interval == 0:
            frame = cv2.imread(imfile)

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        # if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))
        print 'tracking ok:', ok
        # Display result
        cv2.imshow("Tracking", frame)

        small_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_rgb = cv2.resize(small_rgb, (0, 0), fx=0.6, fy=0.6)
        if i_save % 5 == 0:
            im_with_bbox.append(small_rgb)

        # cv2.waitKey()

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        # pdb.set_trace()


    make_gif(im_with_bbox)
