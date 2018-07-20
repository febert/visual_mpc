import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl

""" CALIBRATES VIEWER. USER CLICKS ON SQUARE BOARD CORNERS IN CLOCKWISE FASHION. START AT TOP LEFT"""

bottom = cv2.imread('bottom_calib.jpg')
kinect = cv2.imread('kinect_calib.jpg')
birds_eye = cv2.imread('birds_eye_calib.jpg')

pnts_bottom = []
pnts_kinect = []
pnts_birds = []

def click_bottom(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		pnts_bottom.append([y, x])
def click_birds(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		pnts_birds.append([y, x])
def click_kinect(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		pnts_kinect.append([y, x])

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_bottom)
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", bottom.copy())
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_birds)
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", birds_eye.copy())
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break


cv2.namedWindow("image")
cv2.setMouseCallback("image", click_kinect)
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", kinect.copy())
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break


pnts_bottom = np.array(pnts_bottom)
pnts_kinect = np.array(pnts_kinect)
pnts_birds = np.array(pnts_birds)

pnts_dict = {'pnts_bottom':pnts_bottom, 'pnts_kinect':pnts_kinect, 'pnts_birds':pnts_birds}
pkl.dump(pnts_dict, open('points.pkl', 'wb'))
