import glob
from shutil import copyfile
import os
import cv2


traj_folders = glob.glob('bench_goal_images/*')
out_dir = 'bench_goal_images_reshaped/traj_group0'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i, traj_folder in enumerate(traj_folders):
    print(traj_folder)
    out_pth = out_dir + '/traj{}/images0'.format(i)
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)
    for j in range(18):
        img_pth = glob.glob(traj_folder + '/traj*_{}.jpg'.format(j))[0]
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (64, 48))
        cv2.imwrite(out_pth + '/im_{}.jpg'.format(j), img)
        

