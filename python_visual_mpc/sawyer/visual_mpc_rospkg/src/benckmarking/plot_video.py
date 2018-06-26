import glob
import cPickle as pkl
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.benckmarking.annotate_benchmark import sorted_alphanumeric
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.grasping_robot_env import sudri_crop
from python_visual_mpc.sawyer.visual_mpc_rospkg.src.utils.robot_dualcam_recorder import low2high
import copy
import cv2
import numpy as np
import imageio
def add_crosshairs_single(im, pos, color=None, opacity = None):
    """
    :param im:  shape: r,c,3
    :param pos:
    :param color:
    :return:
    """

    if color is None:
        if im.dtype == np.float32:
            color = np.array([0., 1., 1.])
        else:
            color = np.array([0, 255, 255])
    assert isinstance(color, np.ndarray)
    assert len(im.shape) and im.shape[2] == 3

    pos = np.clip(pos, [0,0], np.array(im.shape[:2]) -1)

    p = pos.astype(np.int)

    if opacity is not None:
        img_float = im.astype(np.float32)
        img_float[p[0] - 20:p[0] - 8, p[1] - 4: p[1] + 4] *= (1 - opacity)
        img_float[p[0] + 8:p[0] + 20, p[1] - 4: p[1] + 4] *= (1 - opacity)

        img_float[p[0] - 4: p[0] + 4, p[1] - 20:p[1] - 8] *= (1 - opacity)

        img_float[p[0] - 4: p[0] + 4, p[1] + 8:p[1] + 20] *= (1 - opacity)
        img_float[p[0] - 2: p[0] + 2, p[1] - 2: p[1] + 2] *= (1 - opacity)

        img_float[p[0] - 20:p[0] - 8, p[1] - 4: p[1] + 4] += opacity * color
        img_float[p[0] + 8:p[0] + 20, p[1] - 4: p[1] + 4] += opacity * color

        img_float[p[0] - 4: p[0] + 4, p[1] - 20:p[1] - 8] += opacity * color

        img_float[p[0] - 4: p[0] + 4, p[1] + 8:p[1] + 20] += opacity * color
        img_float[p[0] - 2: p[0] + 2, p[1] - 2: p[1] + 2] += opacity * color

        im[:] = img_float.astype(np.uint8)

        return im

    im[p[0]-20:p[0]-8,p[1] - 4 : p[1] + 4] = color
    im[p[0]+8:p[0]+20, p[1] - 4 : p[1] + 4] = color

    im[p[0] - 4: p[0] + 4,p[1]-20:p[1]-8] = color

    im[p[0] - 4: p[0] + 4, p[1]+8:p[1]+20] = color
    im[p[0] - 2: p[0] + 2, p[1] - 2 : p[1] + 2] = color



    return im
def main(exp_folder):
    traj_data_folder = exp_folder + '/traj_data'
    verbose_folder = exp_folder + '/verbose/plan'
    print(traj_data_folder)
    red = np.array([0, 0, 255], dtype = np.uint8)
    blue = np.array([255, 0, 0], dtype=np.uint8)
    pix_pos_pkls = sorted_alphanumeric(glob.glob(verbose_folder + '/pix_pos_*.pkl'))
    for p in pix_pos_pkls:
        d = pkl.load(open(p, 'rb'))
        t = int(p.split('pix_pos_dict')[1].split('iter')[0])

        for i, name in enumerate(['front_cam', 'left_cam']):
            img = cv2.imread('{}/images{}/im_med{}.png'.format(traj_data_folder, i, t))

            point_start = d['desig'][i, 0]
            high_point = low2high(point_start, sudri_crop[name], 401, 625, 48, 64)
            add_crosshairs_single(img, high_point.astype(np.int32), red)

            point_end = d['desig'][i, 1]
            high_point = low2high(point_end, sudri_crop[name], 401, 625, 48, 64)
            add_crosshairs_single(img, high_point.astype(np.int32), blue)

            cv2.imwrite('points/points_{}_{}.png'.format(t, i), img)

        print(t, d.keys())
        print(d['desig'])
        print(d['goal_pix'])

def kinect_low_res_to_highres(inp):
    colstart = 180
    rowstart = 0

    shrink_after_crop = .75
    h = {'colstart':colstart, 'rowstart':rowstart,'shrink_after_crop':shrink_after_crop}

    l_rowstart = 10  # 10
    l_colstart = 32  # 28
    l_shrink_before_crop = 1. / 9
    l = {'colstart': l_colstart, 'rowstart': l_rowstart, 'shrink_before_crop': l_shrink_before_crop}

    highres = (inp + np.array([l['rowstart'], l['colstart']])).astype(np.float) / l['shrink_before_crop']

    highres = highres.astype(np.int64)
    return highres
def main_singlecam(exp_folder):
    video_folder = exp_folder + '/videos'
    verbose_folder = exp_folder + '/verbose/plan'

    start_color = np.array([0, 0, 255], dtype=np.uint8)
    end_color = np.array([255, 0 , 0], dtype = np.uint8)
    pix_pos_pkls = sorted_alphanumeric(glob.glob(verbose_folder + '/pix_pos_*.pkl'))

    # tradeoff_pkl = pkl.load(open(verbose_folder + '/warperrs_tradeoff.pkl'))
    # tradeoffs = tradeoff_pkl['tradeoff']
    min_t = 19000
    rendered = {}
    for p in pix_pos_pkls:
        d = pkl.load(open(p, 'rb'))
        raw_t = int(p.split('pix_pos_dict')[1].split('iter')[0]) - 1

        t = int(np.round(raw_t * 120./30))
        min_t = min(t, min_t)
        print(raw_t, t, d.keys())
        print(d['desig'].shape)
        print(d['goal_pix'].shape)
        # print('tradeoffs', tradeoffs[raw_t])

        img = cv2.imread('{}/main_full_cropped_im{}.jpg'.format(video_folder, t))

        point = d['desig'][0, 0]
        high_point = kinect_low_res_to_highres(point)
        add_crosshairs_single(img, high_point.astype(np.int32), start_color, 0.74)

        point = d['desig'][0, 1]
        high_point = kinect_low_res_to_highres(point)
        add_crosshairs_single(img, high_point.astype(np.int32), end_color, 0.26)

        rendered[t] = img
        cv2.imwrite('points_single/points_{}.png'.format(t), img)
    vid = []
    for i in range(min_t, 120):
        if i in rendered.keys():
            for _ in range(5):
                vid.append(copy.deepcopy(rendered[i][:, :, ::-1]))
        else:
            img = cv2.imread('{}/main_full_cropped_im{}.jpg'.format(video_folder, i))[:, :, ::-1]
            vid.append((img * (8./ 8)).astype(np.uint8))

    writer = imageio.get_writer('points_single/vid_fancy2.mp4')
    for img in vid:
        writer.append_data(img)
    writer.close()



if __name__ == '__main__':
    main_singlecam('/home/sudeep/Desktop/try_one')
