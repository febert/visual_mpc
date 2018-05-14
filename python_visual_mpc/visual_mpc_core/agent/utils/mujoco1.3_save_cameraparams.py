from copy import deepcopy

import numpy as np

import mujoco_py
from mujoco_py.mjtypes import *

from PIL import Image
import pickle

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os

from python_visual_mpc import __file__ as python_vmpc_path

cwd = os.getcwd()
BASE_DIR = '/'.join(str.split(cwd, '/')[:-4])

def save_params(xmlname, cam_ids=[0], savename='params'):
    model= mujoco_py.MjModel(xmlname)

    height, width = 480, 640
    viewports, model_views, projections = [], [], []

    root_dir = '/'.join(str.split(python_vmpc_path, '/')[:-1])
    save_dir = root_dir + '/visual_mpc_core/agent/utils/{}.pkl'.format(savename)

    for icam in cam_ids:

        print('cam ', icam)
        viewer = mujoco_py.MjViewer(visible=True, init_width=width, init_height=height)
        viewer.start()
        viewer.set_model(model)
        viewer.cam.camid = icam
        print(viewer.cam.camid)

        model.data.qpos = np.array([0.1, 0.1, 0.])

        # T = 1
        # for t in range(T):
        # viewer.loop_once()

        mj_U = np.array([0., 0.])
        model.data.ctrl = mj_U
        model.step()

        viewer.loop_once()

        img_string, width, height= viewer.get_image()
        model_view, proj, viewport = viewer.get_mats()

        model_views.append(model_view)
        projections.append(proj)
        viewports.append(viewport)

        save_dir= root_dir + '/visual_mpc_core/agent/utils'
        with open(save_dir + '/{}_cam{}.txt'.format(savename, icam),'w') as f:
            f.write("proj \n")
            for i in range(proj.shape[0]):
                f.write("{} \n".format(proj[i]))

            f.write("modelview\n")
            for i in range(model_view.shape[0]):
                f.write("{} \n".format(model_view[i]))

            f.write("{}".format(viewport))

        largeimage = np.fromstring(img_string, dtype='uint8').reshape(
            (height, width, 3))[::-1, :, :]

        Image.fromarray(largeimage).save('testimg.png')

        # img_string, width, height = viewer.get_depth()
        # largedimage = np.fromstring(img_string, dtype=np.float32).reshape(
        #     (height, width, 1))[::-1, :, :]
        # plt.imshow(np.squeeze(largedimage))
        # plt.show()

        # if t % 10 ==0:
        r, c = viewer.project_point(model.data.qpos)
        print('model.data.qpos', model.data.qpos)
        print("row, col", r, c)
        r = int(r)
        c = int(c)

        largeimage[r-1:r+1,:] = [255, 255, 255]
        largeimage[:, c-1:c+1] = [255, 255, 255]
        plt.imshow(largeimage)
        plt.show()

        model.data.qvel.setflags(write=True)
        # import pdb; pdb.set_trace()
        viewer.finish()
        # j = Image.fromarray(largeimage)
        # j.save("img", "BMP")

    mats = {}
    mats['viewport'] = viewports
    mats['modelview'] = model_views
    mats['projection'] = projections
    pickle.dump(mats, open(save_dir + '/proj_mats_dual.pkl', 'wb'))

if __name__ == '__main__':
    filename = BASE_DIR + '/mjc_models/cartgripper_updown_2cam_noautogen.xml'
    save_params(filename, [0,1], 'proj_mats')
