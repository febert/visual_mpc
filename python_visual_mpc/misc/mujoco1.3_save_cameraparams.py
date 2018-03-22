from copy import deepcopy

import numpy as np

import mujoco_py
from mujoco_py.mjtypes import *

from PIL import Image
import cPickle

import os
cwd = os.getcwd()

BASE_DIR = '/'.join(str.split(cwd, '/')[:-2])
filename = BASE_DIR + '/mjc_models/cartgripper_noautogen.xml'

model= mujoco_py.MjModel(filename)

height, width = 480, 640
viewer = mujoco_py.MjViewer(visible=True, init_width=width, init_height=height)
viewer.start()
viewer.set_model(model)
viewer.cam.camid = 0
print viewer.cam.camid

import matplotlib.pyplot as plt

from python_visual_mpc import __file__ as python_vmpc_path
root_dir = '/'.join(str.split(python_vmpc_path, '/')[:-1])

model.data.qpos = q = np.array([0., 0., 0.])

T = 1000
for t in range(T):
    viewer.loop_once()

    mj_U = np.array([-10. , -10.])
    model.data.ctrl = mj_U

    model.step()

    img_string, width, height= viewer.get_image()
    model_view, proj, viewport = viewer.get_mats()

    with open('cam_params_1.3.txt','w') as f:
        f.write("proj \n")
        for i in range(proj.shape[0]):
            f.write("{} \n".format(proj[i]))

        f.write("modelview\n")
        for i in range(model_view.shape[0]):
            f.write("{} \n".format(model_view[i]))

        f.write("{}".format(viewport))

    mats = {}
    mats['viewport'] = viewport
    mats['modelview'] = model_view
    mats['projection'] = proj
    cPickle.dump(mats, open(root_dir + '/visual_mpc_core/agent/utils/proj_mats.pkl', 'wb'))

    largeimage = np.fromstring(img_string, dtype='uint8').reshape(
        (height, width, 3))[::-1, :, :]

    Image.fromarray(largeimage).save('testimg.png')

    img_string, width, height = viewer.get_depth()
    largedimage = np.fromstring(img_string, dtype=np.float32).reshape(
        (height, width, 1))[::-1, :, :]
    # plt.imshow(np.squeeze(largedimage))

    if t % 10 ==0:
        r, c = viewer.project_point(model.data.qpos)
        print 'model.data.qpos', model.data.qpos
        print "row, col", r, c
        r = int(r)
        c = int(c)

        largeimage[r,:] = [255, 255, 255]
        largeimage[:, c] = [255, 255, 255]
        plt.imshow(largeimage)
        plt.show()

    model.data.qvel.setflags(write=True)

    import pdb; pdb.set_trace()

viewer.finish()


j = Image.fromarray(largeimage)
j.save("img", "BMP")