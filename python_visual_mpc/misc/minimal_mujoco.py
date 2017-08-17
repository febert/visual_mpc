from copy import deepcopy

import numpy as np

import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *

from PIL import Image
import cPickle

import os
cwd = os.getcwd()

BASE_DIR = '/'.join(str.split(cwd, '/')[:-2])
filename = BASE_DIR + '/mjc_models/pushing2d_controller_nomarkers.xml'

model= mujoco_py.MjModel(filename)
viewer = mujoco_py.MjViewer(visible=True, init_width=480, init_height=480)
viewer.start()
viewer.set_model(model)
viewer.cam.camid = 0
print viewer.cam.camid


T = 1
for t in range(T):

    viewer.loop_once()

    mj_U = np.zeros(2)
    model.data.ctrl = mj_U
    model.step()

    img_string, width, height = viewer.get_image()
    largeimage = np.fromstring(img_string, dtype='uint8').reshape(
        (480, 480, 3))[::-1, :, :]

    model.data.qvel.setflags(write=True)

    import pdb; pdb.set_trace()



j = Image.fromarray(largeimage)
j.save("img", "BMP")