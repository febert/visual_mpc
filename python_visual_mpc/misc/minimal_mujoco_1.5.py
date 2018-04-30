#!/usr/bin/env python3
"run this with LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so"
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
import numpy as np
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer, MjViewerBasic
import math
import os

from python_visual_mpc.visual_mpc_core.agent.utils.convert_world_imspace_mj1_5 import project_point
from python_visual_mpc import __file__ as python_vmpc_path
root_dir = '/'.join(str.split(python_vmpc_path, '/')[:-2])
MODEL_XML = root_dir + "/mjc_models/cartgripper_noautogen.xml"
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

model = load_model_from_path(MODEL_XML)
sim = MjSim(model)
viewer = MjViewerBasic(sim)
# viewer = MjViewer(sim)
t = 0

height, width = 480, 640

for t in range(1000):
    sim.data.ctrl[0] = 100
    sim.data.ctrl[1] = 100

    cam_xmat = sim.data.cam_xmat.reshape((3, 3))
    print(cam_xmat)
    cam_xpos = sim.data.cam_xpos
    print(cam_xpos )

    with open('cam_params_1.5.txt','w') as f:
        f.write("cam_xmat \n")
        for i in range(cam_xmat.shape[0]):
            f.write("{} \n".format(cam_xmat[i]))

        f.write("cam_xpos\n")
        for i in range(cam_xpos .shape[0]):
            f.write("{} \n".format(cam_xpos [i]))
    t += 1
    sim.step()
    # viewer.render()
    largeimage = viewer.render(width, height, "maincam")[::-1, :, :]
    # largeimage = sim.render(width, height, camera_name="maincam")[::-1, :, :]

    r, c = project_point(sim.data.qpos)
    print(('model.data.qpos', sim.data.qpos))
    print(("row, col", r, c))
    r = int(r)
    c = int(c)

    largeimage[r,:] = [255, 255, 255]
    largeimage[:, c] = [255, 255, 255]
    plt.imshow(largeimage)
    # plt.savefig('/outputs/testimg.png')
    plt.show()

    # import sys
    # sys.exit()

