#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
import math
import os

MODEL_XML = "/home/frederik/Documents/catkin_ws/src/visual_mpc/mjc_models/cartgripper_noautogen.xml"
import matplotlib.pyplot as plt

model = load_model_from_path(MODEL_XML)
# model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    sim.step()
    viewer.render()

    # image = sim.render(400, 400, camera_name="maincam")
    # plt.imshow(image)
    # plt.show()

    if t > 100 and os.getenv('TESTING') is not None:
        break