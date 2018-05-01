import gym
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import mujoco_py
print((mujoco_py.__file__))

path = "/home/frederik/Documents/catkin_ws/src/visual_mpc/mjc_models/cartgripper_noautogen.xml"
env = MujocoEnv(path, 1)


for i in range(100):

    env.do_simulation(np.zeros(3), 10)
    env.render()

    im = env.get_image()
    plt.imshow(im)




