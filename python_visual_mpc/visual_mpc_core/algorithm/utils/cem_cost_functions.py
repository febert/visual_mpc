import numpy as np
import pdb
import matplotlib.pyplot as plt


def mse_based_cost(gen_images, goal_image, hp, normalize):
    """
    computes planning cost based on MSE error between predicted image and goal_image(s)
    :param gen_images:
    :param goal_image:
    :param hp:
    :param normalize: wether to normalize the images before computing the difference
    :return:
    """
    if normalize:
        gen_images = gen_images/np.sum(gen_images.reshape[gen_images.shape[0], gen_images.shape[1], -1])
        assert len(goal_image.squeeze().shape) == 3 # make sure we only have a single goalimage
        goal_image = goal_image /np.sum(goal_image)

    sq_diff = np.square(gen_images - goal_image[None])
    mean_sq_diff = np.mean(sq_diff.reshape([sq_diff.shape[0],sq_diff.shape[1],-1]), -1)

    per_time_multiplier = np.ones([1, gen_images.shape[1]])
    per_time_multiplier[:, -1] = hp.finalweight
    return np.sum(mean_sq_diff * per_time_multiplier, axis=1)