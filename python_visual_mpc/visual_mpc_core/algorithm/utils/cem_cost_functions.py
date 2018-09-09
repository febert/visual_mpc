import numpy as np


def mse_based_cost(gen_images, goal_image, hp):
    """
    computes planning cost based on MSE error between predicted image and goal_image(s)
    :param gen_images:
    :param goal_image:
    :param hp:
    :return:
    """

    sq_diff = np.square(gen_images - goal_image[None])
    mean_sq_diff = np.mean(sq_diff.reshape([sq_diff.shape[0],sq_diff.shape[1],-1]), -1)

    per_time_multiplier = np.ones([1, gen_images.shape[1]])
    per_time_multiplier[:, -1] = hp.finalweight
    return np.sum(mean_sq_diff * per_time_multiplier, axis=1)