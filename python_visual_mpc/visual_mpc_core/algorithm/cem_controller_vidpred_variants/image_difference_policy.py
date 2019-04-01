from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.utils.make_cem_visuals import CEM_Visual_Preparation_FullImage
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.full_image_reg_controller import Full_Image_Reg_Controller
from python_visual_mpc.visual_mpc_core.algorithm.utils.cem_cost_functions import mse_based_cost
import cv2
from scipy import ndimage


class Image_Difference_Policy(Full_Image_Reg_Controller, CEM_Controller_Vidpred):

    """
    This policy is the baseline comparison for the die task. It uses a pre-saved image of the sensor with
    no contact. To compute actions, it takes the current sensor image and computes a difference image.
    The image is then converted to grayscale, thresholded, and the mathematical centroid of the resulting matrix
    is computed. By doing the same with the goal image as well, it attempts to determine the contact locations at the
    current timestep and at the goal, and makes a movement to get the current image closer.
    """

    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        CEM_Controller_Vidpred.__init__(self, ag_params, policyparams, gpu_id, ngpu)
        self._hp = self._default_hparams()
        self.override_defaults(policyparams)
        self.pred_len = self.seqlen - self.ncontxt
        self.visualizer = CEM_Visual_Preparation_FullImage()
        self.zero_image = cv2.imread('/home/stian/Documents/visual_mpc/experiments/gelsight/zero_image.jpg')[:, :, ::-1]
        self.MOVEMENT_SCALE = 10 # Experimentally calibrated

    def _default_hparams(self):
        default_dict = {
        }
        parent_params = super()._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):
        scores = mse_based_cost(gen_images, self.goal_image, self._hp)
        self.vd.goal_image = self.goal_image
        return scores

    def centroid(self, img, empty):
        difference_image = np.abs(img.astype(float) - empty.astype(float))
        difference_image = difference_image.astype('uint8')
        diff_img = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)

        ret, diff_img = cv2.threshold(diff_img, 50, 255, cv2.THRESH_TOZERO)
        cent = list(ndimage.measurements.center_of_mass(diff_img))
        print('Calculated centroid (pixel coordinates): {}'.format(cent))
        if np.isnan(cent[0]):
            cent[0] = 0
        if np.isnan(cent[1]):
            cent[1] = 0
        return np.array([cent[0], cent[1]]).astype(float), (not cent[0]) and (not cent[1])

    def act(self, t, i_tr, images, goal_image, state):
        last_goal_image = (goal_image[-1][0]*255).astype('uint8')
        curr = np.copy(images[-1][0])
        curr_centroid, fail_curr = self.centroid(curr, self.zero_image)
        goal_centroid, fail_goal = self.centroid(last_goal_image, self.zero_image)
        grad = (goal_centroid - curr_centroid) / self.MOVEMENT_SCALE
        print(grad)
        if fail_curr or fail_goal:
            return {'actions': np.array([0, 0, 0])}
        return {'actions': np.array([grad[0], -grad[1], 0])}

