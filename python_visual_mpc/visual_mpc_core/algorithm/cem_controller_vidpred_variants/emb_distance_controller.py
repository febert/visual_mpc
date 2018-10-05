from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred import CEM_Controller_Vidpred
import copy
import numpy as np
from python_visual_mpc.visual_mpc_core.algorithm.utils.make_cem_visuals import CEM_Visual_Preparation_FullImage
import pdb
from python_visual_mpc.visual_mpc_core.algorithm.cem_controller_vidpred_variants.full_image_reg_controller import Full_Image_Reg_Controller

from glow.setup_dist_func import setup_embedding

def cosine_dist(a, b):
    distances = []
    n_layer = len(a)

    def dist(a, b):
        return np.sum(a*b, 1)/np.linalg.norm(a)/np.linalg.norm(b)
    for i in range(n_layer):
        distances.append(dist(a[i], b[i]))

    return np.mean(distances)

def mse_dist(emb1, emb2):
    distances = []
    n_layer = len(emb1)
    bsize = emb1[0].shape[0]

    for i in range(n_layer):
        distances.append(np.mean(np.square(emb1[i] - emb2[i]).reshape([bsize, -1]), 1))

    return np.mean(np.stack(distances, 0), 0)

def append_black(images):
    bsize, height, width, channels = images.shape
    app = np.zeros([bsize, 64 - height, width, channels])
    return np.concatenate([images, app], 1)

class Embedding_Dist_Controller(Full_Image_Reg_Controller, CEM_Controller_Vidpred):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super().__init__(ag_params, policyparams, gpu_id, ngpu)
        self.visualizer = CEM_Visual_Preparation_FullImage()

    def build_costnet(self, ag_params, gpu_id=0):
        self.make_embedding = setup_embedding(self.bsize)

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):
        goal_images = np.tile(self.goal_image[None], [self.bsize, 1, 1, 1,1,1])   # shape b,t,n, r, c, 3

        gen_images = append_black(gen_images[:,-1,0])
        goal_images = append_black(goal_images[:,-1,0])

        scores = np.zeros([self.bsize,self.len_pred])

        t_mult = np.ones([self.seqlen - self.ncontxt])
        t_mult[-1] = self._hp.finalweight

        for t in range(self.len_pred):
            pred_embedding = self.make_embedding(gen_images)
            goal_embedding = self.make_embedding(goal_images)

            dist = ['mse']
            if 'cosine' in dist:
                scores[:, t] += cosine_dist(pred_embedding, goal_embedding)
            if 'mse' in dist:
                scores[:, t] += mse_dist(pred_embedding, goal_embedding)

        self.vd.goal_image = self.goal_image
        return scores

    def act(self, t, i_tr, images, goal_image, state, desig_pix, goal_pix):
        self.goal_pix = np.array(goal_pix).reshape((self.ncam, self.ntask, 2))
        return super().act(t, i_tr, images, goal_image, state, desig_pix, goal_pix)

