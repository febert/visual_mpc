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
        self.pred_len = self.seqlen - self.ncontxt
        self.visualizer = CEM_Visual_Preparation_FullImage()

    def build_costnet(self, ag_params, gpu_id=0):
        self.make_embedding = setup_embedding(self.bsize)

    def eval_planningcost(self, cem_itr, gen_distrib, gen_images):
        goal_images = np.tile(self.goal_image[None], [self.bsize, 1, 1, 1,1,1])   # shape b,t,n, r, c, 3
        gen_images = append_black(gen_images[:,-1,0])
        goal_images = append_black(goal_images[:,-1,0])

        pred_embedding = self.make_embedding(gen_images)
        goal_embedding = self.make_embedding(goal_images)
        # pred_embedding = [np.zeros([self.bsize, 10])]
        # goal_embedding = [np.zeros([self.bsize, 10])]

        dist = ['mse']
        scores = np.zeros(self.bsize)
        if 'cosine' in dist:
            scores += cosine_dist(pred_embedding, goal_embedding)
        if 'mse' in dist:
            scores += mse_dist(pred_embedding, goal_embedding)

        self.vd.goal_image = self.goal_image
        return scores

    def act(self, t, i_tr, images, goal_image, state, desig_pix, goal_pix):
        self.goal_pix = np.array(goal_pix).reshape((self.ncam, self.ntask, 2))
        return super().act(t, i_tr, images, goal_image, state, desig_pix, goal_pix)

