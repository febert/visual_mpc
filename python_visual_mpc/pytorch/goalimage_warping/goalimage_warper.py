from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

plt.ion()   # interactive mode

use_cuda = torch.cuda.is_available()


class Sequential_cache_output(nn.Sequential):
    def __init__(self, *args):
        super(Sequential_cache_output, self).__init__(*args)
        self.output_cache = []

    def forward(self, input):
        self.output_cache = []
        for module in self._modules.values():
            input = module(input)
            self.output_cache.append(input)
        return input

def conv_bilin_layer(n_in, n_out, k=3, smp_dim=None, relu=True, upsmp=True):
    l = []
    l.append(nn.Conv2d(n_in, n_out, kernel_size=k, padding=k/2))
    if relu:
        l.append(nn.ReLU(True))
    if upsmp:
        l.append(nn.Upsample(smp_dim, mode='bilinear'))
    return nn.Sequential(*l)

class Conv_Bilin_Module(nn.Module):
    def __init__(self, n_in, n_out, k=3, smp_dim=None, relu=True, upsmp=True):
        super(Conv_Bilin_Module, self).__init__()
        l = []
        l.append(nn.Conv2d(n_in, n_out, kernel_size=k, padding=k / 2))
        if relu:
            l.append(nn.ReLU(True))
        if upsmp:
            l.append(nn.Upsample(smp_dim, mode='bilinear'))

        self.func = nn.Sequential(*l)

    def forward(self, x):
        return self.func(x)


class GoalImageWarper(nn.Module):
    def __init__(self):
        super(GoalImageWarper, self).__init__()

        # input is N,3,40,64
        # self.l1 = conv_bilin_layer(6, 32, 5, [20, 32])
        # self.l2 = conv_bilin_layer(32, 64, 3, [10, 16])
        # self.l3 = conv_bilin_layer(64, 128, 3, [5, 8])
        # self.l4 = conv_bilin_layer(128, 64, 3, [10, 16])
        # self.l5 = conv_bilin_layer(64, 32, 3, [20, 32])
        # self.l6 = conv_bilin_layer(32, 16, 3, [40, 64])
        # self.l7 = conv_bilin_layer(16, 2, 5, relu=False, upsmp=False)

        self.make_flowfield = Sequential_cache_output(
            Conv_Bilin_Module(6, 32, 5, [20, 32]),
            Conv_Bilin_Module(32, 64, 3, [10, 16]),
            Conv_Bilin_Module(64, 128, 3, [5, 8]),
            Conv_Bilin_Module(128, 64, 3, [10, 16]),
            Conv_Bilin_Module(64, 32, 3, [20, 32]),
            Conv_Bilin_Module(32, 16, 3, [40, 64]),
            Conv_Bilin_Module(16, 2, 5, relu=False, upsmp=False),
        )

    # Define appearance flow operator
    def apply_app_flow(self, x, flow):
        theta = np.repeat(np.array([[1, 0, 0], [ 0, 1, 0]]).reshape(1,2,3), x.size()[0], axis=0)
        theta = torch.from_numpy(theta.astype(np.float32)).cuda()
        identity_grid = F.affine_grid(theta, x.size())
        sample_pos = identity_grid + flow.permute(0,2,3,1)
        x = F.grid_sample(x, sample_pos)
        return x

    def forward(self, I1, I2):
        I1_I2 = torch.cat([I1,I2], 1)

        # f = self.l1(I1_I2)
        # f = self.l2(f)
        # f = self.l3(f)
        # f = self.l4(f)
        # f = self.l5(f)
        # f = self.l6(f)
        # f = self.l7(f)

        f = self.make_flowfield(I1_I2)
        self.f = f

        warped = self.apply_app_flow(I1,f)

        return warped