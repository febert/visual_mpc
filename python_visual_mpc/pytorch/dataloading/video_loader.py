import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import imp
import glob
import cv2
import cPickle
import tarfile

import time

from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory

from dataloading_utils import get_maxtraj, make_traj_name_list

# Ignore warnings
import warnings
from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import comp_single_video
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class Crop(object):
    def __init__(self, rowstart, rowend):
        self.rowstart = rowstart
        self.rowend = rowend
    def __call__(self, sample):
        images, states, actions = sample['images'], sample['states'], sample['actions']
        images = images[:, self.rowstart:self.rowend]
        return {'images': images, 'states': states, 'actions': actions}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, states, actions = sample['images'], sample['states'], sample['actions']
        # swap color axis because
        # numpy image: T x H x W x C
        # torch image: T x C X H X W
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images), 'states': torch.from_numpy(states), 'actions': torch.from_numpy(actions)}

def read_images(conf, trajname, traj, tar):
    trajind = 0  # trajind is the index in the target trajectory
    end = conf['T']

    take_ev_nth_step = 1
    for dataind in range(0, end, take_ev_nth_step):  # dataind is the index in the source trajetory

        im_filename = trajname + '/images/im{}.png'.format(dataind)

        img_stream = tar.extractfile(im_filename[1:])
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        traj._sample_images[trajind] = cv2.imdecode(file_bytes, 1)

        trajind += 1

class VideoDataset(Dataset):
    def __init__(self, dataconf, trainconf, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.agentparams = dataconf['agent']
        self.policyparams = dataconf['policy']
        self.conf = dataconf

        if train:
            self.root_dir = self.agentparams['data_files_dir']
        else: self.root_dir = '/'.join(str.split(self.agentparams['data_files_dir'] , '/')[:-1]) + '/test'

        self.transform=transforms.Compose([Crop(trainconf['startrow'], trainconf['endrow']), ToTensor()])

        get_maxtraj(self.root_dir)
        self.traj_name_list = make_traj_name_list(dataconf, self.root_dir, shuffle=False)

    def __len__(self):
        return len(self.traj_name_list)

    def __getitem__(self, idx):
        t0 = time.time()

        trajname = self.traj_name_list[idx]
        traj = Trajectory(self.agentparams)

        with tarfile.open(trajname + "/traj.tar") as tar:
            read_images(self.agentparams, trajname, traj, tar)
            pkl_file_stream = tar.extractfile(trajname[1:] + '/state_action.pkl')
            pkldata = cPickle.load(pkl_file_stream)

        states = np.concatenate([pkldata['qpos'],  pkldata['qvel']], axis=1)
        actions = pkldata['actions']

        images = traj._sample_images.astype(np.float32)/255.
        sample = {'images': images, 'states':states, 'actions':actions}

        if self.transform:
            sample = self.transform(sample)

        if idx % 10 ==0:
            print 'tread images: ', time.time() - t0

        return sample

def make_video_loader(dataconf, trainconf, train=True):
    dataset = VideoDataset(dataconf, trainconf, train=train)

    return DataLoader(dataset, batch_size=trainconf['batch_size'],
                            shuffle=True, num_workers=20)

def test_videoloader():
    hyperparams_file = '/home/frederik/Documents/catkin_ws/src/visual_mpc/pushing_data/cartgripper_bench_conf/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    config = hyperparams.config

    trainconf = {'startrow': 15,
                 'endrow':63,
                 'batch_size':32}

    dataloader = make_video_loader(config, trainconf)

    deltat = []
    end = time.time()
    for i_batch, sample_batched in enumerate(dataloader):

        images = sample_batched['images']
        states = sample_batched['states']
        actions = sample_batched['actions']

        deltat.append(time.time() - end)
        if i_batch % 10 == 0:

            print 'tload{}'.format(time.time() - end)
            print 'average time:',np.average(np.array(deltat))
        end = time.time()

        # print i_batch, images.size(), states.size(), actions.size()

        images = images.numpy().transpose((0, 1, 3, 4, 2))
        file = '/'.join(str.split(config['agent']['data_files_dir'], '/')[:-1]) + '/example'
        comp_single_video(file, images)
        #
        # # observe 4th batch and stop.
        # if i_batch == 10:
        #     break

if __name__ == '__main__':
    test_videoloader()