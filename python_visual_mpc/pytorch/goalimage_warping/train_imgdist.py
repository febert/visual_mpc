from goalimage_warper import GoalImageWarper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time

from python_visual_mpc.pytorch.dataloading.video_loader import make_video_loader

import tensorflow as tf
import imp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def MSE(x, y):
    loss = (x - y).pow(2).mean()
    return loss

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class ModelTrainer():
    def __init__(self):
        self.model = GoalImageWarper()
        self.use_cuda = True
        if self.use_cuda:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.batch_idx = 0

        parser = argparse.ArgumentParser()
        parser.add_argument("conf", help="path to configuration file")
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        args = parser.parse_args()
        hyperparams = imp.load_source('conf', args.conf)
        self.conf = conf = hyperparams.configuration
        data_conf = imp.load_source('conf', conf['data_dir'] + '/hyperparams.py').config

        self.train_loader = make_video_loader(data_conf, conf, train=True)
        self.test_loader = make_video_loader(data_conf, conf, train=False)

        self.init_tflogger()

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else: print("=> no checkpoint found at '{}'".format(args.resume))

        for epoch in range(conf['num_epochs']):
            self.train(epoch)
            self.test()

            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, conf['output_dir']+'/weights_ep{}.pth.tar'.format(epoch))

    def init_tflogger(self):
        self.sess = tf.InteractiveSession()

        self.trainloss_pl = tf.placeholder(tf.float32, shape=())
        self.train_summ_op = tf.summary.merge([tf.summary.scalar('train_loss', self.trainloss_pl)])

        self.valloss_pl = tf.placeholder(tf.float32, shape=())
        self.val_summ_op = tf.summary.merge([tf.summary.scalar('val_loss', self.valloss_pl)])

        self.summary_writer = tf.summary.FileWriter(self.conf['output_dir'], graph=self.sess.graph, flush_secs=10)

    def tf_logg(self, itr, train_loss=None, val_loss=None):
        if train_loss != None:
            [summary_str] = self.sess.run([self.train_summ_op], feed_dict={self.trainloss_pl: train_loss})
            self.summary_writer.add_summary(summary_str, itr)

        if val_loss != None:
            [summary_str] = self.sess.run([self.val_summ_op], feed_dict={self.valloss_pl: val_loss})
            self.summary_writer.add_summary(summary_str, itr)

    def sel_timesteps(self, images, epoch):
        sequence_length = images.size()[1]
        delta_t = np.ceil(sequence_length * float(epoch+1)/self.conf['num_epochs'])

        tstart = np.random.randint(0, sequence_length-delta_t)

        tend = tstart + np.random.randint(1,delta_t)
        I0 = images[:,tstart]
        I1 = images[:, tend]

        if self.batch_idx % 100 == 0:
            print 'tstart',tstart
            print 'tend', tend
        return I0, I1


    def train(self, epoch):
        self.model.train()
        epoch_len = len(self.train_loader)
        end = time.time()

        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time = time.time() - end

            if self.use_cuda:
                images = sample_batched['images'].cuda()
            I0, I1 = self.sel_timesteps(images, epoch)
            I0, I1 = Variable(I0), Variable(I1)
            self.optimizer.zero_grad()
            output = self.model(I0, I1)
            loss = MSE(output, I1)
            loss.backward()
            self.optimizer.step()

            self.global_step = (epoch)*epoch_len + self.batch_idx
            self.tf_logg(self.global_step, train_loss=loss.data[0])

            batch_time = time.time() - end
            end = time.time()

            if self.batch_idx % 10 == 0:
                print('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.global_step, epoch, self.batch_idx * self.conf['batch_size'], len(self.train_loader.dataset),
                    100. * self.batch_idx / len(self.train_loader), loss.data[0]))

                print 'avg loading time: {}, avg batch time: {}'.format(data_load_time, batch_time)

    def test(self):
        self.model.eval()
        loss_list = []

        test_delta_t = self.conf['sequence_length']-1

        for batch_idx, sample_batched in enumerate(self.train_loader):
            if self.use_cuda:
                images = sample_batched['images'].cuda()
            I0 = images[:,0]
            I1 = images[:,test_delta_t]

            I0, I1 = Variable(I0, volatile=True), Variable(I1)
            self.optimizer.zero_grad()
            output = self.model(I0, I1)

            val_loss = MSE(output, I1)
            loss_list.append(val_loss)

        loss_list = [l.data[0] for l in loss_list]

        avg_loss = np.mean(loss_list)
        self.tf_logg(self.global_step, train_loss=avg_loss)
        print('\nTest set: Average loss: {:.4f}\n'.format(avg_loss))


def main():
    ModelTrainer()

if __name__ == "__main__":
    main()

