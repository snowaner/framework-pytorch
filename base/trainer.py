#! /usr/bin/env python
#################################################################################
#     File Name           :     Trainer.py
#     Created By          :     WanCQ
#     Creation Date       :     [2018-04-24 05:49]
#     Last Modified       :     [2018-12-10 08:35]
#     Description         :      
#################################################################################

import time
import pdb

import torch
from torch import nn
from torch.autograd import Variable

class Trainer(object):
    # This class define the training procedure with forward and backward. The
    # function train() is the base function support iteration for each epoch.
    # It is suitable for a normal training procedure with a feature net and a
    # loss net in whatever the architecture is. However, when training
    # procedure is complicated like GAN, one need to reload _forward() and
    # _backward() in needs. This version just fit for a normal iteration.

    # Besides, the Trainer blocsk the network architecture and construct with
    # feature and loss two part. Thus, the trainer can hold still in many
    # situations and we just need to adjust network architecture for our needs
    def __init__(self, feature_net, loss_net, config=None):
        if config is not None:
            self._transfer_config(config)
        self.feature_net = feature_net
        self.loss_net = loss_net

    def train(self, epoch, data_loader, optimizer):
        raise NotImplementedError

    def _forward(self, inputs, labels):
        # we recommand all operations are defined in the graph. users can get
        # features by _get_features() and _get_losses() to get inner results.
        features = self.feature_net(inputs)
        self.loss = self.loss_net(features, labels)
        return self.loss

    def _get_features(self, inputs, labels=None):
        features = self.feature_net(inputs)
        return self.features

    def _get_losses(self, inputs, labels):
        self.loss = self.loss_net(inputs, labels)
        return self.loss

    def _backward(self, loss=None):
        self.optimizer.zero_grad()
        if loss is None:
            self.loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

    def _transfer_config(config):
        self.optim = config.get('train', 'optim')
        self.criterion = config.get('train', 'criterion')
        self.start_epoch = config.get('train', 'start_epoch')
        self.end_epoch = config.get('train', 'end_epoch')
        self.start_save = config.get('train', 'start_save')
        self.show_freq = config.get('train', 'show_freq')
