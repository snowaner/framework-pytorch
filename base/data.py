#! /usr/bin/env python
#################################################################################
#     File Name           :     Data.py
#     Created By          :     WanCQ
#     Creation Date       :     [2018-04-24 03:11]
#     Last Modified       :     [2018-12-04 03:49]
#     Description         :     base data implementation
#################################################################################

from __future__ import absolute_import

import os.path as osp
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision.transform import Compose

import pdb

class Data(object):
    def __init__(self, config=None, **kwargs):
        # init params for Data Class
        if config is not None:
            _transfer_config(config)

    def load_data(self, file_name, mode=None, **kwargs):
        _train = ['train.txt']
        _test = ['test.txt', 'val.txt', 'query.txt', 'gallery.txt']
        if mode is None:
            if osp.basename(file_name) in _train:
                params = self.train
            elif osp.basename(file_name) in _test:
                params = self.test
            else:
                params = self.train
                print('Warning: the list file {} is not in the pre-defined \
                      types, we regard this file as training list and adopt \
                      training settings to generate the dataloader!'
                      .format(osp.basename(file_name)))
        elif mode == 'train':
            params = self.train
        elif mode == 'test':
            params = self.test
        else:
            raise RuntimeError('data loader has a wrong mode, only '\train'\ and \
                               '\test'\ are available')
        # avoid not defining list_dir both in config file or code
        if params['list_dir'] is None:
            if file_name in _train or file_name in _test:
                params['list_dir'] = params['data_dir']
            else:
                raise RuntimeError('The path of label list is required, you \
                                   need to add this in config file or in code.')
        if not osp.exists(params['list_dir']):
            raise RuntimeError('Invalid path of list_dir {}'.format(params[list_dir]))
        # fetch the label list.
        data_list = readtxt(osp.join(params['list_dir'], file_name),
                            params['label_info'])
        # automatically generate the transformer based on augmentation
        # definition.
        transformer = []
        for aug in params['augmentation']:
            if aug == 'resize':
                transformer.append(RectScale(self.height, self.width))
            elif aug == 'random_crop':
                transformer.append(RandomSizedRectCrop(self.height, self.width))
            elif aug == 'random_horizontal_flip':
                transformer.append(RandomHorizontalFlip())
            elif aug == 'normlize':
                transformer.append(Normalize(mean=self.norm_mean, std=self.var))
            elif aug == 'totensor':
                transformer.append(ToTensor())
            elif aug == 'random_erasing':
                transformer.append(RE(mean=re_mean))
        params['transformer'] = Compose(transformer)
        # Once someone attempt to directly call this function with **kwargs, or
        # add some new settings regardless of config file, the following part
        # can update these settings.
        for key in **kwargs.keys():
            params[key] = **kwargs.keys()
        return DataLoader(
            P(osp.join(params.list_dir, file_name),
              transform=params['transformer']),
            batch_size = params['batch_size'],
            shuffle = params['shuffle'],
            pin_memory = params['pin_memory'],
            drop_last = params['drop_last'],
            sampler = params['sampler']
        )


    # Transform the value in config file into self.value
    def _transfer_config(self, config):
        self.train = {}
        self.test = {}
        root = config.get('data', 'root')
        list_dir = config.get('data', 'list_dir')
        dataset = config.get('data', 'dataset')
        if root is not None:
            if data_dir is not None:
                data_dir = osp.join(root, data_dir)
                if list_dir is not None:
                    list_dir = osp.join(root, list_dir)
        if data_dir is None:
            raise RuntimeError('The path of dataset are required, \
                               you need to check the config file \
                               and add \'data_dir\'!')
        if not osp.exist(list_dir):
            raise RuntimeError('Invalid path for list_dir!')
        if not osp.exist(data_dir):
            raise RuntimeError('Invalid path for data_dir!')
        num_workers = config.get('data', 'num_workers')
        if num_workers is None:
            num_workers = 4
        label_info = config.get('data', 'label_info')
        if label_info is None:
            label_info = ['fname', 'label']
        self.train['data_dir'] = data_dir
        self.train['list_dir'] = list_dir
        self.train['num_workers'] = num_workers
        self.train['label_info'] = label_info
        self.test['data_dir'] = data_dir
        self.test['list_dir'] = list_dir
        self.test['num_workers'] = num_workers
        self.test['label_info'] = label_info
        self.train['augmentation'] = config.get('data', 'train_augmentation')
        self.test['augmentation'] = config.get('data', 'test_augmentation')
        if train_aug is None:
            self.train['augmentation'] = config.get('data', 'augmentation')
            if self.train['augmentation'] is None:
                self.train['augmentation'] = 'resize, totensor, normlize'
        if self.test_aug is None:
            self.test['augmentation'] = 'resize, totensor, normlize'
        self.norm_mean = config.get('data', 'norm_mean')
        self.norm_var = config.get('data', 'norm_var')
        self.re_mean = config.get('data', 're_mean')
        self.height = config.get('data', 'height')
        self.width = config.get('data', 'width')
        if self.norm_mean is None:
            self.norm_mean = [0.485, 0.456, 0.406]
        if self.norm_var is None:
            self.norm_var = [0.229, 0.224, 0.225]
        if self.re_mean is None:
            self.re_mean = [0.0, 0.0, 0.0]
        if self.height is None:
            self.height = 256
        if self.width is None:
            self.width = 256
        self.pin_memory = config.get('data', 'pin_memory')
        if self.pin_memory is None:
            self.train['pin_memory'] = True
            self.test['pin_memory'] = True
        self.train['shuffle'] = config.get('data', 'train_shuffle')
        self.test['shuffle'] = config.get('data', 'test_shuffle')
        if self.train['shuffle'] is None:
            self.train['shuffle'] = True
        if self.test['shuffle'] is None:
            self.test['shuffle'] = False
        self.train['drop_last'] = config.get('data', 'train_drop_last')
        self.test['drop_last'] = config.get('data', 'test_drop_last')
        if self.train['drop_last'] is None:
            self.train['drop_last'] = True
        if self.test['drop_last'] is None:
            self.test['drop_last'] = False
        self.train['sampler'] = None
        self.test['sampler'] = None
        self.train['transformer'] = None
        self.test['transformer'] = None
