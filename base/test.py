#! /usr/bin/env python
#################################################################################
#     File Name           :     Test.py
#     Created By          :     WanCQ
#     Creation Date       :     [2018-04-24 08:12]
#     Last Modified       :     [2018-12-10 08:35]
#     Description         :      
#################################################################################

import time
import sys
import pdb
import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from Utils import Logger as L

from sklearn.metrics import average_precision_score

class Test(object):
    # This class defines several testing functions for different tasks, like
    # classification loss/accuracy, reid mAP/rank-1. Meanwhile, some general
    # functions like extract features and visualization is also provided.
    def __init__(self, config=None):
        if config is None:
            raise NotImplementedError ('config file is needed in this version!')
        else:
            _config_extract(config)
        net = __import__(config.network)
        self.feature_net = net.feature_net

    def _config_extract(config):
        self.test_type = config.get('test', 'test_type')
        self.task_type = config.get('normal', 'task_type')

    def extract(data_loader, features=None, labels=None):
        self.feature_net.eval()
        for i, inputs in enumerate(data_loader):
            if self.task_type == 'classification':
                inputs, label = split_data(inputs)
            elif self.task_type == 'reid':
                inputs, pids, cams = split_data(inputs)
            outputs = self.feature_net(inputs)
            if features is None:
                features = outputs
            else:
                features = torch.concat(features, outputs)
            if labels is None:
                labels = label
            else:
                labels = torch.concat(labels, outputs)
        return features, labels

    def pairwise_distance(feature1, feature2):
        assert type(feature1)=='torch.tensor'
        assert type(feature2)=='torch.tensor'
        m, n = feature1.size(0), feature2.size(0)
        dist = torch.pow(feature1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(feature2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, feature1, feature2.t())
        return dist

    def meanap(feature1, feature2, query_ids, gallery_ids,
               query_cams, gallery_cams, dist=None):
        if dist is None:
            dist = self.pairwise_distance(feature1, feature2)
        dist = dist.to_numpy(dist)
        indices = np.argsort(dist, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        aps = []
        for i in range(m):
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            y_true = matches[i, valid]
            y_score = -dist[i][indices[i]][valid]
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)

    def cmc(feature1, feature2, query_ids, gallery_ids, query_cams,
            gallery_cams, dist=None, topk=100):
        if dist is None:
            dist = self.pairwise_distance(feature1, feature2)
        dist = dist.to_numpy(dist)
        m, n = dist.shape
        indices = np.argsort(dist, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        ret = np.zeros(topk)
        num_valid_querier = 0
        for i in range(m):
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            index = np.nonzero(matches[i, valid])[0]
            for j, k in enumerate(index):
                if k - j >= topk: break
                ret[k-j] += 1
                break
            num_valid_queries += 1
            if num_valid_queries == 0:
                raise RuntimeError("No valid query")
            return ret.cumsum() / num_valid_queries

    def accuracy(output, target, topk=(1,)):
        output, target = to_torch(output), to_torch(target)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret
