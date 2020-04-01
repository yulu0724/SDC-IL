from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from utils import to_numpy
import numpy as np

from utils.meters import AverageMeter
import pdb

def extract_features(model, data_loader, print_freq=1, metric=None):
    model=model.cuda()
    model.eval()
  
    features = []
    labels = []
 
    for i, data in enumerate(data_loader,0):
        imgs, pids=data
      
        inputs = imgs.cuda()
        with torch.no_grad():
            _,outputs = model(inputs)
            outputs = outputs.cpu().numpy()
     
        if features==[]:
            features=outputs
            labels=pids
        else:
            features=np.vstack((features,outputs))
            labels = np.hstack((labels,pids))

    return features, labels


def pairwise_distance(features, metric=None):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    dist = dist.expand(n, n)
    dist = dist + dist.t()
    dist = dist - 2 * torch.mm(x, x.t()) + 1e5 * torch.eye(n)
    dist = torch.sqrt(dist)
    return dist


def pairwise_similarity(features):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    similarity = torch.mm(x, x.t()) - 1e5 * torch.eye(n)
    return similarity


