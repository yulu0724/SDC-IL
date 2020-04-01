from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable


def orth_reg(net, loss, cof=1):
    orth_loss = 0
    for m in net.modules():
        if isinstance(m, nn.Linear):
            w = m.weight
            # embedding dimension
            dimension = w.size()[0]
            # eye_ = torch.eye(dimension).cuda()
            eye_ = Variable(torch.eye(dimension),  requires_grad=False).cuda()
            diff = torch.matmul(w, w.t()) - eye_

            # ignore the diagonal elements
            mask_ = eye_ == 0
            diff = torch.masked_select(diff, mask=mask_)

            _loss = torch.mean(torch.abs(diff))
            orth_loss += cof*_loss
            loss = loss + orth_loss
    return loss
