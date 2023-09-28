#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

import numpy as np

def FedAvg(w,theta_locals):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def FedOT(w,theta_locals,theta_1_locals,maxVar_locals,maxVar_1_locals):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    theta_ =  torch.Tensor(np.array(theta_locals)).mean(dim=0)
    theta_1 =  torch.Tensor(np.array(theta_1_locals)).mean(dim=0)
    maxVar =  torch.Tensor(np.array(maxVar_locals)).mean(dim=0)
    maxVar_1 = torch.Tensor(np.array(maxVar_1_locals)).mean(dim=0)

    # theta_ = torch.tensor(np.array(theta_locals), dtype=torch.float32, requires_grad=True).mean(dim=0)
    # theta_1 = torch.tensor(np.array(theta_1_locals), dtype=torch.float32, requires_grad=True).mean(dim=0)
    # maxVar = torch.tensor(np.array(maxVar_locals), dtype=torch.float32, requires_grad=True).mean(dim=0)
    # maxVar_1 = torch.tensor(np.array(maxVar_1_locals), dtype=torch.float32, requires_grad=True).mean(dim=0)

    return w_avg,theta_,theta_1,maxVar,maxVar_1
