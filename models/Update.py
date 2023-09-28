#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
noise_std = 5.0
INPUT_DIM = 3*32*32
LAMBDA_0 = 10.0
LAMBDA_1 = 10.0
adv_stepsize = 2.0
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None,idx=1):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.perturbation_add_train = noise_std * torch.Tensor(np.random.normal(size=[1, INPUT_DIM]))
        self.matrix_mult_train = (noise_std / (np.sqrt(INPUT_DIM))) * torch.Tensor(
            np.random.normal(size=[INPUT_DIM, INPUT_DIM]))
        self.idx =idx

    def train(self, net, theta_,theta_1_,maxVar_,maxVar_1_ ):
        net.train()
        # train and update

        Classifier_train_op =  torch.optim.SGD([{'params': net.parameters(), 'lr': self.args.lr},
                                               {'params': theta_, 'lr': self.args.var_lr},
                                               {'params': theta_1_, 'lr': self.args.var_lr}]
                                               )
        Max_train_op =  torch.optim.SGD([{'params': maxVar_, 'lr': self.args.var_lr},
                                         {'params': maxVar_1_, 'lr': self.args.var_lr}
                                         ] )



        epoch_loss = []
        for iter in range(self.args.local_ep):

            print('iter',iter)
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                bs = images.shape[0]
                _data = images.reshape(-1, 3 * 32 * 32)
                _data = torch.matmul(_data, self.matrix_mult_train)
                _data = _data + self.perturbation_add_train

                data_perturb = torch.mul(theta_1_.clone(), _data) + theta_.clone()
                data_perturbed_max = data_perturb
                data_perturbed_pow_2_max = data_perturb ** 2

                images = data_perturb.clone().reshape(bs,3,32,32)
                net.zero_grad()
                log_probs = net(images.clone())

                loss = self.loss_func(log_probs, labels)
                # print('loss',loss)
                max_loss = torch.sum(data_perturbed_max * (maxVar_.clone()[self.idx,:] - maxVar_.clone().mean(dim=0)))
                max_loss_1 = torch.sum(data_perturbed_pow_2_max * (maxVar_1_.clone()[self.idx,:]- maxVar_1_.clone().mean(dim=0)))
                l1 = torch.sum(maxVar_.clone()[self.idx,:] ** 2)
                l2 = torch.sum(maxVar_1_.clone()[self.idx,:] ** 2)
                loss_2 = loss - LAMBDA_0 * l1- LAMBDA_1 * l2+ adv_stepsize * (max_loss + max_loss_1)

                Classifier_loss = loss.clone()
                Max_loss = - loss_2.clone()

                Max_train_op.zero_grad()
                Max_loss.backward(retain_graph=True)
                Max_train_op.step()

                Classifier_train_op.zero_grad()
                Classifier_loss.backward()
                Classifier_train_op.step()
                #
                # loss_2.backward()
                # optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print('theta_1_',theta_1_)
            # print('theta_', theta_)
            #
            # print('maxVar_',maxVar_)
            # print('maxVar_.grad',maxVar_.grad)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), theta_.clone(),theta_1_.clone(),maxVar_.clone(),maxVar_1_.clone()

