#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
from itertools import permutations
import numpy as np
import pandas as pd
import torch
import pdb
from collections import Counter

def iid(targets, num_users, data_frac=1.0):
    """
    Sample I.I.D. client data from dataset
    :param targets: dataset.targets
    :param num_users:
    :return: dict of {user_id: image indexs}
    """

    if isinstance(targets, torch.Tensor):
        targets_array = targets.numpy()
    elif isinstance(targets, list):
        targets_array = np.array(targets)
    else:
        targets_array = targets

    all_idxs = np.array([i for i in range(len(targets_array))])
    n_classes = len(np.unique(targets_array))
    dict_users = {i:[] for i in range(num_users)}

    for c in range(n_classes):
        num_items_per_cls = math.floor(len(targets_array[targets_array==c]) / num_users)
        for i in range(num_users):
            ids_cls = all_idxs[np.where(targets_array[all_idxs]==c)[0]]
            ids_select_cls = np.random.choice(ids_cls, num_items_per_cls, replace=False)
            ids_select_cls = ids_select_cls.tolist()
            dict_users[i] += ids_select_cls
            all_idxs = np.array(list(set(all_idxs) - set(ids_select_cls)))
    for i in dict_users.keys():
        random.shuffle(dict_users[i])

    for i in dict_users.keys():
        data_num = int(len(dict_users[i]) * data_frac)
        dict_users[i] = dict_users[i][:data_num]
    return dict_users

def imbalance(targets, num_users, data_frac=1.0):
    """
    Sample I.I.D. client data from dataset
    :param targets: dataset.targets
    :param num_users:
    :return: dict of {user_id: image indexs}
    """

    if isinstance(targets, torch.Tensor):
        targets_array = targets.numpy()
    elif isinstance(targets, list):
        targets_array = np.array(targets)
    else:
        targets_array = targets

    all_idxs = np.array([i for i in range(len(targets_array))])
    n_classes = len(np.unique(targets_array))
    dict_users = {i:[] for i in range(num_users)}

    for c in range(n_classes):
        num_items_per_cls = math.floor(len(targets_array[targets_array==c]) / num_users)
        for i in range(num_users):
            ids_cls = all_idxs[np.where(targets_array[all_idxs]==c)[0]]
            ids_select_cls = np.random.choice(ids_cls, num_items_per_cls, replace=False)
            ids_select_cls = ids_select_cls.tolist()
            dict_users[i] += ids_select_cls
            all_idxs = np.array(list(set(all_idxs) - set(ids_select_cls)))
    for i in dict_users.keys():
        random.shuffle(dict_users[i])

    for i in dict_users.keys():
        data_num = int(len(dict_users[i]) * data_frac)
        dict_users[i] = dict_users[i][:data_num]
    return dict_users


def noniid_label(targets, num_users, shard_per_user, data_frac=1.0, client_cls_set=[], is_uniform=False):
    """
    Sample non-I.I.D client data.
    :param targets:
    :param num_users:
    :param shard_per_user: the number of classes assigned to every user, e.g. 2=each user only have samples from 2 classes.
    :return dict_users: dict{'userid': data}
    :return client_cls_set: list(each user' classes)
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    #
    idxs_dict = {}  # {'label': 'dataset samples' id'}
    if type(targets) == torch.Tensor:
        targets_array = targets.numpy()
    elif type(targets) == list:
        targets_array = np.array(targets)
    else:
        targets_array = targets

    for i in range(len(targets)):
        label = targets_array[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    for i in idxs_dict.keys():
        data_num = int(len(idxs_dict[i]) * data_frac)
        idxs_dict[i] = idxs_dict[i][:data_num]

    num_classes = len(np.unique(targets_array))
    assert shard_per_user <= num_classes, "The number of classes assigned to each user must <= num_classes"
    shard_per_class = int(shard_per_user * num_users / num_classes)
    # reshape the array of each label to (shard_pre_class, *)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        """
            data of each class need to split into shared_per_class blocks.
            According to data's num and block's num of the class, identify the num of each block of the class.
            If we have more data which can't be modded, put them into former blocks one by one.
        """
        num_leftover = len(x) % shard_per_class
        # get indexes of [len-num_leftover, ..., len]
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x
    # -- assign each user's classes.
    if len(client_cls_set) == 0:
        client_cls_set = list(range(num_classes)) * shard_per_class
        if is_uniform is False:
            random.shuffle(client_cls_set)
        client_cls_set = np.array(client_cls_set).reshape((num_users, -1))

    # -- divide and assign
    # give each user his corresponding class data according to rand_set_label.
    for i in range(num_users):
        rand_set_label = client_cls_set[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))  # pop‘s function is to delete the idx'th element and return it.
        dict_users[i] = np.concatenate(rand_set)

    # -- check whether the assignment results are resonable.
    test = []
    for key, value in dict_users.items():
        x = np.unique(targets_array[value])
        assert (len(x) <= shard_per_user)
        test.append(value)
    test = np.concatenate(test)
    # assert (len(test) == len(dataset))
    # assert (len(set(list(test))) == len(dataset))

    return dict_users, client_cls_set

def one_label(targets, num_users, data_frac=1.0, client_cls_set=[]):
    """
    Sample non-I.I.D client data.
    :param targets: dataset.targets or np.array
    :param num_users:
    :param shard_per_user: the number of classes assigned to every user, e.g. 2=each user only have samples from 2 classes.
    :return dict_users: dict{'userid': data}
    :return client_cls_set: list(each user' classes)
    """
    return noniid_label(targets, num_users, shard_per_user=1, data_frac=data_frac, client_cls_set=client_cls_set, is_uniform=False)

def get_frequncey_classes(x, n_classes):
    frequency_classes = Counter(x)
    frequency_classes = np.array([frequency_classes[i] if i in frequency_classes else 0.0 for i in range(n_classes)], dtype='float')
    frequency_classes /= frequency_classes.sum()
    return frequency_classes

def noniid_dir(targets, num_users, data_frac=1.0, alpha=1.0):
    """
    Sample non-I.I.D client data by dirichlet distribution from dataset 
    :param targets: type{list|np.array|torch.tensor}
    :param num_users:
    :param shard_per_user: the number of every user's data blocks.
    :return dict_users: dict{'userid': data}
    :return rand_set_all: list(each user' classes)
    """
    dict_users = {i:[] for i in range(num_users)}
    if type(targets) == torch.Tensor:
        targets_array = targets.numpy()
    elif type(targets) == list:
        targets_array = np.array(targets)
    else:
        targets_array = targets
    n_classes = len(np.unique(targets_array))
    n_samples = targets_array.shape[0]
    n_samples_per_user = n_samples // num_users
    ids_assigned_samples = []
    # frequency_classes = get_frequncey_classses(targets_array, n_classes)

    for u in range(num_users):
        prob_samples = torch.zeros(n_samples)
        dist = np.random.dirichlet(np.repeat(alpha, n_classes))
        for c in range(n_classes):
            ids_c = np.where(targets_array==c)[0]
            prob_samples[ids_c] = dist[c]
        prob_samples[ids_assigned_samples] = 0.0
        dict_users[u] = (torch.multinomial(prob_samples, n_samples_per_user, replacement=False)).tolist()
        random.shuffle(dict_users[u])
        ids_assigned_samples += dict_users[u]
    
    dict_frequency_classes = {}
    for u in range(num_users):
        dict_frequency_classes[u] = get_frequncey_classes(targets_array[dict_users[u]], n_classes)

    for i in dict_users.keys():
        data_num = int(len(dict_users[i]) * data_frac)
        dict_users[i] = dict_users[i][:data_num]

    return dict_users, dict_frequency_classes

def imitate_sampling(targets, num_users, dict_frequency_classes):
    dict_users = {i:[] for i in range(num_users)}
    if type(targets) == torch.Tensor:
        targets_array = targets.numpy()
    elif type(targets) == list:
        targets_array = np.array(targets)
    else:
        targets_array = targets
    n_classes = len(np.unique(targets_array))
    n_samples = targets_array.shape[0]
    n_samples_per_user = n_samples // num_users
    ids_assigned_samples = []

    for u in range(num_users):
        dist = dict_frequency_classes[u]
        for c in range(n_classes):
            prob_samples = torch.zeros(n_samples)
            ids_c = np.where(targets_array==c)[0]
            prob_samples[ids_c] = 1.0
            prob_samples[ids_assigned_samples] = 0.0
            n_samples_class = int(dist[c] * n_samples_per_user)
            try:
                if n_samples_class > 0:
                    if n_samples_class > len(ids_c) - len(ids_assigned_samples):
                        n_samples_class = len(ids_c) - len(ids_assigned_samples)
                        dict_users[u] += (torch.multinomial(prob_samples, n_samples_class, replacement=False)).tolist()
            except Exception:
                 if prob_samples[ids_c].sum() == 0:
                    raise Exception('all samples in ids_c has been sampled, you should change the random seed. May be caused by `n_samples_class = int(dist[c] * n_samples_per_user)`')
        random.shuffle(dict_users[u])
        ids_assigned_samples += dict_users[u]
    
    return dict_users
