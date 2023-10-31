
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset

from utils1 import *

from otdd.pytorch.distance import  FeatureCost
from otdd.pytorch.moments import *
from otdd.pytorch.utils import *

import os
from utils.options import get_basic_parser, add_fl_parser
from utils.data_utils import get_fed_domain_data
from utils.log_tools import generate_log_dir
from utils.global_utils import set_random_seed, resolve_args
from otdd.pytorch.distance import DatasetDistance, FeatureCost
import sys

from algs.alg_tools import check_client_class_distribution
from utils.global_utils import draw_bars
from utils.data_utils import get_fed_data, get_normal_data
from utils.global_utils import load_pkl
from torchvision.models import resnet18
import math
import random

def bary_difference(diff_list):
    df = pd.DataFrame(diff_list)
    plot = plt.figure(figsize=(10,6))
    plt.plot(df,label='Square Error',linewidth=2, marker='*')
    plt.grid()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.ylabel('Error (Approx Bary - True Bary) ',size=18)
    plt.xlabel('Iterations',size=18)
    plt.legend(fontsize=18)
    plt.savefig('/Users/muz1lee/Desktop/代码/fedselect/results/gaussian_toy.png')
    plt.show()

def load_data():
    # only MNIST datasets
    parser = add_fl_parser(get_basic_parser())
    args = parser.parse_args(['--exp_name', 'exp0_ot', '--dataset', 'mnist', '--split', 'iid'])

    args, setting_name, method_name = resolve_args(root='.', args=args)
    if not args.save_clients:
        save_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name, method_name)
    else:
        save_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name, '_grad', method_name)
    args.save_dir = generate_log_dir(path=save_dir, is_use_tb=False, has_timestamp=args.timestamp)
    setting_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name)
    # # ---------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # set random seed
    set_random_seed(seed=args.seed)

    if args.dataset in ["digits"]:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_domain_data(args.dataset,
                                                                                             args.bs_train,
                                                                                             args.data_frac)
    else:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_data(setting_dir, args.dataset,
                                                                                      args.split, args.num_users,
                                                                                      args.dir_alpha,
                                                                                      args.clsnum_peruser,
                                                                                      args.imb_alpha, args.data_frac)


    sys.path.append("/Users/muz1lee/Desktop/代码/fedselect/")

    path = "/Users/muz1lee/Desktop/代码/fedselect/exp0_ot/cifar10_dFr1.0_nUs10_iid_f1.0_e100_lEp5_s1/dict_users.pkl"
    dict_users = load_pkl(path)
    dataset = get_normal_data('mnist')
    dataset_train, dataset_test = dataset['train'], dataset['test']

    return dataset_train, dataset_test,dict_users,dict_users_train

def data_process(case):
    # only MNIST datasets
    dataset_train, dataset_test,dict_users,dict_users_train = load_data()
    # visualizations
    # results = check_client_class_distribution(dataset_train.targets, dict_users[0])
    # results = results.iloc[:, 4:]
    # draw_bars(results.to_numpy(), title="mnist")
    # df = check_client_class_distribution(dataset_train.targets, dict_users[0])
    # print(df)

    sys_path = '/Users/muz1lee/Desktop/代码/fedselect/fedewasserstein/data/'
    # print('extract test data... ')
    # path = sys_path + 'central'
    # extract_augdata(path, dataset_test)
    if case == 2:
        dict_users_train = imbalance(dataset_train.targets, 10)

    if case ==5 :
        for cls in range(10):
            indices = np.where(np.array(dataset_test.targets)==cls)[0]
            local_test = DatasetSplit(dataset_test, indices)
            testloader = torch.utils.data.DataLoader(local_test, batch_size=32, shuffle=True)
            path1 = sys_path + 'cls' + str(cls + 1)
            # path2 = sys_path +  'c' + str(cls + 1)
            # local_train = DatasetSplit(dataset_train, dict_users_train[cls])
            # trainloader = torch.utils.data.DataLoader(local_train, batch_size=32, shuffle=False)
            # extract_augdata(path2, trainloader)
            extract_augdata(path1, testloader)
    else:
        for client_idx in range(10):
            print('extract client {}th data...'.format(client_idx + 1))
            path = sys_path + 'c' + str(client_idx + 1)
            local_train = DatasetSplit(dataset_train, dict_users_train[client_idx])
            if case ==3:
                collate_fn1 = partial(collate_fn_label_noise, label_ratio=0.1 * client_idx)
                trainloader = torch.utils.data.DataLoader(local_train, batch_size=32, collate_fn= collate_fn1,shuffle=False)
            elif case ==4:
                collate_fn1 = partial(collate_fn, noise=0.1 * client_idx)
                trainloader = torch.utils.data.DataLoader(local_train, batch_size=32, collate_fn=
                collate_fn1, shuffle=False)
            else:
                trainloader = torch.utils.data.DataLoader(local_train, batch_size=32, shuffle=False)
            extract_augdata(path,trainloader)


def collate_fn(samples, noise=0): # feature noise
    # samples 是一个样本列表，每个样本可以是任意数据类型
    # 在这里可以对样本进行进一步处理和转换
    # print(f"noise={noise}")
    # 例如，假设每个样本都是一个元组 (image, label)
    images, labels = zip(*samples)

    # 对图像进行处理，例如转换为 Tensor 格式、归一化等
    images = torch.stack(images)
    if noise > 0:
        images += torch.Tensor(np.random.normal(0.0, noise, (1, images.size(-2), images.size(-1))))

    # 对标签进行处理，例如转换为 Tensor 格式
    labels = torch.tensor(labels)

    # 返回处理后的批次数据
    return images, labels
def collate_fn_label_noise(samples, label_ratio=0.0):
    # samples 是一个样本列表，每个样本可以是任意数据类型
    # 在这里可以对样本进行进一步处理和转换
    # print(f"noise={noise}")
    # 例如，假设每个样本都是一个元组 (image, label)
    images, labels = zip(*samples)

    # 对图像进行处理，例如转换为 Tensor 格式、归一化等
    images = torch.stack(images)

    # 对标签进行处理，例如转换为 Tensor格式
    labels = list(labels)
    cls_list = list(range(10))
    for i in range(len(labels)):
        # label flip
        if random.uniform(0, 1) < label_ratio:
            rnd_class = random.sample(cls_list, k=2)
            if rnd_class[0] != labels[i]:
                labels[i] = rnd_class[0]
            else:
                labels[i] = rnd_class[1]
    labels = torch.tensor(labels)  # labels是tuple类型。

    # 返回处理后的批次数据
    return images, labels
def collate_fn_label_noise(samples, label_ratio=0.0):
    # samples 是一个样本列表，每个样本可以是任意数据类型
    # 在这里可以对样本进行进一步处理和转换
    # print(f"noise={noise}")
    # 例如，假设每个样本都是一个元组 (image, label)
    images, labels = zip(*samples)

    # 对图像进行处理，例如转换为 Tensor 格式、归一化等
    images = torch.stack(images)

    # 对标签进行处理，例如转换为 Tensor格式
    labels = list(labels)
    cls_list = list(range(10))
    for i in range(len(labels)):
        # label flip
        if random.uniform(0, 1) < label_ratio:
            rnd_class = random.sample(cls_list, k=2)
            if rnd_class[0] != labels[i]:
                labels[i] = rnd_class[0]
            else:
                labels[i] = rnd_class[1]
    labels = torch.tensor(labels)  # labels是tuple类型。

    # 返回处理后的批次数据
    return images, labels

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
        for i in range(num_users):
            ratio = 0.05 + i // 2 * 0.025
            num_items_per_cls = math.floor(len(targets_array[targets_array==c]) * ratio)
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

def cal_dis():
    dataset_train, dataset_test,dict_users,dict_users_train = load_data()
    results = check_client_class_distribution(dataset_train.targets, dict_users[0])
    results = results.iloc[:, 4:]
    draw_bars(results.to_numpy(), title="mnist")
    df = check_client_class_distribution(dataset_train.targets, dict_users[0])
    print(df)

    # Embed using a pretrained (+frozen) resnet
    embedder = resnet18(pretrained=True).eval()
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    feature_cost = FeatureCost(src_embedding=embedder,
                               src_dim=(3, 32, 32),
                               tgt_embedding=embedder,
                               tgt_dim=(3, 32, 32),
                               p=2,
                               device='-1')
    ot_results = []
    for client_idx in range(2):
        print('client_idx', client_idx)
        local_train = DatasetSplit(dataset_train, dict_users_train[client_idx])
        trainloader = torch.utils.data.DataLoader(local_train, batch_size=32, shuffle=False)
        dist = DatasetDistance(trainloader, dataset_test,
                               inner_ot_method='gaussian_approx',
                               debiased_loss=True,
                            feature_cost=feature_cost,
                               sqrt_method='spectral',
                               sqrt_niters=10,
                               precision='single',
                               p=2, entreg=1e-1,
                               device='cpu')
        ot_results.append(dist.distance())

        return cal_dis
# corrupt data for detections
def load_data_corrupted(corrupt_type='shuffle', dataname=None, data=None, valid_size=0, random_seed=2021, resize=None,
                                        stratified=True, shuffle=False,
                                        training_size=None, test_size=None, currupt_por=0):
    if corrupt_type == 'shuffle':
        loaders, full_dict, shuffle_ind = load_torchvision_data_shuffle(dataname, valid_size=valid_size,
                                                                        random_seed=random_seed,
                                                                        resize=resize, stratified=stratified,
                                                                        shuffle=shuffle, maxsize=training_size,
                                                                        maxsize_test=test_size, shuffle_per=currupt_por)
        return loaders, shuffle_ind
    # elif corrupt_type == 'feature':
    # elif corrupt_type == 'backdoor-blend', 'backdoor-trojan-sq', 'backdoor-trojan-wm'
    else:  # empty or non-implemented == Loading Clean Data
        shuffle_ind = []
        loaders, full_dict = load_torchvision_data_shuffle(dataname, valid_size=valid_size, random_seed=random_seed,
                                                           resize=resize, stratified=stratified, shuffle=shuffle,
                                                           maxsize=training_size, maxsize_test=test_size, shuffle_per=0)
        return loaders, shuffle_ind


def extract_augdata(path,data):
    targets1, classes1, idxs1 = extract_data_targets(data)
    vals1, cts1 = torch.unique(targets1, return_counts=True)
    min_labelcount = 2
    V1 = torch.sort(vals1[cts1 >= min_labelcount])[0]
    X1, Y1 = load_full_dataset(data, targets=True,
                               labels_keep=V1,
                               maxsamples=10000,
                               device='cpu',
                               dtype=torch.FloatTensor,
                               reindex=True,
                               reindex_start=0)
    DA = (X1, Y1)

    M1, C1 = compute_label_stats(data, targets1, idxs1, classes1, diagonal_cov=True)
    XA = augmented_dataset(DA, means=M1, covs=C1, maxn=10000)

    # you can choose whether to save
    # np.save(path + 'mean.npy', M1.numpy())
    # np.save(path + 'cov.npy', C1.numpy())
    np.save(path + 'xa.npy', XA.numpy())

class DatasetSplit(Dataset):
    """
        数据集和Dataloader之间的接口。
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.LongTensor(self.dataset.targets)[idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
